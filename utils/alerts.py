"""Sending alerts: the deployment's SNS topic, and a monitor's own chat
webhook.

- `ALERTS_TOPIC_ARN` set: every alert is published to that Amazon SNS
  topic. Operators subscribe email addresses to it, or connect Slack or
  Microsoft Teams through AWS Chatbot.
- A monitor may also name an incoming webhook (Slack or Microsoft Teams).
  Only HTTPS URLs on those services' webhook hosts are accepted: the server
  posts to the URL, so an arbitrary one would let a user make it call
  internal addresses.

Sending never raises: a failed alert is logged and reported back, and the
check that raised it still completes.
"""
import json
import logging
import os
import urllib.request
from urllib.parse import urlparse

from utils import log

_log = log.get("alerts")

WEBHOOK_HOSTS = ("hooks.slack.com",)
WEBHOOK_SUFFIXES = (".webhook.office.com", ".logic.azure.com")


def webhook_problem(url):
    """None when `url` is an acceptable webhook, else why not."""
    if not url:
        return None
    try:
        u = urlparse(url)
    except ValueError:
        return "Not a URL."
    host = (u.hostname or "").lower()
    if u.scheme != "https":
        return "The webhook must use https."
    if host not in WEBHOOK_HOSTS and not host.endswith(WEBHOOK_SUFFIXES):
        return ("Only Slack (hooks.slack.com) and Microsoft Teams "
                "(*.webhook.office.com) webhooks are accepted.")
    return None


def _sns(subject, message):
    topic = os.environ.get("ALERTS_TOPIC_ARN")
    if not topic:
        return None
    import boto3
    boto3.client("sns").publish(TopicArn=topic, Subject=subject[:100],
                                Message=message)
    return "sns"


def _webhook(url, subject, message):
    if webhook_problem(url):
        raise ValueError("webhook not allowed")
    text = f"*{subject}*\n{message}"
    body = json.dumps({"text": text}).encode()
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        resp.read()
    return "webhook"


def send(subject, message, webhook=None, sender=None):
    """Deliver an alert everywhere configured; the channels it reached.
    `sender` replaces the real delivery in tests."""
    reached = []
    channels = [("sns", lambda: _sns(subject, message))]
    if webhook:
        channels.append(("webhook", lambda: _webhook(webhook, subject,
                                                     message)))
    for name, deliver in channels:
        try:
            got = (sender(name, subject, message, webhook)
                   if sender is not None else deliver())
            if got:
                reached.append(name)
        except Exception:
            log.event(_log, "alert_failed", level=logging.WARNING,
                      exc_info=True, channel=name)
    log.event(_log, "alert_sent", channels=len(reached))
    return reached
