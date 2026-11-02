"""The training worker: runs background jobs from the SQS queue.

    python -m worker.trainer

Long-polls `JOBS_QUEUE_URL`. For each message it checks the job in the
database (the source of truth) and:

- runs it when it is open (queued, or interrupted elsewhere), keeping the
  message hidden while it works and asking ECS not to scale this task in
  until it finishes (task scale-in protection);
- leaves the message to come back later when another worker is actively
  running the job;
- deletes the message when the job is done.

A message that keeps failing to be handled lands in the dead-letter queue.
Once a minute the worker also re-sends interrupted jobs whose message was
lost, so nothing waits forever. SIGTERM stops it after the current job.
"""
import json
import logging
import os
import signal
import threading
import time
import urllib.request

from logic import jobs
from utils import db, log

_log = log.get("trainer")

VISIBILITY = 300          # seconds a received message stays hidden
KEEP_ALIVE = 120          # how often it is hidden again while a job runs
RESCAN = 60               # seconds between checks for interrupted jobs
PROTECT_MINUTES = 240     # longest a job may hold off a scale-in


def _protect(enabled):
    """ECS task scale-in protection, when running under ECS."""
    agent = os.environ.get("ECS_AGENT_URI")
    if not agent:
        return
    body = {"ProtectionEnabled": enabled}
    if enabled:
        body["ExpiresInMinutes"] = PROTECT_MINUTES
    req = urllib.request.Request(f"{agent}/task-protection/v1/state",
                                 data=json.dumps(body).encode(),
                                 method="PUT",
                                 headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=5).read()
    except Exception:
        log.event(_log, "task_protection_failed", level=logging.WARNING,
                  exc_info=True, enabled=enabled)


def _keep_hidden(sqs, queue_url, receipt, done):
    while not done.wait(KEEP_ALIVE):
        try:
            sqs.change_message_visibility(QueueUrl=queue_url,
                                          ReceiptHandle=receipt,
                                          VisibilityTimeout=VISIBILITY)
        except Exception:
            log.event(_log, "keep_hidden_failed", level=logging.WARNING,
                      exc_info=True)


def handle(sqs, queue_url, message):
    """One message: run, defer or drop. Returns what happened."""
    receipt = message["ReceiptHandle"]
    try:
        job_id = json.loads(message["Body"])["job_id"]
    except (ValueError, KeyError, TypeError):
        sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt)
        return "malformed"
    what = jobs.state(job_id)
    if what == "busy":
        # Someone else is on it; look again after a while.
        sqs.change_message_visibility(QueueUrl=queue_url,
                                      ReceiptHandle=receipt,
                                      VisibilityTimeout=VISIBILITY)
        return "deferred"
    if what == "open":
        done = threading.Event()
        keeper = threading.Thread(target=_keep_hidden, daemon=True,
                                  args=(sqs, queue_url, receipt, done))
        keeper.start()
        _protect(True)
        try:
            jobs.run_job(job_id)
        finally:
            done.set()
            _protect(False)
    sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=receipt)
    return "ran" if what == "open" else "dropped"


def main():
    log.setup()
    queue_url = os.environ["JOBS_QUEUE_URL"]
    db.migrate()
    import boto3
    sqs = boto3.client("sqs")
    stopping = threading.Event()
    signal.signal(signal.SIGTERM, lambda *a: stopping.set())
    last_scan = 0.0
    log.event(_log, "trainer_started")
    while not stopping.is_set():
        if time.monotonic() - last_scan > RESCAN:
            last_scan = time.monotonic()
            for job_id in jobs.stale_jobs():
                jobs.dispatch(job_id)
        got = sqs.receive_message(QueueUrl=queue_url, MaxNumberOfMessages=1,
                                  WaitTimeSeconds=20,
                                  VisibilityTimeout=VISIBILITY)
        for message in got.get("Messages", []):
            outcome = handle(sqs, queue_url, message)
            log.event(_log, "message_handled", outcome=outcome)
    log.event(_log, "trainer_stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
