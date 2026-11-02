"""Sign-out, behind the ALB rule for /logout.

The ALB keeps its own session cookie (AWSELBAuthSessionCookie-0, -1, ...,
HttpOnly), so neither the app nor the browser's scripts can clear it. This
expires every shard of it and sends the browser to Cognito's logout, which
ends the Cognito session and returns to the app — where the ALB asks the
user to sign in again.
"""
import os

SHARDS = 4   # the ALB splits a large session cookie into numbered parts
EXPIRED = "Expires=Thu, 01 Jan 1970 00:00:00 GMT; Path=/; Secure; HttpOnly"


def handler(event, context):
    cookies = [f"AWSELBAuthSessionCookie-{i}=deleted; {EXPIRED}"
               for i in range(SHARDS)]
    return {"statusCode": 302,
            "statusDescription": "302 Found",
            "isBase64Encoded": False,
            "multiValueHeaders": {
                "Location": [os.environ["LOGOUT_URL"]],
                "Set-Cookie": cookies,
                "Cache-Control": ["no-store"]},
            "body": ""}
