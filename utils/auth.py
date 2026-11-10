"""Who is signed in, and what they may do.

In production (`AUTH_MODE=alb`) the Application Load Balancer authenticates
every request with Amazon Cognito before it reaches the app, and forwards:

- `x-amzn-oidc-data`: the user's claims, as a JWT the ALB signs (ES256). It
  is verified here — signature, the signing load balancer (`ALB_ARN`) and
  expiry — so a request that reaches a task without passing the ALB gets
  nothing.
- `x-amzn-oidc-accesstoken`: Cognito's access token, which carries the
  user's groups (`cognito:groups`). Verified against the user pool's keys.

Roles are Cognito groups. A user in several gets the highest; one in none
gets `DEFAULT_ROLE` (viewer unless set). Permissions are by role:

    viewer   load data, explore, drift analysis, view results
    analyst  + change data, feature engineering, train, AI features,
               use own models, Snowflake
    admin    + administration: the audit log across every project

For local work, `AUTH_MODE=dev` signs in a development user (role from
`DEV_ROLE`, admin by default) and offers a role switcher to test what each
role sees.
"""
import base64
import functools
import json
import logging
import os
import time
import urllib.request

import streamlit as st

from utils import log

_log = log.get("auth")

ROLES = ("viewer", "analyst", "admin")
PERMISSIONS = {
    "edit": "analyst",        # preparation steps, feature engineering
    "train": "analyst",
    "ai": "analyst",          # Claude-backed features (they cost money)
    "own_model": "analyst",   # upload a model file
    "snowflake": "analyst",   # connect, read and write
    "admin": "admin",
}
WHAT = {
    "edit": "change the data",
    "train": "train models",
    "ai": "use the AI features",
    "own_model": "use your own models",
    "snowflake": "connect to Snowflake",
    "admin": "use administration (the audit log)",
}
KEY = "auth_user"            # not ml_*: survives loading a new file
DEV_ROLE_KEY = "auth_dev_role"


class AuthError(Exception):
    """A request that cannot be trusted, in words safe to log."""


def mode():
    return os.environ.get("AUTH_MODE", "dev").strip().lower()


# ── verifying the load balancer's headers ───────────────────────────────
def _b64(segment):
    return base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4))


@functools.lru_cache(maxsize=16)
def _alb_key(region, kid):
    """The ALB's public key for `kid`, in PEM. Cached: keys rotate rarely
    and each fetch is a network round trip."""
    if not kid.replace("-", "").isalnum():
        raise AuthError("malformed key id")
    url = f"https://public-keys.auth.elb.{region}.amazonaws.com/{kid}"
    with urllib.request.urlopen(url, timeout=5) as resp:
        return resp.read()


def verify_alb_data(token, *, region, alb_arn, now=None, key_loader=None):
    """The claims in an `x-amzn-oidc-data` token, after checking the
    signature (ES256, raw r||s as the ALB signs), the signer and expiry.

    Verified by hand rather than with a JWT library: the ALB's tokens are
    standard base64 with padding in places, which strict decoders refuse."""
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.asymmetric.utils import \
        encode_dss_signature
    try:
        head_b64, body_b64, sig_b64 = token.split(".")
        header = json.loads(_b64(head_b64))
        claims = json.loads(_b64(body_b64))
        sig = _b64(sig_b64)
    except (ValueError, TypeError):
        raise AuthError("malformed token") from None
    if header.get("alg") != "ES256":
        raise AuthError("unexpected algorithm")
    if header.get("signer") != alb_arn:
        raise AuthError("signed by another load balancer")
    pem = (key_loader or _alb_key)(region, header.get("kid", ""))
    key = serialization.load_pem_public_key(pem)
    if len(sig) != 64:
        raise AuthError("malformed signature")
    der = encode_dss_signature(int.from_bytes(sig[:32], "big"),
                               int.from_bytes(sig[32:], "big"))
    try:
        key.verify(der, f"{head_b64}.{body_b64}".encode(),
                   ec.ECDSA(hashes.SHA256()))
    except Exception:
        raise AuthError("bad signature") from None
    exp = header.get("exp") or claims.get("exp")
    if exp is not None and float(exp) < (now or time.time()):
        raise AuthError("expired")
    return claims


@functools.lru_cache(maxsize=4)
def _jwks_client(url):
    import jwt
    return jwt.PyJWKClient(url, cache_keys=True, lifespan=3600)


def verify_cognito_access(token, *, region, pool_id, client_id,
                          jwks_client=None):
    """The claims of a Cognito access token, verified against the pool's
    keys, issuer, token use and app client."""
    import jwt
    issuer = f"https://cognito-idp.{region}.amazonaws.com/{pool_id}"
    client = jwks_client or _jwks_client(f"{issuer}/.well-known/jwks.json")
    try:
        key = client.get_signing_key_from_jwt(token).key
        claims = jwt.decode(token, key, algorithms=["RS256"], issuer=issuer,
                            options={"verify_aud": False})
    except Exception:
        raise AuthError("bad access token") from None
    if claims.get("token_use") != "access":
        raise AuthError("not an access token")
    if claims.get("client_id") != client_id:
        raise AuthError("issued to another app client")
    return claims


def role_for(groups, default="viewer"):
    """The highest role among `groups`; `default` when none is a role."""
    held = [r for r in ROLES if r in set(groups or ())]
    return held[-1] if held else (default if default in ROLES else "viewer")


def _from_alb(headers):
    """The user from the load balancer's headers, or raise AuthError."""
    env = os.environ
    region, alb_arn = env.get("AWS_REGION"), env.get("ALB_ARN")
    if not region or not alb_arn:
        raise AuthError("AUTH_MODE=alb needs AWS_REGION and ALB_ARN")
    data = headers.get("x-amzn-oidc-data")
    if not data:
        raise AuthError("no identity header")
    claims = verify_alb_data(data, region=region, alb_arn=alb_arn)
    groups = []
    access = headers.get("x-amzn-oidc-accesstoken")
    pool, client = env.get("COGNITO_USER_POOL_ID"), env.get(
        "COGNITO_CLIENT_ID")
    if access and pool and client:
        access_claims = verify_cognito_access(access, region=region,
                                              pool_id=pool, client_id=client)
        groups = list(access_claims.get("cognito:groups") or [])
    email = claims.get("email") or ""
    return {"sub": claims.get("sub") or headers.get("x-amzn-oidc-identity"),
            "email": email,
            "name": claims.get("name") or claims.get("given_name")
            or email.split("@")[0] or "Signed-in user",
            "groups": groups,
            "role": role_for(groups, env.get("DEFAULT_ROLE", "viewer")),
            "expires": float(claims.get("exp") or 0) or None}


def _dev_user():
    ss = st.session_state
    role = ss.get(DEV_ROLE_KEY) or os.environ.get("DEV_ROLE", "admin")
    email = os.environ.get("DEV_USER_EMAIL", "dev@localhost")
    return {"sub": "dev", "email": email,
            "name": os.environ.get("DEV_USER_NAME", "Local developer"),
            "groups": [role], "role": role_for([role]), "expires": None}


# ── the session's user ──────────────────────────────────────────────────
def current_user():
    """The signed-in user, or None when the request cannot be trusted.
    Verified once per browser session, again when the ALB's token
    expires."""
    ss = st.session_state
    if mode() == "dev":
        return _dev_user()
    user = ss.get(KEY)
    if user and (not user.get("expires") or user["expires"] > time.time()):
        return user
    try:
        user = _from_alb({k.lower(): v for k, v in st.context.headers
                          .items()})
    except AuthError as e:
        log.event(_log, "auth_rejected", level=logging.WARNING,
                  reason=str(e))
        ss.pop(KEY, None)
        return None
    except Exception:
        log.event(_log, "auth_failed", level=logging.ERROR, exc_info=True)
        ss.pop(KEY, None)
        return None
    ss[KEY] = user
    log.event(_log, "auth_signed_in", role=user["role"],
              groups=len(user["groups"]))
    return user


def user():
    return current_user() or {}


def role():
    return user().get("role", "viewer")


def can(permission):
    need = PERMISSIONS[permission]
    return ROLES.index(role()) >= ROLES.index(need)


def why_not(permission):
    """A tooltip for a control the user's role cannot use, or None."""
    if can(permission):
        return None
    return (f"Your role ({role()}) cannot {WHAT[permission]}. Ask an admin "
            f"for the {PERMISSIONS[permission]} role.")


def denied(permission):
    """Say why a section is unavailable; True when it is."""
    if can(permission):
        return False
    st.info(why_not(permission))
    return True


# ── pages ───────────────────────────────────────────────────────────────
def render_signed_out():
    st.markdown("## Sign-in required")
    st.warning("This request did not come through the sign-in service, or "
               "the sign-in has expired. Reload the page to sign in again.")
    url = os.environ.get("SIGN_OUT_URL")
    if url:
        st.link_button("Sign in again", url)


def account_menu():
    """Who is signed in, their role, sign-out — and in dev mode a role
    switcher to see each role's view."""
    u = user()
    if not u:
        return
    with st.popover(f"{u['name']} · {u['role']}", width="stretch",
                    icon=":material/account_circle:"):
        st.markdown(f"**{u['name']}**  \n{u['email']}")
        st.caption(f"Role: {u['role']}"
                   + (f" · groups: {', '.join(u['groups'])}"
                      if u["groups"] and mode() != "dev" else ""))
        if mode() == "dev":
            st.caption("Development sign-in (AUTH_MODE=dev).")
            st.session_state.setdefault(DEV_ROLE_KEY, u["role"])
            st.selectbox("View as", ROLES, key=DEV_ROLE_KEY)
        url = os.environ.get("SIGN_OUT_URL")
        if url:
            st.link_button("Sign out", url, width="stretch")
