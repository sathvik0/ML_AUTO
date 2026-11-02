"""The sign-in checks, against tokens built the way the ALB and Cognito
build them — signed here with throwaway keys."""
import base64
import json
import time
import types

import jwt
import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from cryptography.hazmat.primitives.asymmetric.utils import \
    decode_dss_signature

from utils import auth

REGION = "us-east-1"
ALB = "arn:aws:elasticloadbalancing:us-east-1:123456789012:loadbalancer/app/ml/abc"
POOL, CLIENT = "us-east-1_TESTPOOL", "test-client-id"


def _b64(raw, pad=True):
    out = base64.urlsafe_b64encode(raw).decode()
    return out if pad else out.rstrip("=")


def alb_token(claims, *, key, signer=ALB, alg="ES256", exp=None):
    """An x-amzn-oidc-data token as the ALB signs it: ES256 with a raw
    r||s signature, and base64 that keeps its padding."""
    header = {"alg": alg, "kid": "kid-1", "signer": signer,
              "iss": "https://cognito", "client": CLIENT,
              "exp": exp or int(time.time()) + 600}
    h = _b64(json.dumps(header).encode())
    b = _b64(json.dumps(claims).encode())
    der = key.sign(f"{h}.{b}".encode(), ec.ECDSA(hashes.SHA256()))
    r, s = decode_dss_signature(der)
    return f"{h}.{b}.{_b64(r.to_bytes(32, 'big') + s.to_bytes(32, 'big'))}"


@pytest.fixture
def ec_key():
    return ec.generate_private_key(ec.SECP256R1())


def loader_for(key):
    pem = key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo)
    return lambda region, kid: pem


CLAIMS = {"sub": "abc-123", "email": "ana@example.com", "name": "Ana"}


def test_alb_token_verifies(ec_key):
    token = alb_token(CLAIMS, key=ec_key)
    got = auth.verify_alb_data(token, region=REGION, alb_arn=ALB,
                               key_loader=loader_for(ec_key))
    assert got["email"] == "ana@example.com"


@pytest.mark.parametrize("change", ["tampered", "other_alb", "expired",
                                    "wrong_alg", "other_key"])
def test_alb_token_rejected(ec_key, change):
    kwargs = {"key": ec_key}
    if change == "other_alb":
        kwargs["signer"] = ALB.replace("abc", "evil")
    if change == "expired":
        kwargs["exp"] = int(time.time()) - 5
    if change == "wrong_alg":
        kwargs["alg"] = "none"
    token = alb_token(CLAIMS, **kwargs)
    if change == "tampered":
        h, _, s = token.split(".")
        forged = _b64(json.dumps({**CLAIMS, "email": "boss@example.com"})
                      .encode())
        token = f"{h}.{forged}.{s}"
    loader = loader_for(ec.generate_private_key(ec.SECP256R1())
                        if change == "other_key" else ec_key)
    with pytest.raises(auth.AuthError):
        auth.verify_alb_data(token, region=REGION, alb_arn=ALB,
                             key_loader=loader)


def test_malformed_alb_token():
    with pytest.raises(auth.AuthError):
        auth.verify_alb_data("not-a-jwt", region=REGION, alb_arn=ALB,
                             key_loader=lambda *a: b"")


@pytest.fixture
def rsa_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def access_token(key, **over):
    claims = {"sub": "abc-123", "token_use": "access", "client_id": CLIENT,
              "cognito:groups": ["analyst"],
              "iss": f"https://cognito-idp.{REGION}.amazonaws.com/{POOL}",
              "exp": int(time.time()) + 600, **over}
    return jwt.encode(claims, key, algorithm="RS256",
                      headers={"kid": "k1"})


def jwks_for(key):
    return types.SimpleNamespace(get_signing_key_from_jwt=lambda t:
                                 types.SimpleNamespace(key=key.public_key()))


def test_cognito_access_token(rsa_key):
    got = auth.verify_cognito_access(access_token(rsa_key), region=REGION,
                                     pool_id=POOL, client_id=CLIENT,
                                     jwks_client=jwks_for(rsa_key))
    assert got["cognito:groups"] == ["analyst"]


@pytest.mark.parametrize("over", [{"token_use": "id"},
                                  {"client_id": "another-app"},
                                  {"iss": "https://evil.example.com"},
                                  {"exp": int(time.time()) - 5}])
def test_cognito_access_token_rejected(rsa_key, over):
    with pytest.raises(auth.AuthError):
        auth.verify_cognito_access(access_token(rsa_key, **over),
                                   region=REGION, pool_id=POOL,
                                   client_id=CLIENT,
                                   jwks_client=jwks_for(rsa_key))


def test_cognito_token_from_another_key(rsa_key):
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    with pytest.raises(auth.AuthError):
        auth.verify_cognito_access(access_token(other), region=REGION,
                                   pool_id=POOL, client_id=CLIENT,
                                   jwks_client=jwks_for(rsa_key))


@pytest.mark.parametrize("groups,default,want", [
    (["viewer", "admin"], "viewer", "admin"),
    (["analyst"], "viewer", "analyst"),
    (["marketing"], "viewer", "viewer"),
    ([], "analyst", "analyst"),
    ([], "superuser", "viewer"),
])
def test_role_for(groups, default, want):
    assert auth.role_for(groups, default) == want


def test_from_alb_end_to_end(ec_key, rsa_key, monkeypatch):
    monkeypatch.setenv("AWS_REGION", REGION)
    monkeypatch.setenv("ALB_ARN", ALB)
    monkeypatch.setenv("COGNITO_USER_POOL_ID", POOL)
    monkeypatch.setenv("COGNITO_CLIENT_ID", CLIENT)
    monkeypatch.setattr(auth, "_alb_key", loader_for(ec_key))
    monkeypatch.setattr(auth, "_jwks_client", lambda url: jwks_for(rsa_key))
    user = auth._from_alb({
        "x-amzn-oidc-data": alb_token(CLAIMS, key=ec_key),
        "x-amzn-oidc-accesstoken": access_token(
            rsa_key, **{"cognito:groups": ["viewer", "analyst"]})})
    assert (user["email"], user["role"]) == ("ana@example.com", "analyst")


def test_from_alb_without_header(monkeypatch):
    monkeypatch.setenv("AWS_REGION", REGION)
    monkeypatch.setenv("ALB_ARN", ALB)
    with pytest.raises(auth.AuthError):
        auth._from_alb({})
