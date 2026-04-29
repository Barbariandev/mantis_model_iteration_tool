"""MANTIS V2 payload encryption.

Generates timelock-encrypted payloads matching the subnet's
generate_and_encrypt.py V2 format. Requires the `timelock` and
`cryptography` packages.

The payload is double-wrapped:
  1. Inner: ChaCha20-Poly1305 with a random key, binding = SHA256(hk:round:owner_pk:pke)
  2. W_owner: X25519 ECDH shared secret wraps the inner key for the subnet owner
  3. W_time: Drand timelock encrypts (ske || key) so validators can decrypt after the round
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import time
from typing import Any

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
try:
    from timelock import Timelock
except ImportError:
    Timelock = None

logger = logging.getLogger(__name__)

DRAND_API = "https://api.drand.sh/v2"
DRAND_BEACON_ID = "quicknet"
DRAND_PUBLIC_KEY = (
    "83cf0f2896adee7eb8b5f01fcad3912212c437e0073e911fb90022d3e760183c"
    "8c4b450b6a0a6c3ac6a5776a2d1064510d1fec758c921cc22b0e17e63aaf4bcb"
    "5ed66304de9cf809bd274ca73bab4af5a6e9c76a4bc09e76eae8991ef5ece45a"
)
OWNER_HPKE_PUBLIC_KEY_HEX = (
    "fbfe185ded7a4e6865effceb23cbac32894170587674e751ac237a06f72b3067"
)
ALG_LABEL_V2 = "x25519-hkdf-sha256+chacha20poly1305+drand-tlock"

SUBNET_CHALLENGES = [
    {"name": "ETH-1H-BINARY", "ticker": "ETH", "dim": 2, "loss_func": "binary"},
    {"name": "ETH-HITFIRST-100M", "ticker": "ETHHITFIRST", "dim": 3, "loss_func": "hitfirst"},
    {"name": "ETH-LBFGS", "ticker": "ETHLBFGS", "dim": 17, "loss_func": "lbfgs"},
    {"name": "BTC-LBFGS-6H", "ticker": "BTCLBFGS", "dim": 17, "loss_func": "lbfgs"},
    {"name": "CADUSD-1H-BINARY", "ticker": "CADUSD", "dim": 2, "loss_func": "binary"},
    {"name": "NZDUSD-1H-BINARY", "ticker": "NZDUSD", "dim": 2, "loss_func": "binary"},
    {"name": "CHFUSD-1H-BINARY", "ticker": "CHFUSD", "dim": 2, "loss_func": "binary"},
    {"name": "XAGUSD-1H-BINARY", "ticker": "XAGUSD", "dim": 2, "loss_func": "binary"},
    {"name": "MULTI-BREAKOUT", "ticker": "MULTIBREAKOUT", "dim": 2, "loss_func": "range_breakout_multi",
     "assets": [
         "BTC", "ETH", "XRP", "SOL", "TRX", "DOGE", "ADA", "BCH", "XMR",
         "LINK", "LEO", "HYPE", "XLM", "ZEC", "SUI", "LTC", "AVAX", "HBAR", "SHIB",
         "TON", "CRO", "DOT", "UNI", "MNT", "BGB", "TAO", "AAVE", "PEPE",
         "NEAR", "ICP", "ETC", "ONDO", "SKY",
     ]},
    {"name": "XSEC-RANK", "ticker": "MULTIXSEC", "dim": 1, "loss_func": "xsec_rank",
     "assets": [
         "BTC", "ETH", "XRP", "SOL", "TRX", "DOGE", "ADA", "BCH", "XMR",
         "LINK", "LEO", "HYPE", "XLM", "ZEC", "SUI", "LTC", "AVAX", "HBAR", "SHIB",
         "TON", "CRO", "DOT", "UNI", "MNT", "BGB", "TAO", "AAVE", "PEPE",
         "NEAR", "ICP", "ETC", "ONDO", "SKY",
     ]},
    {"name": "FUNDING-XSEC", "ticker": "FUNDINGXSEC", "dim": 1, "loss_func": "funding_xsec",
     "assets": [
         "BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "AVAX", "LINK", "DOT", "SUI",
         "NEAR", "AAVE", "UNI", "LTC", "HBAR", "PEPE", "TRX", "SHIB", "TAO", "ONDO",
     ]},
]

NAME_TO_TICKER = {c["name"]: c["ticker"] for c in SUBNET_CHALLENGES}


def _target_round(lock_seconds: int = 30) -> int:
    """Compute the Drand round roughly lock_seconds in the future."""
    resp = requests.get(
        f"{DRAND_API}/beacons/{DRAND_BEACON_ID}/info", timeout=10,
    )
    resp.raise_for_status()
    info = resp.json()
    future_time = time.time() + lock_seconds
    return int((future_time - info["genesis_time"]) // info["period"])


def _hkdf_key_nonce(
    shared_secret: bytes, info: bytes,
    key_len: int = 32, nonce_len: int = 12,
) -> tuple[bytes, bytes]:
    out = HKDF(
        algorithm=hashes.SHA256(),
        length=key_len + nonce_len,
        salt=None,
        info=info,
    ).derive(shared_secret)
    return out[:key_len], out[key_len:]


def _binding(hk: str, rnd: int, owner_pk: bytes, pke: bytes) -> bytes:
    h = hashes.Hash(hashes.SHA256())
    h.update(hk.encode("utf-8"))
    h.update(b":")
    h.update(str(rnd).encode("ascii"))
    h.update(b":")
    h.update(owner_pk)
    h.update(b":")
    h.update(pke)
    return h.finalize()


def _derive_pke(ske_raw: bytes) -> bytes:
    return X25519PrivateKey.from_private_bytes(ske_raw).public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def build_plaintext(
    hotkey: str,
    embeddings: dict[str, Any],
) -> dict[str, Any]:
    """Build the V2 plaintext object keyed by subnet ticker.

    embeddings should be {ticker: list_or_dict, ...} matching the
    subnet's CHALLENGES order.  Missing tickers are filled with zeros.
    """
    obj: dict[str, Any] = {}
    for c in SUBNET_CHALLENGES:
        ticker = c["ticker"]
        if ticker in embeddings:
            obj[ticker] = embeddings[ticker]
        elif c.get("assets"):
            if c["dim"] == 2:
                obj[ticker] = {a: [0.5, 0.5] for a in c["assets"]}
            elif c["dim"] == 1:
                obj[ticker] = {a: 0.0 for a in c["assets"]}
            else:
                obj[ticker] = {a: [0.0] * c["dim"] for a in c["assets"]}
        else:
            obj[ticker] = [0.0] * c["dim"]
    obj["hotkey"] = hotkey
    return obj


def encrypt_v2(
    hotkey: str,
    embeddings: dict[str, Any],
    lock_seconds: int = 30,
    owner_pk_hex: str = OWNER_HPKE_PUBLIC_KEY_HEX,
) -> dict[str, Any]:
    """Generate a V2 timelock-encrypted payload.

    Returns the JSON-serialisable dict ready to be written to R2.
    """
    owner_pk = bytes.fromhex(owner_pk_hex)
    obj = build_plaintext(hotkey, embeddings)
    pt = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    rnd = _target_round(lock_seconds)
    ske = os.urandom(32)
    key = os.urandom(32)
    pke = _derive_pke(ske)
    bind = _binding(hotkey, rnd, owner_pk, pke)

    c_nonce = os.urandom(12)
    c_ct = ChaCha20Poly1305(key).encrypt(c_nonce, pt, bind)

    shared = X25519PrivateKey.from_private_bytes(ske).exchange(
        X25519PublicKey.from_public_bytes(owner_pk),
    )
    wrap_key, _ = _hkdf_key_nonce(shared, info=b"mantis-owner-wrap")
    wrap_nonce = os.urandom(12)
    w_owner_ct = ChaCha20Poly1305(wrap_key).encrypt(wrap_nonce, key, bind)

    if Timelock is None:
        raise ImportError(
            "timelock package is required for encryption but not installed. "
            "Install from https://github.com/ideal-lab5/timelock"
        )
    tlock = Timelock(DRAND_PUBLIC_KEY)
    combined_hex = (ske + key).hex()
    w_time_ct = tlock.tle(rnd, combined_hex, secrets.token_bytes(32))

    payload = {
        "v": 2,
        "round": rnd,
        "hk": hotkey,
        "owner_pk": owner_pk_hex,
        "C": {"nonce": c_nonce.hex(), "ct": c_ct.hex()},
        "W_owner": {
            "pke": pke.hex(),
            "nonce": wrap_nonce.hex(),
            "ct": w_owner_ct.hex(),
        },
        "W_time": {"ct": w_time_ct.hex()},
        "binding": bind.hex(),
        "alg": ALG_LABEL_V2,
    }
    logger.info("Encrypted V2 payload for round %d (hotkey=%s…)", rnd, hotkey[:12])
    return payload
