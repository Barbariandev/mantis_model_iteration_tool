"""Cloudflare R2 upload for MANTIS miner payloads.

Uses boto3's S3-compatible API to PUT encrypted payload JSON to an R2
bucket.  The object key is always the hotkey SS58 address (the
validator enforces this).

Requires either:
  - A full-permission Cloudflare API token, OR
  - An API token scoped to a specific R2 bucket with read/write

Configuration is passed via R2Config dataclass or environment variables.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import boto3
from botocore.config import Config as BotoConfig

logger = logging.getLogger(__name__)


@dataclass
class R2Config:
    account_id: str = ""
    access_key_id: str = ""
    secret_access_key: str = ""
    bucket_name: str = ""
    public_base_url: str = ""

    @classmethod
    def from_env(cls) -> "R2Config":
        return cls(
            account_id=os.environ.get("R2_ACCOUNT_ID", ""),
            access_key_id=os.environ.get("R2_ACCESS_KEY_ID", ""),
            secret_access_key=os.environ.get("R2_SECRET_ACCESS_KEY", ""),
            bucket_name=os.environ.get("R2_BUCKET_NAME", ""),
            public_base_url=os.environ.get("R2_PUBLIC_BASE_URL", ""),
        )

    def validate(self) -> list[str]:
        missing = []
        if not self.account_id:
            missing.append("account_id")
        if not self.access_key_id:
            missing.append("access_key_id")
        if not self.secret_access_key:
            missing.append("secret_access_key")
        if not self.bucket_name:
            missing.append("bucket_name")
        return missing

    @property
    def endpoint_url(self) -> str:
        return f"https://{self.account_id}.r2.cloudflarestorage.com"


class R2Client:
    """Thin wrapper around boto3 S3 client for Cloudflare R2."""

    def __init__(self, config: R2Config):
        missing = config.validate()
        if missing:
            raise ValueError(f"R2Config missing required fields: {missing}")
        self._config = config
        self._client = boto3.client(
            "s3",
            endpoint_url=config.endpoint_url,
            aws_access_key_id=config.access_key_id,
            aws_secret_access_key=config.secret_access_key,
            config=BotoConfig(
                retries={"max_attempts": 3, "mode": "adaptive"},
                connect_timeout=10,
                read_timeout=30,
            ),
        )

    def upload_payload(self, hotkey: str, payload: dict[str, Any]) -> str:
        """Upload an encrypted payload JSON to R2.

        Object key = hotkey (no directory prefix).
        Returns the public URL of the uploaded object.
        """
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self._client.put_object(
            Bucket=self._config.bucket_name,
            Key=hotkey,
            Body=body,
            ContentType="application/json",
        )
        url = self._public_url(hotkey)
        logger.info(
            "Uploaded payload to R2: %s (%d bytes)", url, len(body),
        )
        return url

    def _public_url(self, hotkey: str) -> str:
        if self._config.public_base_url:
            base = self._config.public_base_url.rstrip("/")
            return f"{base}/{hotkey}"
        logger.warning(
            "R2 public_base_url not set — validators may not be able to "
            "fetch payloads. Set it to your r2.dev subdomain URL."
        )
        return f"https://{self._config.bucket_name}.r2.dev/{hotkey}"
