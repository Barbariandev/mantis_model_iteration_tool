"""Block <-> UTC <-> sidx synchronization for Bittensor SN123.

Calibration: block 7,898,337 = 2026-04-05 01:10:00 UTC
             (Sat Apr 4 7:10 PM Costa Rica / UTC-6)
Each finney block is 12 seconds.  Validators sample every SAMPLE_EVERY=5
blocks, so one sample index (sidx) = 60 seconds = 1 minute.
"""

import datetime

REFERENCE_BLOCK = 7_898_337
REFERENCE_UTC = datetime.datetime(2026, 4, 5, 1, 10, 0, tzinfo=datetime.timezone.utc)
REFERENCE_EPOCH = REFERENCE_UTC.timestamp()
BLOCK_TIME = 12  # seconds
SAMPLE_EVERY = 5  # blocks per sample


def block_to_epoch(block: int) -> float:
    return REFERENCE_EPOCH + (block - REFERENCE_BLOCK) * BLOCK_TIME


def epoch_to_block(epoch: float) -> int:
    return REFERENCE_BLOCK + int((epoch - REFERENCE_EPOCH) / BLOCK_TIME)


def sidx_to_block(sidx: int) -> int:
    return sidx * SAMPLE_EVERY


def block_to_sidx(block: int) -> int:
    return block // SAMPLE_EVERY


def sidx_to_epoch(sidx: int) -> float:
    return block_to_epoch(sidx_to_block(sidx))


def epoch_to_sidx(epoch: float) -> int:
    return block_to_sidx(epoch_to_block(epoch))


def epoch_to_utc(epoch: float) -> datetime.datetime:
    return datetime.datetime.fromtimestamp(epoch, tz=datetime.timezone.utc)


def ohlcv_minute_to_sidx(minute_epoch_ms: int) -> int:
    """Convert an OHLCV minute-open timestamp (ms) to a datalog sidx."""
    return epoch_to_sidx(minute_epoch_ms / 1000.0)
