"""MANTIS miner — live inference, encryption, R2 upload, and subnet registration.

Runs inside a Docker container or on Targon.  The GUI launches via
sandbox.launch_miner_container() and monitors through the miner directory.

SECURITY: Use a DEDICATED mining hotkey with MINIMAL funds (~0.3 TAO).
For registration, use a SEPARATE low-balance coldkey — never your main wallet.
Mnemonics live only in container memory and are never written to disk.

Lifecycle: register -> resolve wallet -> commit R2 URL -> inference loop.
Signals: MINER_STOP (graceful shutdown), MINER_FAILED (fatal error).
"""

from __future__ import annotations

import json
import logging
import os
import signal
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from mantis_model_iteration_tool.encryption import encrypt_v2, NAME_TO_TICKER
from mantis_model_iteration_tool.evaluator import CHALLENGES as EVAL_CHALLENGES
from mantis_model_iteration_tool.inferencer import (
    ModelSlot,
    fetch_live_data,
    load_model_slot,
    run_all_inference,
    _collect_required_assets,
)
from mantis_model_iteration_tool.r2_comms import R2Client, R2Config

logger = logging.getLogger(__name__)

PKG_DIR = Path(__file__).parent
NETUID = 123


def _uid_from_metagraph(metagraph, hotkey_ss58):
    """Look up UID for a hotkey in a metagraph, return int or None."""
    if hotkey_ss58 in metagraph.hotkeys:
        idx = metagraph.hotkeys.index(hotkey_ss58)
        return int(metagraph.uids[idx])
    return None


SECURITY_WARNING = """
╔══════════════════════════════════════════════════════════════════════╗
║  ⚠  MANTIS MINER — SECURITY WARNING                                ║
║                                                                      ║
║  • Use a DEDICATED hotkey with MINIMAL funds (~0.3 TAO to register) ║
║  • DO NOT reuse a hotkey that holds significant TAO                  ║
║  • DO NOT give this hotkey ownership, Senate, or delegation power    ║
║  • The mnemonic is held in container memory — treat as exposed      ║
║  • NEVER enter your coldkey mnemonic anywhere in this system        ║
║  • If this hotkey is compromised, only its small balance is at risk ║
╚══════════════════════════════════════════════════════════════════════╝
""".strip()

REGISTRATION_WARNING = """
╔══════════════════════════════════════════════════════════════════════╗
║  ⚠  SUBNET REGISTRATION — READ CAREFULLY                           ║
║                                                                      ║
║  This will register a hotkey on SN123 via burned_register().        ║
║  It REQUIRES a coldkey mnemonic to sign the on-chain transaction    ║
║  and pay the registration burn fee.                                  ║
║                                                                      ║
║  ┌────────────────────────────────────────────────────────────────┐  ║
║  │  USE A SEPARATE WALLET.  DO NOT use the mnemonic for your     │  ║
║  │  main TAO wallet or any wallet holding significant funds.     │  ║
║  │  Create a DEDICATED coldkey wallet, fund it with only the     │  ║
║  │  minimum TAO needed for registration (~0.3 TAO).              │  ║
║  │                                                                │  ║
║  │  The coldkey mnemonic is a SERIOUS SECURITY HAZARD if         │  ║
║  │  exposed. It grants FULL CONTROL over all TAO in that wallet. │  ║
║  │  Treat any mnemonic you enter here as potentially compromised.│  ║
║  │                                                                │  ║
║  │  THIS IS NOT SUITABLE FOR COLD STORAGE WALLETS.               │  ║
║  └────────────────────────────────────────────────────────────────┘  ║
║                                                                      ║
║  The coldkey mnemonic is used ONCE for registration, then            ║
║  discarded from memory.  It is never written to disk or logged.     ║
╚══════════════════════════════════════════════════════════════════════╝
""".strip()


def _resolve_hotkey(
    mnemonic: str = "",
    keyfile_path: str = "",
    wallet_name: str = "",
    hotkey_name: str = "",
) -> tuple[str, Any]:
    """Derive the SS58 hotkey address and a bittensor wallet object.

    Accepts one of:
      - mnemonic: 12/24-word BIP39 phrase
      - keyfile_path: path to a bittensor hotkey JSON keyfile
      - wallet_name + hotkey_name: standard ~/.bittensor/wallets/ layout

    Returns (ss58_address, bt.wallet).
    """
    import bittensor as bt

    if mnemonic:
        wallet = bt.Wallet(name="_mantis_miner", hotkey="_derived")
        wallet.regenerate_hotkey(mnemonic=mnemonic, overwrite=True, suppress=True)
        return wallet.hotkey.ss58_address, wallet

    if keyfile_path:
        p = Path(keyfile_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Hotkey keyfile not found: {p}")
        wallet = bt.Wallet(name="_mantis_miner", hotkey="_from_file")
        wallet.hotkey_file._path = str(p)
        if not wallet.hotkey_file.exists_on_device():
            raise FileNotFoundError(f"Keyfile not readable: {p}")
        return wallet.hotkey.ss58_address, wallet

    if wallet_name and hotkey_name:
        wallet = bt.Wallet(name=wallet_name, hotkey=hotkey_name)
        return wallet.hotkey.ss58_address, wallet

    raise ValueError(
        "Must provide one of: HOTKEY_MNEMONIC, HOTKEY_PATH, "
        "or (BT_WALLET_NAME + BT_HOTKEY_NAME)"
    )


# ── Registration ────────────────────────────────────────────────────────────

@dataclass
class RegistrationResult:
    success: bool = False
    hotkey_ss58: str = ""
    coldkey_ss58: str = ""
    uid: int = -1
    already_registered: bool = False
    error: str = ""
    cost: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def check_registration(
    hotkey_ss58: str,
    network: str = "finney",
) -> dict[str, Any]:
    """Check if a hotkey is already registered on SN123.

    Returns dict with 'registered', 'uid', 'stake', etc.
    Does NOT require any mnemonic — only the SS58 address.
    """
    import bittensor as bt

    result: dict[str, Any] = {
        "registered": False,
        "hotkey_ss58": hotkey_ss58,
        "uid": -1,
        "stake": 0.0,
        "network": network,
    }
    try:
        subtensor = bt.Subtensor(network=network)
        metagraph = subtensor.metagraph(netuid=NETUID)

        uid = _uid_from_metagraph(metagraph, hotkey_ss58)
        if uid is not None:
            idx = metagraph.hotkeys.index(hotkey_ss58)
            result["registered"] = True
            result["uid"] = uid
            result["stake"] = float(metagraph.S[idx])
            result["trust"] = float(metagraph.T[idx])
            result["incentive"] = float(metagraph.I[idx])
            logger.info(
                "Hotkey %s is registered on SN%d (uid=%d)",
                hotkey_ss58, NETUID, result["uid"],
            )
        else:
            logger.info(
                "Hotkey %s is NOT registered on SN%d", hotkey_ss58, NETUID,
            )
    except Exception as exc:
        result["error"] = str(exc)
        logger.error("Registration check failed: %s", exc)

    return result


def register_miner(
    coldkey_mnemonic: str,
    hotkey_mnemonic: str,
    network: str = "finney",
) -> RegistrationResult:
    """Register a hotkey on MANTIS subnet (SN123) via burned_register.

    SECURITY: This requires the coldkey mnemonic to sign the on-chain
    transaction and pay the registration burn fee.  The coldkey should be
    a DEDICATED low-balance wallet — NEVER your main TAO wallet.

    Both mnemonics are used only in memory and never written to disk.

    Args:
        coldkey_mnemonic: 12/24-word BIP39 phrase for the COLDKEY (pays fee)
        hotkey_mnemonic: 12/24-word BIP39 phrase for the HOTKEY (gets registered)
        network: Bittensor network ('finney' or 'test')

    Returns:
        RegistrationResult with success status and details.
    """
    import bittensor as bt

    print(REGISTRATION_WARNING, flush=True)
    result = RegistrationResult()

    try:
        wallet = bt.Wallet(name="_mantis_reg", hotkey="_reg_hk")
        wallet.regenerate_coldkey(
            mnemonic=coldkey_mnemonic, overwrite=True, suppress=True,
        )
        wallet.regenerate_hotkey(
            mnemonic=hotkey_mnemonic, overwrite=True, suppress=True,
        )

        result.hotkey_ss58 = wallet.hotkey.ss58_address
        result.coldkey_ss58 = wallet.coldkey.ss58_address
        logger.info(
            "Wallet loaded: coldkey=%s hotkey=%s",
            result.coldkey_ss58, result.hotkey_ss58,
        )

        subtensor = bt.Subtensor(network=network)

        metagraph = subtensor.metagraph(netuid=NETUID)
        uid = _uid_from_metagraph(metagraph, result.hotkey_ss58)
        if uid is not None:
            result.success = True
            result.already_registered = True
            result.uid = uid
            logger.info(
                "Hotkey %s already registered (uid=%d) — no action needed",
                result.hotkey_ss58, result.uid,
            )
            return result

        balance = subtensor.get_balance(result.coldkey_ss58)
        burn_cost = subtensor.recycle(netuid=NETUID)
        result.cost = float(burn_cost)

        logger.info(
            "Coldkey balance: %s TAO | Registration burn cost: %s TAO",
            balance, burn_cost,
        )

        if float(balance) < float(burn_cost) + 0.001:
            result.error = (
                f"Insufficient balance: have {balance} TAO, "
                f"need {burn_cost} TAO + fees. "
                f"Fund coldkey {result.coldkey_ss58} with at least "
                f"{float(burn_cost) + 0.01:.4f} TAO."
            )
            logger.error(result.error)
            return result

        logger.info(
            "Registering hotkey %s on SN%d (burn=%s TAO)...",
            result.hotkey_ss58, NETUID, burn_cost,
        )
        response = subtensor.burned_register(
            wallet=wallet,
            netuid=NETUID,
        )

        if response.success:
            metagraph = subtensor.metagraph(netuid=NETUID)
            uid = _uid_from_metagraph(metagraph, result.hotkey_ss58)
            if uid is not None:
                result.uid = uid
            result.success = True
            logger.info(
                "Registration successful! hotkey=%s uid=%d",
                result.hotkey_ss58, result.uid,
            )
        else:
            result.error = f"burned_register failed: {response.message}"
            logger.error(result.error)

    except Exception as exc:
        result.error = f"Registration failed: {exc}"
        logger.error(result.error)

    return result


# ── Wallet generation & balance ─────────────────────────────────────────────

@dataclass
class GeneratedWallet:
    coldkey_mnemonic: str = ""
    coldkey_ss58: str = ""
    hotkey_mnemonic: str = ""
    hotkey_ss58: str = ""
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def generate_wallet() -> GeneratedWallet:
    """Generate a fresh coldkey + hotkey pair for mining.

    Both are created in-memory only — mnemonics are returned to the caller
    and NEVER written to disk by this function.
    """
    result = GeneratedWallet()
    try:
        import bittensor as bt

        ck_mnemonic = bt.Keypair.generate_mnemonic(12)
        ck_pair = bt.Keypair.create_from_mnemonic(ck_mnemonic)

        hk_mnemonic = bt.Keypair.generate_mnemonic(12)
        hk_pair = bt.Keypair.create_from_mnemonic(hk_mnemonic)

        result.coldkey_mnemonic = ck_mnemonic
        result.coldkey_ss58 = ck_pair.ss58_address
        result.hotkey_mnemonic = hk_mnemonic
        result.hotkey_ss58 = hk_pair.ss58_address

        logger.info(
            "Generated wallet: coldkey=%s hotkey=%s",
            result.coldkey_ss58, result.hotkey_ss58,
        )
    except Exception as exc:
        result.error = f"Wallet generation failed: {exc}"
        logger.error(result.error)

    return result


def check_balance(
    ss58_address: str,
    network: str = "finney",
) -> dict[str, Any]:
    """Check the balance of an SS58 address on the Bittensor network.

    Returns dict with 'balance' (float, in TAO), 'ss58', 'network', and
    optionally 'burn_cost' (the current registration cost on SN123).
    """
    import bittensor as bt

    result: dict[str, Any] = {
        "ss58": ss58_address,
        "network": network,
        "balance": 0.0,
        "burn_cost": 0.0,
        "sufficient": False,
    }
    try:
        subtensor = bt.Subtensor(network=network)
        balance = subtensor.get_balance(ss58_address)
        result["balance"] = float(balance)

        try:
            burn_cost = subtensor.recycle(netuid=NETUID)
            result["burn_cost"] = float(burn_cost)
            result["sufficient"] = float(balance) >= float(burn_cost) + 0.001
        except Exception:
            pass

        logger.info(
            "Balance for %s: %s TAO (burn cost: %s TAO, sufficient: %s)",
            ss58_address, balance,
            result["burn_cost"], result["sufficient"],
        )
    except Exception as exc:
        result["error"] = str(exc)
        logger.error("Balance check failed: %s", exc)

    return result


# ── Config ──────────────────────────────────────────────────────────────────

@dataclass
class MinerConfig:
    miner_dir: str = ""
    hotkey_mnemonic: str = ""
    hotkey_path: str = ""
    wallet_name: str = ""
    hotkey_name: str = ""
    hotkey: str = ""
    interval_seconds: int = 60
    lock_seconds: int = 30
    coinglass_key: str = ""
    lookback_minutes: int = 5000
    network: str = "finney"
    r2: R2Config = field(default_factory=R2Config)
    models: list[ModelSlot] = field(default_factory=list)
    commit_url: str = ""

    def add_model(
        self,
        agent_id: str,
        iteration: int,
        challenge_name: str,
        strategy_path: str,
    ) -> None:
        self.models.append(ModelSlot(
            agent_id=agent_id,
            iteration=iteration,
            challenge_name=challenge_name,
            strategy_path=strategy_path,
        ))

    def validate(self) -> list[str]:
        errors = []
        has_key_source = bool(
            self.hotkey_mnemonic or self.hotkey_path
            or (self.wallet_name and self.hotkey_name)
        )
        if not has_key_source and not self.hotkey:
            errors.append(
                "hotkey credentials required: provide HOTKEY_MNEMONIC, "
                "HOTKEY_PATH, or (BT_WALLET_NAME + BT_HOTKEY_NAME)"
            )
        if self.interval_seconds < 30:
            errors.append("interval_seconds must be >= 30")
        if not self.models:
            errors.append("at least one model is required")
        r2_missing = self.r2.validate()
        if r2_missing:
            errors.append(f"R2 config missing: {r2_missing}")
        for slot in self.models:
            if not Path(slot.strategy_path).exists():
                errors.append(
                    f"Strategy file not found: {slot.strategy_path} "
                    f"(agent={slot.agent_id} iter={slot.iteration})"
                )
        return errors

    def to_dict(self) -> dict:
        return {
            "hotkey": self.hotkey,
            "interval_seconds": self.interval_seconds,
            "lock_seconds": self.lock_seconds,
            "lookback_minutes": self.lookback_minutes,
            "network": self.network,
            "models": [
                {
                    "agent_id": m.agent_id,
                    "iteration": m.iteration,
                    "challenge_name": m.challenge_name,
                    "strategy_path": m.strategy_path,
                }
                for m in self.models
            ],
            "r2_bucket": self.r2.bucket_name,
        }


# ── Status ──────────────────────────────────────────────────────────────────

@dataclass
class MinerStatus:
    running: bool = False
    last_submission: float = 0.0
    last_round: int = 0
    submissions_count: int = 0
    errors_count: int = 0
    consecutive_errors: int = 0
    last_error: str = ""
    last_embeddings: dict[str, Any] = field(default_factory=dict)
    last_predictions: dict[str, Any] = field(default_factory=dict)
    loaded_models: list[str] = field(default_factory=list)
    commit_done: bool = False
    fatal: bool = False

    def to_dict(self) -> dict:
        return {
            "running": self.running,
            "last_submission": self.last_submission,
            "last_submission_ago": (
                f"{time.time() - self.last_submission:.0f}s"
                if self.last_submission else "never"
            ),
            "last_round": self.last_round,
            "submissions_count": self.submissions_count,
            "errors_count": self.errors_count,
            "consecutive_errors": self.consecutive_errors,
            "last_error": self.last_error,
            "loaded_models": self.loaded_models,
            "commit_done": self.commit_done,
            "fatal": self.fatal,
            "predictions": self.last_predictions,
        }


# ── Process ─────────────────────────────────────────────────────────────────

MAX_CONSECUTIVE_ERRORS = 10


class MinerProcess:
    """Miner loop with process management signals.

    Communicates with the GUI via files in miner_dir:
      miner_status.json  — written every cycle (GUI polls this)
      miner_pid          — PID for process tracking
      MINER_STOP         — GUI writes to request graceful shutdown
      MINER_FAILED       — miner writes on fatal error
    """

    def __init__(
        self,
        config: MinerConfig,
        on_status: Optional[Callable[[MinerStatus], None]] = None,
    ):
        errors = config.validate()
        if errors:
            raise ValueError(f"Invalid miner config: {'; '.join(errors)}")

        self._config = config
        self._on_status = on_status
        self._status = MinerStatus()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._r2: Optional[R2Client] = None
        self._wallet = None
        self._lock = threading.Lock()

        self._miner_dir = Path(config.miner_dir) if config.miner_dir else None
        if self._miner_dir:
            self._miner_dir.mkdir(parents=True, exist_ok=True)

    @property
    def status(self) -> MinerStatus:
        return self._status

    def start(self, background: bool = False) -> None:
        if self._status.running:
            logger.warning("Miner already running")
            return

        self._stop_event.clear()
        if background:
            self._thread = threading.Thread(
                target=self._run_loop, daemon=True, name="mantis-miner",
            )
            self._thread.start()
        else:
            self._run_loop()

    def stop(self) -> None:
        logger.info("Miner stop requested")
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=30)

    # ── File-based process management ───────────────────────────────────

    def _write_pid(self) -> None:
        if self._miner_dir:
            (self._miner_dir / "miner_pid").write_text(str(os.getpid()))

    def _clear_pid(self) -> None:
        if self._miner_dir:
            (self._miner_dir / "miner_pid").unlink(missing_ok=True)

    def _check_stop_signal(self) -> bool:
        if self._stop_event.is_set():
            return True
        if self._miner_dir and (self._miner_dir / "MINER_STOP").exists():
            logger.info("MINER_STOP file detected — shutting down")
            self._stop_event.set()
            return True
        return False

    def _write_failed(self, error: str) -> None:
        self._status.fatal = True
        self._status.last_error = error
        if self._miner_dir:
            (self._miner_dir / "MINER_FAILED").write_text(
                json.dumps({
                    "error": error[:2000],
                    "timestamp": time.time(),
                    "submissions_count": self._status.submissions_count,
                    "errors_count": self._status.errors_count,
                }, indent=2)
            )
        logger.error("MINER_FAILED: %s", error[:500])

    def _write_status(self) -> None:
        if self._miner_dir:
            try:
                status_path = self._miner_dir / "miner_status.json"
                tmp = status_path.with_suffix(".tmp")
                tmp.write_text(json.dumps(self._status.to_dict(), indent=2))
                tmp.replace(status_path)
            except OSError:
                pass

    # ── Main loop ───────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        self._status.running = True
        self._write_pid()
        self._emit_status()

        logger.info(
            "Miner starting: interval=%ds models=%d",
            self._config.interval_seconds, len(self._config.models),
        )

        try:
            self._resolve_wallet()
            logger.info("Hotkey: %s", self._config.hotkey)
            self._init_r2()
            self._load_models()
            self._do_commit()

            if self._miner_dir:
                (self._miner_dir / "MINER_FAILED").unlink(missing_ok=True)

            while not self._check_stop_signal():
                cycle_start = time.monotonic()
                try:
                    self._run_one_cycle()
                    self._status.consecutive_errors = 0
                except Exception:
                    self._status.errors_count += 1
                    self._status.consecutive_errors += 1
                    self._status.last_error = traceback.format_exc()[-500:]
                    logger.error("Miner cycle error:\n%s", self._status.last_error)

                    if self._status.consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        self._write_failed(
                            f"Too many consecutive errors "
                            f"({self._status.consecutive_errors}): "
                            f"{self._status.last_error}"
                        )
                        break

                self._write_status()
                self._emit_status()

                elapsed = time.monotonic() - cycle_start
                sleep_time = max(1, self._config.interval_seconds - elapsed)
                if self._stop_event.wait(timeout=sleep_time):
                    break

        except Exception:
            tb = traceback.format_exc()[-1000:]
            self._write_failed(tb)
            logger.error("Miner fatal error:\n%s", tb)
        finally:
            self._status.running = False
            self._write_status()
            self._clear_pid()
            self._emit_status()
            logger.info("Miner stopped (submissions=%d errors=%d)",
                        self._status.submissions_count, self._status.errors_count)

    # ── Initialization ──────────────────────────────────────────────────

    def _resolve_wallet(self) -> None:
        has_key_source = bool(
            self._config.hotkey_mnemonic or self._config.hotkey_path
            or (self._config.wallet_name and self._config.hotkey_name)
        )
        if has_key_source:
            ss58, wallet = _resolve_hotkey(
                mnemonic=self._config.hotkey_mnemonic,
                keyfile_path=self._config.hotkey_path,
                wallet_name=self._config.wallet_name,
                hotkey_name=self._config.hotkey_name,
            )
            self._wallet = wallet
            if self._config.hotkey and self._config.hotkey != ss58:
                logger.warning(
                    "Provided hotkey %s does not match derived %s — using derived",
                    self._config.hotkey, ss58,
                )
            self._config.hotkey = ss58
            logger.info("Wallet loaded: hotkey=%s", ss58)
        elif self._config.hotkey:
            logger.warning(
                "Only SS58 address provided (no mnemonic/keyfile) — "
                "chain commit will not be possible"
            )
        else:
            raise RuntimeError("No hotkey credentials provided")

    def _init_r2(self) -> None:
        self._r2 = R2Client(self._config.r2)
        logger.info("R2 client initialized: bucket=%s", self._config.r2.bucket_name)

    def _load_models(self) -> None:
        loaded = []
        for slot in self._config.models:
            load_model_slot(slot)
            if slot.loaded:
                loaded.append(
                    f"{slot.challenge_name} "
                    f"(agent={slot.agent_id} iter={slot.iteration})"
                )
            else:
                logger.error(
                    "Model load failed: %s — %s", slot.strategy_path, slot.error,
                )
        self._status.loaded_models = loaded
        if not loaded:
            raise RuntimeError("No models loaded successfully — cannot mine")
        logger.info("Loaded %d/%d models", len(loaded), len(self._config.models))

    def _do_commit(self) -> None:
        """Commit the R2 payload URL to the Bittensor chain."""
        url = self._r2._public_url(self._config.hotkey)
        self._config.commit_url = url

        if self._wallet is None:
            logger.warning(
                "No wallet loaded — skipping chain commit. URL: %s", url,
            )
            return

        import bittensor as bt

        logger.info(
            "Committing URL to chain: %s (network=%s, netuid=%d)",
            url, self._config.network, NETUID,
        )
        try:
            subtensor = bt.Subtensor(network=self._config.network)
            response = subtensor.set_commitment(
                wallet=self._wallet,
                netuid=NETUID,
                data=url,
            )
            if hasattr(response, "success") and not response.success:
                raise RuntimeError(
                    getattr(response, "message", "set_commitment returned failure")
                )
            self._status.commit_done = True
            logger.info("Chain commit successful")
        except Exception as exc:
            error_msg = f"Chain commit failed: {exc}"
            logger.error(error_msg)

    # ── Cycle ───────────────────────────────────────────────────────────

    def _run_one_cycle(self) -> None:
        t0 = time.monotonic()

        assets = _collect_required_assets(self._config.models)
        logger.info("Fetching live data for %d assets...", len(assets))
        provider = fetch_live_data(
            assets=assets,
            lookback_minutes=self._config.lookback_minutes,
            coinglass_key=self._config.coinglass_key,
        )

        logger.info(
            "Running inference across %d model(s)...",
            sum(1 for s in self._config.models if s.loaded),
        )
        embeddings = run_all_inference(self._config.models, provider)
        self._status.last_embeddings = embeddings

        pred_info: dict[str, dict] = {}
        for slot in self._config.models:
            if not slot.loaded:
                continue
            ticker = NAME_TO_TICKER.get(slot.challenge_name)
            if ticker and ticker in embeddings:
                cfg = EVAL_CHALLENGES.get(slot.challenge_name)
                ctype = cfg.challenge_type if cfg else "unknown"
                pred_info[ticker] = {
                    "challenge": slot.challenge_name,
                    "type": ctype,
                    "agent_id": slot.agent_id,
                    "iteration": slot.iteration,
                    "value": _serialize_prediction(embeddings[ticker]),
                }
        self._status.last_predictions = pred_info

        if not embeddings:
            raise RuntimeError("All models failed inference — nothing to submit")

        logger.info("Encrypting V2 payload (lock=%ds)...", self._config.lock_seconds)
        payload = encrypt_v2(
            hotkey=self._config.hotkey,
            embeddings=embeddings,
            lock_seconds=self._config.lock_seconds,
        )
        self._status.last_round = payload.get("round", 0)

        logger.info("Uploading to R2...")
        self._r2.upload_payload(self._config.hotkey, payload)

        elapsed = time.monotonic() - t0
        self._status.submissions_count += 1
        self._status.last_submission = time.time()
        logger.info(
            "Submission #%d OK (%.1fs, round=%d, %d embeddings)",
            self._status.submissions_count, elapsed,
            self._status.last_round, len(embeddings),
        )

    def _emit_status(self) -> None:
        if self._on_status:
            try:
                self._on_status(self._status)
            except Exception:
                pass


def _serialize_prediction(val: Any) -> Any:
    """Make prediction values JSON-safe with limited precision."""
    if isinstance(val, dict):
        return {k: _serialize_prediction(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [round(float(x), 6) if isinstance(x, (int, float)) else x for x in val]
    if isinstance(val, (int, float)):
        return round(float(val), 6)
    return val


# ── CLI entry point (runs INSIDE the container) ────────────────────────────

def _parse_model_arg(arg: str) -> tuple[str, int, str, str]:
    """Parse 'agent_id:iteration:challenge:path' model specification."""
    parts = arg.split(":", 3)
    if len(parts) != 4:
        raise ValueError(
            f"Model spec must be 'agent_id:iteration:challenge:path', got: {arg}"
        )
    return parts[0], int(parts[1]), parts[2], parts[3]


def _cmd_check_registration(args) -> int:
    """CLI handler for --check-registration."""
    hotkey_ss58 = args.hotkey_ss58 or os.environ.get("HOTKEY_SS58", "")
    if not hotkey_ss58:
        hotkey_mnemonic = os.environ.get("HOTKEY_MNEMONIC", "")
        if hotkey_mnemonic:
            ss58, _ = _resolve_hotkey(mnemonic=hotkey_mnemonic)
            hotkey_ss58 = ss58
    if not hotkey_ss58:
        logger.error(
            "Provide --hotkey-ss58 or set HOTKEY_SS58 / HOTKEY_MNEMONIC env var"
        )
        return 1

    result = check_registration(hotkey_ss58, network=args.network)
    print(json.dumps(result, indent=2), flush=True)

    if args.miner_dir:
        miner_dir = Path(args.miner_dir)
        miner_dir.mkdir(parents=True, exist_ok=True)
        (miner_dir / "registration_check.json").write_text(
            json.dumps(result, indent=2)
        )

    return 0 if result.get("registered") or not result.get("error") else 1


def _cmd_register(args) -> int:
    """CLI handler for --register."""
    coldkey_mnemonic = args.coldkey_mnemonic or os.environ.get("COLDKEY_MNEMONIC", "")
    hotkey_mnemonic = args.hotkey_mnemonic or os.environ.get("HOTKEY_MNEMONIC", "")

    if not coldkey_mnemonic:
        logger.error(
            "Coldkey mnemonic required for registration. "
            "Pass --coldkey-mnemonic or set COLDKEY_MNEMONIC env var."
        )
        return 1
    if not hotkey_mnemonic:
        logger.error(
            "Hotkey mnemonic required for registration. "
            "Pass --hotkey-mnemonic or set HOTKEY_MNEMONIC env var."
        )
        return 1

    result = register_miner(
        coldkey_mnemonic=coldkey_mnemonic,
        hotkey_mnemonic=hotkey_mnemonic,
        network=args.network,
    )
    print(json.dumps(result.to_dict(), indent=2), flush=True)

    if args.miner_dir:
        miner_dir = Path(args.miner_dir)
        miner_dir.mkdir(parents=True, exist_ok=True)
        safe_result = result.to_dict()
        (miner_dir / "registration_result.json").write_text(
            json.dumps(safe_result, indent=2)
        )

    return 0 if result.success else 1


def _cmd_generate(_args) -> int:
    """CLI handler for wallet generation."""
    result = generate_wallet()
    print(json.dumps(result.to_dict(), indent=2), flush=True)
    return 0 if not result.error else 1


def _cmd_balance(args) -> int:
    """CLI handler for balance check."""
    ss58 = args.ss58 or os.environ.get("SS58_ADDRESS", "")
    if not ss58:
        logger.error("Provide --ss58 or set SS58_ADDRESS env var")
        return 1
    network = getattr(args, "network", "finney")
    result = check_balance(ss58, network=network)
    print(json.dumps(result, indent=2), flush=True)
    return 0 if "error" not in result else 1


def _cmd_mine(args) -> int:
    """CLI handler for the main mining loop."""
    cfg = MinerConfig(
        miner_dir=args.miner_dir,
        hotkey_mnemonic=os.environ.get("HOTKEY_MNEMONIC", ""),
        hotkey_path=os.environ.get("HOTKEY_PATH", ""),
        wallet_name=os.environ.get("BT_WALLET_NAME", ""),
        hotkey_name=os.environ.get("BT_HOTKEY_NAME", ""),
        hotkey=os.environ.get("HOTKEY_SS58", ""),
        interval_seconds=args.interval,
        lock_seconds=args.lock_seconds,
        lookback_minutes=args.lookback,
        network=args.network,
        coinglass_key=os.environ.get("COINGLASS_API_KEY", ""),
        r2=R2Config(
            account_id=os.environ.get("R2_ACCOUNT_ID", ""),
            access_key_id=os.environ.get("R2_ACCESS_KEY_ID", ""),
            secret_access_key=os.environ.get("R2_SECRET_ACCESS_KEY", ""),
            bucket_name=os.environ.get("R2_BUCKET_NAME", ""),
            public_base_url=os.environ.get("R2_PUBLIC_BASE_URL", ""),
        ),
    )

    for model_spec in args.model:
        agent_id, iteration, challenge, path = _parse_model_arg(model_spec)
        cfg.add_model(agent_id, iteration, challenge, path)

    errors = cfg.validate()
    if errors:
        for e in errors:
            logger.error("Config error: %s", e)
        miner_dir = Path(args.miner_dir)
        miner_dir.mkdir(parents=True, exist_ok=True)
        (miner_dir / "MINER_FAILED").write_text(json.dumps({
            "error": "; ".join(errors),
            "timestamp": time.time(),
        }, indent=2))
        return 1

    mp = MinerProcess(cfg)

    def _handle_signal(signum, frame):
        logger.info("Signal %d received, stopping miner...", signum)
        mp.stop()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    print(SECURITY_WARNING, flush=True)
    logger.info("Starting MANTIS miner (SIGTERM or MINER_STOP to stop)")
    mp.start(background=False)
    return 0


def main() -> int:
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="MANTIS miner — runs inside Docker/Targon container",
    )
    sub = parser.add_subparsers(dest="command")

    # ── mine (default) ──────────────────────────────────────────────────
    mine_p = sub.add_parser("mine", help="Run the miner loop (default)")
    mine_p.add_argument(
        "--model", action="append", required=True,
        help="Model spec: agent_id:iteration:challenge_name:strategy_path "
             "(can be repeated for multiple models)",
    )
    mine_p.add_argument("--miner-dir", default="/miner")
    mine_p.add_argument("--interval", type=int, default=60)
    mine_p.add_argument("--lock-seconds", type=int, default=30)
    mine_p.add_argument("--lookback", type=int, default=5000)
    mine_p.add_argument("--network", default="finney")

    # ── check ───────────────────────────────────────────────────────────
    check_p = sub.add_parser(
        "check", help="Check if a hotkey is registered on SN123",
    )
    check_p.add_argument("--hotkey-ss58", default="")
    check_p.add_argument("--miner-dir", default="/miner")
    check_p.add_argument("--network", default="finney")

    # ── register ────────────────────────────────────────────────────────
    reg_p = sub.add_parser(
        "register",
        help="Register a hotkey on SN123 via burned_register",
    )
    reg_p.add_argument(
        "--coldkey-mnemonic", default="",
        help="Coldkey 12/24 word BIP39 mnemonic (or COLDKEY_MNEMONIC env var). "
             "USE A DEDICATED LOW-BALANCE WALLET — NOT your main TAO wallet!",
    )
    reg_p.add_argument(
        "--hotkey-mnemonic", default="",
        help="Hotkey 12/24 word BIP39 mnemonic (or HOTKEY_MNEMONIC env var)",
    )
    reg_p.add_argument("--miner-dir", default="/miner")
    reg_p.add_argument("--network", default="finney")

    # ── generate ──────────────────────────────────────────────────────
    sub.add_parser(
        "generate",
        help="Generate a fresh coldkey + hotkey pair and output JSON",
    )

    # ── balance ───────────────────────────────────────────────────────
    bal_p = sub.add_parser(
        "balance",
        help="Check TAO balance of an SS58 address",
    )
    bal_p.add_argument("--ss58", default="",
                       help="SS58 address to check (or SS58_ADDRESS env var)")
    bal_p.add_argument("--network", default="finney")

    # ── backwards compat: if called with --model directly (no subcommand)
    parser.add_argument(
        "--model", action="append", default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--miner-dir", default="/miner", help=argparse.SUPPRESS)
    parser.add_argument("--interval", type=int, default=60, help=argparse.SUPPRESS)
    parser.add_argument("--lock-seconds", type=int, default=30, help=argparse.SUPPRESS)
    parser.add_argument("--lookback", type=int, default=5000, help=argparse.SUPPRESS)
    parser.add_argument("--network", default="finney", help=argparse.SUPPRESS)

    args = parser.parse_args()

    if args.command == "check":
        return _cmd_check_registration(args)
    elif args.command == "register":
        return _cmd_register(args)
    elif args.command == "generate":
        return _cmd_generate(args)
    elif args.command == "balance":
        return _cmd_balance(args)
    elif args.command == "mine":
        return _cmd_mine(args)
    elif args.model:
        return _cmd_mine(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
