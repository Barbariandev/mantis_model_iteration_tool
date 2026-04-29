#!/usr/bin/env bash
# install_reqs.sh — adapted from github.com/Barbariandev/MANTIS
# Sets up a Python venv at /data/.venv with timelock + miner deps.

set -Eeuo pipefail
IFS=$'\n\t'
export DEBIAN_FRONTEND=noninteractive

step() { echo -e "\n\033[1;36m▶ $*\033[0m"; }
ok()   { echo -e "\033[1;32m✔ $*\033[0m"; }
warn() { echo -e "\033[1;33m⚠ $*\033[0m"; }
die()  { echo -e "\033[1;31m✖ $*\033[0m" >&2; exit 1; }
trap 'die "Error on or near line $LINENO. Aborting."' ERR

VENV="/data/.venv"
SRC="${SRC:-/data/timelock-src}"
MARKER="$VENV/.installed"
SCRIPT_HASH=$(md5sum "$0" | cut -d' ' -f1)

if [ -f "$MARKER" ] && [ "$(cat "$MARKER")" = "$SCRIPT_HASH" ]; then
    ok "Dependencies already installed, skipping."
    exit 0
fi

# ── Choose interpreter ──────────────────────────────────────────────
if [[ -n "${PY_BIN:-}" ]]; then
  command -v "$PY_BIN" >/dev/null 2>&1 || die "PY_BIN=$PY_BIN not found"
elif command -v python3.10 >/dev/null 2>&1; then
  PY_BIN="python3.10"
elif command -v python3.11 >/dev/null 2>&1; then
  PY_BIN="python3.11"
elif command -v python3.12 >/dev/null 2>&1; then
  PY_BIN="python3.12"
elif command -v python3 >/dev/null 2>&1; then
  PY_BIN="python3"
else
  die "No Python 3 interpreter found."
fi

"$PY_BIN" -c "import sys; assert sys.version_info >= (3, 10)" \
  || die "$PY_BIN is older than 3.10"

PY_VERSION="$("$PY_BIN" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}")')"
step "Using interpreter: $PY_BIN (Python $PY_VERSION)"

# ── Create (or reuse) venv ──────────────────────────────────────────
if [[ ! -d "$VENV" ]]; then
  step "Creating virtualenv at $VENV"
  "$PY_BIN" -m venv --system-site-packages "$VENV"
  ok "Created $VENV"
else
  ok "Reusing existing $VENV"
fi

source "$VENV/bin/activate"
PY="$VENV/bin/python"

step "Upgrading pip tooling"
$PY -m pip install -qU pip "setuptools~=70.0" wheel
ok "pip ready"

# ── Install miner deps ─────────────────────────────────────────────
step "Installing miner dependencies"
$PY -m pip install -q --no-input \
    "boto3>=1.35" \
    "bittensor>=7.0" \
    "cryptography>=42"
ok "Miner deps installed"

# ── Install timelock ────────────────────────────────────────────────
step "Attempting timelock install from PyPI (prebuilt wheels)"
PREBUILT=0
if "$PY" -c 'import timelock' >/dev/null 2>&1; then
  ok "timelock already installed in venv"
  PREBUILT=1
else
  if $PY -m pip install -q --no-input timelock; then
    if "$PY" -c 'import timelock' >/dev/null 2>&1; then
      ok "Installed timelock from PyPI"
      PREBUILT=1
    else
      warn "timelock import failed after PyPI install; will build from source"
      PREBUILT=0
    fi
  else
    warn "PyPI install did not succeed; will build from source"
    PREBUILT=0
  fi
fi

if [[ "$PREBUILT" -eq 0 ]]; then
  step "Installing system build dependencies"
  if command -v apt-get >/dev/null 2>&1; then
    apt-get update -qq || true
    apt-get install -y --no-install-recommends \
      build-essential pkg-config libssl-dev ca-certificates git curl
    pyver="$($PY -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
    apt-get install -y "python${pyver}-dev" || apt-get install -y python3-dev || true
  fi

  step "Ensuring Rust toolchain"
  if ! command -v rustup >/dev/null 2>&1; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  fi
  source "$HOME/.cargo/env" 2>/dev/null || true
  rustup toolchain install stable 2>/dev/null
  rustup default stable 2>/dev/null

  step "Cloning/updating timelock sources"
  if [[ ! -d "$SRC/.git" ]]; then
    git clone --depth 1 https://github.com/ideal-lab5/timelock.git "$SRC"
  else
    git -C "$SRC" pull --ff-only || true
  fi

  if [[ -d "$SRC/wasm/src" ]]; then
    sed -i.bak 's|ark_std::rand::rng::OsRng|ark_std::rand::rngs::OsRng|g' "$SRC/wasm/src/"{py,js}.rs || true
    sed -i 's|Identity::new(b"", id)|Identity::new(b"", \&id)|g' "$SRC/wasm/src/"{py,js}.rs || true
  fi

  step "Building timelock_wasm_wrapper"
  $PY -m pip install -qU maturin
  pushd "$SRC/wasm" >/dev/null
  $PY -m maturin build --release --features "python" --interpreter "$PY"
  popd >/dev/null

  WHEEL_PATH="$(ls -1 "$SRC"/target/wheels/timelock_wasm_wrapper-*.whl 2>/dev/null | head -n1 || true)"
  if [[ -z "${WHEEL_PATH:-}" ]]; then
    WHEEL_PATH="$(ls -1 "$SRC"/wasm/target/wheels/timelock_wasm_wrapper-*.whl 2>/dev/null | head -n1 || true)"
  fi
  [[ -n "${WHEEL_PATH:-}" ]] || die "Failed to find built wheel"

  $PY -m pip install -U "$WHEEL_PATH"

  step "Installing Python bindings"
  $PY -m pip install -U "$SRC/py"

  ok "Built and installed timelock from source"
fi

# ── Validate ────────────────────────────────────────────────────────
step "Verifying critical imports"
$PY -c "import boto3; import bittensor; import timelock; print('All imports OK')" \
  || die "Critical import check failed"

echo "$SCRIPT_HASH" > "$MARKER"
ok "Installation complete in $VENV"
