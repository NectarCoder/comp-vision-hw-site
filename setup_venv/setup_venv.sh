#!/usr/bin/env bash
set -euo pipefail

# Creates and activates a Python virtual environment and installs dependencies.
# Usage: ./scripts/setup_venv.sh [VENV_DIR] [--no-activate]
# Default VENV_DIR is ./.venv

VENV_DIR=${1:-.venv}
NO_ACTIVATE=false

if [[ "${2-}" == "--no-activate" || "${3-}" == "--no-activate" ]]; then
  NO_ACTIVATE=true
fi

echo "Using Python interpreter: ${PYTHON:-python3}"

# prefer python3 from PATH
PYTHON=${PYTHON:-python3}

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Error: $PYTHON not found in PATH. Install Python 3 or set PYTHON env var to a valid interpreter." >&2
  exit 2
fi

if [[ -d "$VENV_DIR" ]]; then
  echo "Virtualenv already exists at $VENV_DIR — reusing it." 
else
  echo "Creating virtual environment in $VENV_DIR..."
  # check that venv and ensurepip are available in the selected interpreter
  if ! "$PYTHON" -c 'import venv, ensurepip' >/dev/null 2>&1; then
    echo "Error: your Python interpreter does not have the venv/ensurepip modules available." >&2
    echo "On Debian/Ubuntu install the venv support (example): sudo apt install python3-venv" >&2
    echo "On other distros, install the package that provides ensurepip/venv or use a Python build with ensurepip enabled." >&2
    exit 2
  fi

  "$PYTHON" -m venv "$VENV_DIR"
fi

ACTIVATE="$VENV_DIR/bin/activate"
if [[ ! -f "$ACTIVATE" ]]; then
  echo "Error: activate script missing at $ACTIVATE" >&2
  exit 3
fi

if [[ "$NO_ACTIVATE" == "false" ]]; then
  # shellcheck disable=SC1090
  source "$ACTIVATE"
  echo "Activated virtualenv at $VENV_DIR"
fi

echo "Upgrading pip and setuptools..."
"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel

# Prefer a requirements.txt located next to this script, fall back to the
# current working directory. This avoids failures when the script is executed
# from the repo root but the requirements live in the script folder.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ_IN_SCRIPT="$SCRIPT_DIR/requirements.txt"
REQ_IN_CWD="$PWD/requirements.txt"
REQ_FILE=""

if [[ -f "$REQ_IN_SCRIPT" ]]; then
  REQ_FILE="$REQ_IN_SCRIPT"
elif [[ -f "$REQ_IN_CWD" ]]; then
  REQ_FILE="$REQ_IN_CWD"
fi

if [[ -n "$REQ_FILE" ]]; then
  echo "Installing packages from $REQ_FILE"
  "$VENV_DIR/bin/pip" install -r "$REQ_FILE"
else
  echo "No $REQ_FILE found — installing Flask and flask-cors directly"
  "$VENV_DIR/bin/pip" install Flask flask-cors
fi

echo "Done. To activate the venv run: source $VENV_DIR/bin/activate"
