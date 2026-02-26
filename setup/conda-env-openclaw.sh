#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-openclaw}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
NODEJS_VERSION="${NODEJS_VERSION:-20}"
PNPM_VERSION="${PNPM_VERSION:-9.15.4}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found on PATH" >&2
  exit 1
fi

has_env() {
  conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"
}

if has_env; then
  echo "Conda env '$ENV_NAME' exists; ensuring packages are present..."
  conda install -y -n "$ENV_NAME" -c conda-forge \
    "python=$PYTHON_VERSION" \
    "nodejs=$NODEJS_VERSION" \
    git
else
  echo "Creating conda env '$ENV_NAME'..."
  conda create -y -n "$ENV_NAME" -c conda-forge \
    "python=$PYTHON_VERSION" \
    "nodejs=$NODEJS_VERSION" \
    git
fi

echo "Configuring pnpm (via Corepack) inside '$ENV_NAME'..."
conda run -n "$ENV_NAME" corepack enable
conda run -n "$ENV_NAME" corepack prepare "pnpm@$PNPM_VERSION" --activate

echo "Done. Verify:" \
  && conda run -n "$ENV_NAME" node --version \
  && conda run -n "$ENV_NAME" pnpm --version
