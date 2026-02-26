#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-openclaw}"
OPENCLAW_DIR="${OPENCLAW_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/../openclaw}"
OPENCLAW_REPO_URL="${OPENCLAW_REPO_URL:-https://github.com/openclaw/openclaw.git}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found on PATH" >&2
  exit 1
fi

if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "Conda env '$ENV_NAME' not found. Run: bash setup/conda-env-openclaw.sh" >&2
  exit 1
fi

mkdir -p "$(dirname "$OPENCLAW_DIR")"

if [ -d "$OPENCLAW_DIR/.git" ]; then
  echo "OpenClaw repo already exists at $OPENCLAW_DIR; updating..."
  git -C "$OPENCLAW_DIR" fetch --all --prune
  git -C "$OPENCLAW_DIR" checkout main
  git -C "$OPENCLAW_DIR" pull --ff-only
else
  echo "Cloning OpenClaw into $OPENCLAW_DIR"
  git clone "$OPENCLAW_REPO_URL" "$OPENCLAW_DIR"
fi

echo "Installing dependencies with pnpm in conda env '$ENV_NAME'..."
conda run -n "$ENV_NAME" pnpm -C "$OPENCLAW_DIR" install

echo "Done. Next steps (example):"
echo "  conda activate $ENV_NAME"
echo "  cd $OPENCLAW_DIR"
echo "  pnpm dev   # or see repo README"
