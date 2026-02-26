# Setup: install OpenClaw into a conda env

This folder contains helper scripts to install OpenClaw into a conda environment named `openclaw`.

## 1) Create / update the conda env

From the playground repo root:

- `bash setup/conda-env-openclaw.sh`

This creates (or updates) an env named `openclaw` with:

- Python (for any Python-side tooling you might add later)
- Node.js (for the OpenClaw monorepo)
- Git

It also enables Corepack and activates a pinned pnpm version inside the env.

## 2) Clone + install OpenClaw

By default the installer clones the repo to a sibling directory `../openclaw`.

- `bash setup/install-openclaw.sh`

You can override the target directory:

- `OPENCLAW_DIR=/path/to/openclaw bash setup/install-openclaw.sh`

## Notes

- These scripts assume `conda` is available on PATH.
- If you already have an `openclaw` env, the scripts are designed to be re-runnable.
