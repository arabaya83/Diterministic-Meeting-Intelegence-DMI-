#!/usr/bin/env bash
# Offline/cache defaults for the AMI NeMo pipeline (repo-local).
# Usage:
#   source scripts/env_offline.sh

_ami_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export HF_HOME="$_ami_repo_root/.cache/hf"
export TORCH_HOME="$_ami_repo_root/.cache/torch"
export JOBLIB_TEMP_FOLDER="$_ami_repo_root/.cache/joblib"
export TMPDIR="$_ami_repo_root/.cache/tmp"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

mkdir -p "$HF_HOME" "$TORCH_HOME" "$JOBLIB_TEMP_FOLDER" "$TMPDIR"

echo "AMI offline env configured:"
echo "  HF_HOME=$HF_HOME"
echo "  TORCH_HOME=$TORCH_HOME"
echo "  JOBLIB_TEMP_FOLDER=$JOBLIB_TEMP_FOLDER"
echo "  TMPDIR=$TMPDIR"
echo "  HF_HUB_OFFLINE=$HF_HUB_OFFLINE"
echo "  TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE"
