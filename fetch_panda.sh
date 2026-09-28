#!/usr/bin/env bash
# Fetch the Franka Emika Panda model from MuJoCo Menagerie (only that folder, pinned commit).
set -euo pipefail
cd "$(dirname "$0")"
DEST=third_party/mujoco_menagerie
if [ -d "$DEST/franka_emika_panda" ]; then
  echo "Panda model already in $DEST"; exit 0
fi
git clone --filter=blob:none --no-checkout https://github.com/google-deepmind/mujoco_menagerie "$DEST"
git -C "$DEST" sparse-checkout set franka_emika_panda
git -C "$DEST" checkout c96a32d
echo "Panda model ready in $DEST/franka_emika_panda"
