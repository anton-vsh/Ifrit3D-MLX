#!/bin/bash
# Ifrit3D-MLX.app's CFBundleExecutable. Resolves paths relative to its own
# location so it works regardless of where the .app is installed, sets up a
# writable app-support location for model weights/HF cache (the bundle
# itself is read-only once installed to /Applications), and hands off to the
# menu bar helper — never app.py directly, so quitting has a real affordance.
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTENTS_DIR="$(dirname "$DIR")"
RESOURCES_DIR="$CONTENTS_DIR/Resources"

APP_SUPPORT="$HOME/Library/Application Support/Ifrit3D-MLX"
mkdir -p "$APP_SUPPORT/models" "$APP_SUPPORT/hf_home" "$APP_SUPPORT/logs"

# Redirect this launcher's own output so a failure before menubar_helper.py's
# internal logging kicks in isn't silently lost (there's no visible Terminal
# window in the packaged app).
exec >> "$APP_SUPPORT/logs/launcher.log" 2>&1
echo "--- launcher started at $(date) ---"

export HY3DGEN_MODELS="$APP_SUPPORT/models"
export HF_HOME="$APP_SUPPORT/hf_home"
export HUGGINGFACE_HUB_CACHE="$APP_SUPPORT/hf_home/hub"
export PYTHONPATH="$RESOURCES_DIR/site-packages"

exec "$RESOURCES_DIR/python/bin/python3.14" "$RESOURCES_DIR/app/menubar_helper.py"
