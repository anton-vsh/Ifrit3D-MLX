#!/usr/bin/env bash
# Assembles Ifrit3D-MLX.app from the current source tree + a fully-resolved
# Python environment. Run from anywhere; always operates on the repo this
# script lives in.
#
# Rather than trying to make a relocated venv's own pyvenv.cfg/shebangs work
# (fragile — absolute paths baked in by design), this copies uv's managed
# standalone interpreter and the already-resolved site-packages as two
# separate directories, and the launcher points a plain PYTHONPATH at the
# site-packages copy. No venv activation, no relocation tricks needed.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

APP_NAME="Ifrit3D-MLX"
PY_VERSION="3.14"
BUILD_DIR="$REPO_ROOT/dist"
APP_BUNDLE="$BUILD_DIR/$APP_NAME.app"

VERSION="$(uv run python3 -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])")"
echo "==> Building $APP_NAME.app v$VERSION"

echo "==> Resolving project dependencies (uv sync — also builds the 4 Metal extensions)"
uv sync

if [ ! -f ".venv/pyvenv.cfg" ]; then
  echo "ERROR: .venv/pyvenv.cfg not found after uv sync" >&2
  exit 1
fi

# Tie the copied interpreter directly to whatever .venv actually resolved
# against, rather than re-deriving it separately — guarantees the compiled
# extensions we're about to copy (built against .venv's interpreter) are
# ABI-compatible with the interpreter we ship.
VENV_HOME="$(grep '^home = ' .venv/pyvenv.cfg | sed 's/^home = //')"
MANAGED_PYTHON_DIR="$(dirname "$VENV_HOME")"
if [ ! -x "$VENV_HOME/python$PY_VERSION" ]; then
  echo "ERROR: expected interpreter not found at $VENV_HOME/python$PY_VERSION" >&2
  exit 1
fi
echo "==> Using managed interpreter: $MANAGED_PYTHON_DIR"

echo "==> Cleaning previous build"
rm -rf "$APP_BUNDLE"
mkdir -p "$APP_BUNDLE/Contents/MacOS" "$APP_BUNDLE/Contents/Resources/app"

echo "==> Copying standalone interpreter into the bundle"
# -L: MANAGED_PYTHON_DIR is itself a symlink (e.g. cpython-3.14-macos-aarch64-none
# -> cpython-3.14.6-macos-aarch64-none) — plain cp -R on macOS copies the
# symlink itself, not its contents, which would silently bake a dev-machine
# -only absolute path into the bundle. -L forces dereferencing.
cp -RL "$MANAGED_PYTHON_DIR" "$APP_BUNDLE/Contents/Resources/python"

echo "==> Copying resolved site-packages (already built, including the 4 Metal extensions)"
cp -RL ".venv/lib/python$PY_VERSION/site-packages" "$APP_BUNDLE/Contents/Resources/site-packages"

echo "==> Copying application source"
cp -R hy3dgen "$APP_BUNDLE/Contents/Resources/app/"
cp -R shape "$APP_BUNDLE/Contents/Resources/app/"
cp app.py main.py "$APP_BUNDLE/Contents/Resources/app/"
cp scripts/menubar_helper.py "$APP_BUNDLE/Contents/Resources/app/"

echo "==> Writing Info.plist"
sed "s/__VERSION__/$VERSION/g" scripts/Info.plist.template > "$APP_BUNDLE/Contents/Info.plist"

echo "==> Installing launcher"
cp scripts/launcher.sh "$APP_BUNDLE/Contents/MacOS/$APP_NAME"
chmod +x "$APP_BUNDLE/Contents/MacOS/$APP_NAME"

if [ -f scripts/AppIcon.icns ]; then
  cp scripts/AppIcon.icns "$APP_BUNDLE/Contents/Resources/AppIcon.icns"
else
  echo "    (no scripts/AppIcon.icns found — app will use the generic icon; add one later)"
fi

BUNDLE_SIZE="$(du -sh "$APP_BUNDLE" | cut -f1)"
echo "==> Done: $APP_BUNDLE ($BUNDLE_SIZE, unsigned)"
echo "    Test with: open \"$APP_BUNDLE\""
echo "    Logs at:   ~/Library/Application Support/Ifrit3D-MLX/logs/"
