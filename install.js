// m3dium ships 4 custom Metal-compiled Python extensions (mtldiffrast, mtlbvh,
// cumesh, flex-gemm) plus a Swift/MLX shape binary -- none of these have PyPI
// wheels, and compiling them requires Xcode Command Line Tools, which defeats
// the point of a 1-click launcher. So instead of a normal `uv sync`/`pip install`
// from source, this downloads the same signed .dmg the project already publishes
// on GitHub Releases (built via the project's own scripts/build_app.sh) and pulls
// the *already-compiled* interpreter + site-packages out of it -- no compilation
// on the user's machine at all. See start.js for why this also means no `venv`.
module.exports = {
  run: [
    {
      method: "fs.download",
      params: {
        uri: "https://github.com/anton-vsh/m3dium/releases/latest/download/m3dium.dmg",
        path: "m3dium.dmg"
      }
    },
    {
      // No custom -mountpoint: verified directly (see conversation/commit history)
      // that hdiutil refuses to create a mountpoint inside this launcher's own
      // folder when PINOKIO_HOME lives on an external volume ("Access denied") --
      // letting hdiutil pick its own default location under /Volumes avoids that
      // entirely. The volume label is fixed by scripts/make_dmg.sh's own
      // `-volname "m3dium"`, so the mount path is always exactly "/Volumes/m3dium"
      // (or "/Volumes/m3dium 1" etc. if that name is somehow already taken) --
      // captured here the same way start.js captures the server URL, rather than
      // hardcoded, so a name collision doesn't silently break the install.
      method: "shell.run",
      params: {
        message: [
          "hdiutil attach m3dium.dmg -nobrowse"
        ],
        on: [{
          "event": "/(\\/Volumes\\/[^\\r\\n]+)/",
          "done": true
        }]
      }
    },
    {
      method: "local.set",
      params: {
        mount: "{{input.event[1]}}"
      }
    },
    // Using shell `cp -RL` here, not the `fs.copy` API: verified directly (see
    // conversation/commit history) that `fs.copy` hangs indefinitely on this
    // specific tree (21+ minutes, near-zero CPU, zero files landed) -- almost
    // certainly the standalone Python distribution's internal symlinks (e.g.
    // versioned dylib links), which scripts/build_app.sh already had to special-case
    // with `-L` for the exact same reason when assembling the .dmg in the first
    // place (see that script's own comment). `-L` dereferences symlinks into real
    // files/dirs instead of copying the link itself, which is also what we want here
    // (a symlink pointing back into the now-unmounted, ephemeral DMG volume would be
    // dangling as soon as we detach it below).
    {
      method: "shell.run",
      params: {
        message: [
          "mkdir -p app",
          "cp -RL \"{{local.mount}}/m3dium.app/Contents/Resources/python\" app/python",
          "cp -RL \"{{local.mount}}/m3dium.app/Contents/Resources/site-packages\" app/site-packages",
          "cp -RL \"{{local.mount}}/m3dium.app/Contents/Resources/app\" app/app"
        ]
      }
    },
    {
      method: "shell.run",
      params: {
        message: [
          "hdiutil detach \"{{local.mount}}\" -force"
        ]
      }
    },
    {
      method: "fs.rm",
      params: {
        path: "m3dium.dmg"
      }
    },
    {
      method: "notify",
      params: {
        html: "Click the 'start' tab to get started!"
      }
    }
  ]
}
