// Removes the interpreter/site-packages/app copy pulled from the .dmg -- does NOT
// touch cache/ (downloaded model weights), so a reset doesn't force a multi-GB
// re-download. Run "Install" again afterward to restore app/.
module.exports = {
  run: [{
    method: "fs.rm",
    params: {
      path: "app"
    }
  }]
}
