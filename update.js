// There's no git repo to `git pull` here (see install.js) -- "update" means
// re-downloading the latest published .dmg and re-extracting it, so this just
// clears the old copy and re-runs install.js.
module.exports = {
  run: [{
    method: "fs.rm",
    params: {
      path: "app"
    }
  }, {
    method: "script.start",
    params: {
      uri: "install.js"
    }
  }]
}
