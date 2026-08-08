// No `venv`/`conda` here on purpose: install.js pulls a complete, already-resolved
// standalone Python interpreter + site-packages out of the published .dmg (see its
// comment for why), so there is no environment to create or activate -- we just
// invoke that interpreter directly, exactly like scripts/launcher.sh does for the
// standalone .app.
module.exports = {
  daemon: true,
  run: [
    {
      method: "shell.run",
      params: {
        path: "app/app",
        env: {
          PYTHONPATH: "{{path.resolve(cwd, 'app/site-packages')}}",
          // Mirrors scripts/launcher.sh's Application-Support layout, just rooted in
          // this launcher's own folder instead -- keeps model weights/HF cache
          // self-contained per Pinokio convention rather than writing outside it.
          HY3DGEN_MODELS: "{{path.resolve(cwd, 'cache/models')}}",
          HF_HOME: "{{path.resolve(cwd, 'cache/hf_home')}}",
          HUGGINGFACE_HUB_CACHE: "{{path.resolve(cwd, 'cache/hf_home/hub')}}"
        },
        message: [
          "{{path.resolve(cwd, 'app/python/bin/python3.14')}} app.py"
        ],
        on: [{
          "event": "/http:\/\/[0-9.:]+/",
          "done": true
        }]
      }
    },
    {
      // This step sets the local variable 'url'.
      // This local variable will be used in pinokio.js to display the "Open WebUI" tab when the value is set.
      method: "local.set",
      params: {
        "url": "{{input.event[0]}}"
      }
    },
    {
      method: "proxy.start",
      params: {
        uri: "{{local.url}}",
        name: "Local Sharing"
      }
    }
  ]
}
