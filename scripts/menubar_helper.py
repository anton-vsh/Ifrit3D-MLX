"""Menu bar wrapper around app.py.

Opening the UI in a plain browser tab means the app has no Dock icon and no
menu bar presence, so closing the tab doesn't quit it and there's no visible
sign it's still running. This gives it a small status-bar icon with just
"Open UI" (reopen a closed tab) and "Quit" (stop everything cleanly).

Runs standalone in dev (`uv run python scripts/menubar_helper.py`, app.py one
directory up) and inside the packaged bundle (menubar_helper.py copied next
to app.py in Contents/Resources/app/, same directory).
"""

import os
import re
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

import rumps

PORT_RE = re.compile(r"Running on local URL:\s*http://127\.0\.0\.1:(\d+)")


def _find_app_py() -> Path:
    here = Path(__file__).resolve().parent
    candidate = here / "app.py"
    if candidate.exists():
        return candidate
    candidate = here.parent / "app.py"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not locate app.py near {here}")


def _log_path() -> Path:
    log_dir = Path.home() / "Library" / "Application Support" / "Ifrit3D-MLX" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "app.log"


class Ifrit3DMenuBarApp(rumps.App):
    def __init__(self):
        super().__init__("[I3D]", quit_button=None)
        self.menu = ["Open UI", "Quit"]
        self.app_py = _find_app_py()
        self.port = None
        self.proc = None
        self.log_file = open(_log_path(), "a", buffering=1)
        self._start_app()
        self.watcher = rumps.Timer(self._poll_process, 2)
        self.watcher.start()

    def _start_app(self):
        self.log_file.write(f"\n--- launching {self.app_py} at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"  # child's stdout is a pipe, not a TTY, so Python fully-buffers it by default
        self.proc = subprocess.Popen(
            [sys.executable, str(self.app_py)],
            cwd=str(self.app_py.parent),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        threading.Thread(target=self._read_output, daemon=True).start()

    def _read_output(self):
        for line in self.proc.stdout:
            self.log_file.write(line)
            if self.port is None:
                m = PORT_RE.search(line)
                if m:
                    self.port = int(m.group(1))

    def _poll_process(self, _timer):
        if self.proc.poll() is not None:
            # app.py exited on its own (e.g. the in-page Shutdown Server
            # button) — don't leave an orphaned menu bar icon behind.
            self.watcher.stop()
            rumps.quit_application()

    @rumps.clicked("Open UI")
    def open_ui(self, _sender):
        port = self.port or 7860
        webbrowser.open(f"http://127.0.0.1:{port}")

    @rumps.clicked("Quit")
    def quit_app(self, _sender):
        self.watcher.stop()
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        rumps.quit_application()


if __name__ == "__main__":
    Ifrit3DMenuBarApp().run()
