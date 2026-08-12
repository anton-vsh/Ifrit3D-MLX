from __future__ import annotations

from contextlib import contextmanager


@contextmanager
def report_hf_downloads(progress_callback, desc="Downloading model weights (first run only)"):
    """Hooks huggingface_hub's download progress bars to report through our
    own progress_callback(fraction, message) for the duration of the block,
    instead of only printing to a terminal the packaged app doesn't have.
    Without this, a cold model download leaves the Gradio UI frozen at
    whatever the last progress step said, indistinguishable from a hang.

    Why we patch more than one thing (hf_hub 1.11.0, verified empirically):
    - huggingface_hub._snapshot_download and huggingface_hub.hf_api do
      `from .utils.tqdm import tqdm as hf_tqdm` at import time, which binds the
      class object permanently. Replacing only the `huggingface_hub.utils.tqdm`
      module attribute has no effect on that pre-bound name — the actual
      download bars keep using the original class (callbacks never fire).
    - _create_progress_bar() decides via `issubclass(cls, tqdm)` whether to
      instantiate the caller-supplied class or the module default; the pre-bound
      hf_tqdm name is what flows in as `cls` for snapshot_download, so it has to
      be our subclass itself for the check to pass.
    - tqdm's `update()` returns immediately when the bar is `disable=True`
      (tqdm/std.py), before incrementing `n`, and the packaged app has no TTY,
      which tqdm's `disable=None` auto-detection resolves to True. Our subclass
      therefore keeps `disable=True` and does its own bookkeeping in `update()`,
      never calling `super().update()`. This also avoids re-entering tqdm's
      shared class-lock machinery: snapshot_download runs its per-file bars
      inside `tqdm.contrib.concurrent.thread_map`, which hands its lock to the
      worker threads — an enabled aggregate bar updating from a worker while the
      main thread iterates that same class lock deadlocks (verified).
    - Only byte-unit bars (the aggregated "Downloading (incomplete total...)"
      bar) are reported; the per-file "Fetching N files" bar (unit="it") and
      upload bars are skipped so the UI gets one smooth bytes signal."""

    if progress_callback is None:
        yield
        return

    import importlib

    # huggingface_hub.utils/__init__.py does `from .tqdm import tqdm`, which
    # rebinds the `tqdm` attribute on the `utils` package to the class —
    # shadowing the submodule for any later `import huggingface_hub.utils.tqdm`.
    # importlib.import_module reaches the real submodule via sys.modules.
    hf_tqdm_mod = importlib.import_module("huggingface_hub.utils.tqdm")
    original_cls = hf_tqdm_mod.tqdm

    class _CallbackTqdm(original_cls):
        def __init__(self, *args, **kwargs):
            # Keep the bar fully disabled: no rendering, no monitor thread,
            # no shared-lock interaction with thread_map's workers. Progress
            # is reported purely through our own update() bookkeeping.
            # NB: tqdm's __init__ with disable=True returns early and never
            # sets `unit`/`desc` (tqdm/std.py), so record them ourselves.
            self._cb_n = kwargs.get("initial", 0)
            self._cb_unit = kwargs.get("unit", "")
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)
            self.unit = self._cb_unit

        def update(self, n=1):
            if n > 0:
                self._cb_n += n
            total = getattr(self, "total", None) or 0
            if self._cb_unit == "B" and total > 0:
                frac = min(1.0, self._cb_n / total)
                mb_done = self._cb_n / (1024 * 1024)
                mb_total = total / (1024 * 1024)
                progress_callback(frac, f"{desc}: {mb_done:.0f}/{mb_total:.0f} MB")
            # deliberately no super().update(): disable=True would make it a
            # no-op anyway, and calling refresh() risks the class-lock deadlock.

    def _patch(targets):
        saved = []
        for mod, attr in targets:
            if hasattr(mod, attr):
                saved.append((mod, attr, getattr(mod, attr)))
                setattr(mod, attr, _CallbackTqdm)
        return saved

    targets = []
    # 1) the submodule attribute (used by _get_progress_bar_context when the
    #    caller passes no tqdm_class, e.g. diffusers' internal downloads).
    targets.append((hf_tqdm_mod, "tqdm"))
    # 2) the pre-bound `hf_tqdm` names (snapshot_download + API upload paths).
    for mod_name in (
        "huggingface_hub._snapshot_download",
        "huggingface_hub.hf_api",
        "huggingface_hub._commit_api",
    ):
        try:
            targets.append((importlib.import_module(mod_name), "hf_tqdm"))
        except ImportError:
            pass

    saved = _patch(targets)
    try:
        yield
    finally:
        for mod, attr, orig in saved:
            setattr(mod, attr, orig)
