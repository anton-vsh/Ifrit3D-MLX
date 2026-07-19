from __future__ import annotations

from contextlib import contextmanager


@contextmanager
def report_hf_downloads(progress_callback, desc="Downloading model weights (first run only)"):
    """Hooks huggingface_hub's download progress bars to report through our
    own progress_callback(fraction, message) for the duration of the block,
    instead of only printing to a terminal the packaged app doesn't have.
    Without this, a cold model download leaves the Gradio UI frozen at
    whatever the last progress step said, indistinguishable from a hang.

    huggingface_hub's downloader defaults to a module-level tqdm class when
    the caller doesn't pass tqdm_class explicitly (diffusers' from_pretrained
    doesn't), so patching that one class here catches both our own direct
    snapshot_download calls and diffusers' internal ones."""
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
        def update(self, n=1):
            result = super().update(n)
            total = self.total or 0
            if total > 0:
                frac = min(1.0, self.n / total)
                mb_done = self.n / (1024 * 1024)
                mb_total = total / (1024 * 1024)
                progress_callback(frac, f"{desc}: {mb_done:.0f}/{mb_total:.0f} MB")
            return result

    hf_tqdm_mod.tqdm = _CallbackTqdm
    try:
        yield
    finally:
        hf_tqdm_mod.tqdm = original_cls
