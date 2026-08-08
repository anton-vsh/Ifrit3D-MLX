# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

from PIL import Image
from rembg import remove, new_session


class BackgroundRemover():
    def __init__(self, progress_callback=None):
        # new_session() downloads the ONNX model (~176MB) via rembg's own
        # pooch-based downloader on first use -- not huggingface_hub, so
        # hf_progress.report_hf_downloads' tqdm patch can't see it, and
        # rembg's new_session() takes no progress hook of its own to attach
        # to. One descriptive message before the (blocking) call is the most
        # we can surface here -- not a real byte-level bar like the
        # huggingface_hub-backed downloads elsewhere, but still turns several
        # seconds-to-minutes of silence into "yes, something is happening."
        if progress_callback is not None:
            progress_callback(0.0, "Loading background-removal model (first run only)...")
        self.session = new_session()

    def __call__(self, image: Image.Image):
        output = remove(image, session=self.session, bgcolor=[255, 255, 255, 0])
        return output


# new_session() loads an ONNX Runtime model from disk — real cost, not a
# no-op. A single generation can call into rembg from both the shape and
# paint stages; sharing one instance avoids paying that load twice for
# what is otherwise a stateless, deterministic operation.
_shared_remover = None


def get_background_remover(progress_callback=None) -> "BackgroundRemover":
    global _shared_remover
    if _shared_remover is None:
        _shared_remover = BackgroundRemover(progress_callback=progress_callback)
    return _shared_remover
