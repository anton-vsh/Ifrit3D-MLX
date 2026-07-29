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

# Zero-shot metallic/roughness estimate for the reference photo, as a cheap postfactum
# alternative to a full PBR (2.1) diffusion re-paint: reuses the same zero-shot CLIP
# technique as subject_classifier.py (which names the subject for the SD Turbo prompt),
# but with materials vocabulary instead of an object-category one -- zero-shot CLIP
# against MINC-style material labels is a known-reasonable technique for coarse
# material classification. This can't localize either property on a mixed-material
# object (it's one score for the whole photo, applied flat across the whole mesh), and
# it's a semantic guess, not a measurement -- a real PBR pass from the 2.1 checkpoint
# will always be more accurate. This exists for when that cost isn't worth it and flat
# metallic/roughness values are an acceptable tradeoff. Verified against real photos
# (see conversation/commit history): correctly low-metallic on ceramic and stone,
# correctly high-metallic on an all-metal-hull model, and -- the known failure mode --
# confidently high-metallic on a mixed object (red-lacquer-body/gold-trim figurine)
# where only the trim is actually metal, since a single global score can't split them.

import threading

import torch
from PIL import Image

_CLIP_LOCK = threading.Lock()

# Each pair is scored independently (softmax within the pair only), so the two
# properties don't compete against each other for probability mass. Kept
# deliberately short: a long list of near-synonyms (as subject_classifier.py
# uses for its 1000+-way object vocabulary) helps there because it needs to
# *name* the right one; here only each group's summed mass matters, so
# redundant synonyms would just double-count that group without adding
# information.
METAL_LABELS = [
    "polished metal", "brushed metal", "chrome", "steel", "gold", "silver",
    "a metal surface", "a metallic object",
]
NONMETAL_LABELS = [
    "ceramic", "glazed ceramic", "fabric", "cloth", "wood", "plastic",
    "stone", "polished stone", "glass", "paper", "leather", "rubber",
    "skin", "painted surface", "matte plastic",
]
GLOSSY_LABELS = [
    "a glossy polished surface", "a shiny reflective surface",
    "a mirror-like surface", "a glazed surface", "a lacquered surface",
    "a wet-looking surface",
]
MATTE_LABELS = [
    "a matte surface", "a rough unpolished surface", "a dull surface",
    "a chalky surface", "a textured fabric surface", "a porous surface",
    "an unfinished stone surface",
]


class MaterialClassifier:
    """Lazy-loaded zero-shot CLIP metallic/roughness estimate. `metallic(image)`
    and `roughness(image)` each return a float in [0, 1] (metallic: 0 = confidently
    non-metallic, 1 = confidently metallic; roughness: 0 = confidently glossy,
    1 = confidently matte/rough). One pair of scores for the whole photo -- see
    module docstring for the known limitation and the real-PBR alternative when
    that's too coarse."""

    def __init__(self, device="cpu", model_id="openai/clip-vit-base-patch32"):
        from transformers import CLIPModel, CLIPProcessor

        self.device = device
        self.model = CLIPModel.from_pretrained(model_id).to(device)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)

        with torch.no_grad():
            self._metal_split = len(METAL_LABELS)
            self._rough_split = len(GLOSSY_LABELS)
            metal_queries = [f"a photo of {l}" for l in METAL_LABELS + NONMETAL_LABELS]
            rough_queries = [f"a photo of {l}" for l in GLOSSY_LABELS + MATTE_LABELS]
            self._metal_text = self._encode_text(metal_queries)
            self._rough_text = self._encode_text(rough_queries)

    def _encode_text(self, queries):
        # Not model.get_text_features(**inputs) -- on this transformers version
        # (5.x) it returns the raw BaseModelOutputWithPooling instead of the
        # projected embedding tensor the classic CLIP API implies. Going
        # through text_model + text_projection directly sidesteps that and is
        # what get_text_features itself would do internally in the versions
        # where it works as documented.
        inputs = self.processor(text=queries, return_tensors="pt", padding=True).to(self.device)
        pooled = self.model.text_model(**inputs).pooler_output
        feats = self.model.text_projection(pooled)
        return feats / feats.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def _image_features(self, image: Image.Image):
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt").to(self.device)
        pooled = self.model.vision_model(**inputs).pooler_output
        feats = self.model.visual_projection(pooled)
        return feats / feats.norm(dim=-1, keepdim=True)

    def _score(self, image_features, text_features, split: int) -> float:
        logit_scale = self.model.logit_scale.exp()
        logits = (logit_scale * image_features @ text_features.T)[0]
        probs = logits.softmax(dim=-1)
        return float(probs[:split].sum())

    @torch.no_grad()
    def metallic(self, image: Image.Image) -> float:
        with _CLIP_LOCK:
            feats = self._image_features(image)
            return self._score(feats, self._metal_text, self._metal_split)

    @torch.no_grad()
    def roughness(self, image: Image.Image) -> float:
        with _CLIP_LOCK:
            feats = self._image_features(image)
            return 1.0 - self._score(feats, self._rough_text, self._rough_split)

    @torch.no_grad()
    def estimate(self, image: Image.Image) -> tuple[float, float]:
        """Both scores from a single image encode -- (metallic, roughness)."""
        with _CLIP_LOCK:
            feats = self._image_features(image)
            metallic = self._score(feats, self._metal_text, self._metal_split)
            roughness = 1.0 - self._score(feats, self._rough_text, self._rough_split)
            return metallic, roughness
