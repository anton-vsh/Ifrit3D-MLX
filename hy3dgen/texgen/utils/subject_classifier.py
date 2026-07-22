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

# Zero-shot subject classification for the reference photo, used to build a
# content-aware SD Turbo upscale prompt instead of a generic one. Verified:
# a generic prompt ("highly detailed, sharp, photorealistic texture") gets
# largely ignored by SD Turbo's img2img pass here since guidance_scale=0
# (no CFG) makes the text conditioning's influence weak -- but naming the
# actual subject measurably recovers fine detail (sharper eyes, visible
# individual whisker strands on a cat test case) that the generic prompt
# and the pre-upscale base image both lacked.

import os
import threading

import torch
from PIL import Image

_CLIP_LOCK = threading.Lock()

# Broad general-category coverage rather than fine-grained (no dog/cat
# breeds, no car models) -- the SD Turbo prompt only needs a grounding
# noun, not a precise label, and a very large or overly-granular candidate
# list dilutes softmax confidence across near-duplicate categories without
# actually improving the prompt. Zero-shot CLIP compares the image against
# every candidate in one batched forward pass, so this list's size barely
# affects cost (still well under 0.1s even at ~150 candidates). Low-
# confidence results fall back to the generic prompt entirely rather than
# force-picking a poor match for a subject outside this list.
CANDIDATE_VOCABULARY = [
    # people
    "a person", "a face", "a portrait", "a child", "a hand", "a group of people",
    # common pets and animals
    "a cat", "a dog", "a bird", "a fish", "a horse", "a rabbit",
    "a reptile", "an insect", "a farm animal", "a wild animal", "an animal",
    # vehicles
    "a car", "a truck", "a motorcycle", "a bicycle", "an airplane",
    "a boat", "a train", "a vehicle",
    # furniture
    "a chair", "a table", "a sofa", "a bed", "a cabinet", "a shelf", "furniture",
    # household objects
    "a lamp", "a vase", "a clock", "a mirror", "a pillow", "a mug",
    "a plate", "a bowl", "kitchenware", "a candle",
    # electronics
    "a phone", "a laptop", "a camera", "headphones", "a television",
    "a speaker", "a game controller", "an electronic device",
    # clothing and accessories
    "clothing", "a shirt", "a dress", "a shoe", "a hat", "a bag",
    "jewelry", "a watch", "glasses", "a mask",
    # food and drink
    "fruit", "a vegetable", "bread", "a cake", "a bottle", "a drink", "food",
    # tools and equipment
    "a tool", "a machine", "a weapon", "a musical instrument",
    # sports and recreation
    "a ball", "sports equipment", "a toy", "a board game",
    # nature and plants
    "a tree", "a flower", "a plant", "a rock", "a shell", "a fossil",
    # buildings and structures
    "a building", "a house", "a bridge", "a monument",
    # containers
    "a box", "a basket", "a jar", "a container",
    # art and decor
    "a statue", "a sculpture", "a painting", "an ornament", "a figurine",
    "a ceramic object", "a robot",
    # fallback
    "a plain object", "an abstract shape",
]


def _add_article(name: str) -> str:
    return f"an {name}" if name[0].lower() in "aeiou" else f"a {name}"


def _imagenet_labels():
    # The standard ImageNet-1000 class list, already bundled locally as
    # metadata on any torchvision classification weights enum (no network
    # call, no extra download) -- the same taxonomy CLIP's own paper
    # benchmarks zero-shot accuracy against. Adds real breadth (accordion,
    # ambulance, apiary, backpack, bakery, ...) our hand-written list above
    # doesn't have.
    from torchvision.models import ResNet50_Weights
    names = ResNet50_Weights.DEFAULT.meta["categories"]
    return [_add_article(name.lower()) for name in names]


# dict.fromkeys dedupes while preserving order -- a handful of hand-written
# labels above (e.g. "a vase", "a laptop") coincide with ImageNet's own
# class names verbatim.
CANDIDATE_VOCABULARY = list(dict.fromkeys(CANDIDATE_VOCABULARY + _imagenet_labels()))

# CLIP-side query text is templated identically for every candidate,
# regardless of the label's own wording -- applied uniformly so no
# candidate is favored purely for reading like a fuller sentence. The
# returned *label* (CANDIDATE_VOCABULARY[i]) stays a plain noun phrase,
# since that's what gets embedded directly into the SD Turbo prompt.
CANDIDATE_QUERIES = [f"a photo of {label}" for label in CANDIDATE_VOCABULARY]

# A raw top-1 probability threshold stopped being a meaningful signal once
# the vocabulary grew to ~1000 candidates: CLIP always confidently picks
# *something*, and correct answers now routinely split across several
# near-duplicate labels (e.g. a photo of a cat scoring "a tabby" 31%,
# "a cat" 27%, "an egyptian cat" 17% -- all correct, none individually
# high). Verified directly: random noise can score a HIGHER top-1 (~20%)
# than a genuinely identifiable photo whose correct answer happens to be
# split several ways (~10%) -- top-1 alone can't tell those apart.
#
# The cumulative probability mass across the top-K candidates is a better
# signal: a real, identifiable subject concentrates probability into a
# handful of near-synonyms even when split, while noise/uncertain inputs
# spread it more thinly even when one candidate spikes by chance. Verified
# on real cases: cat=87%, urinal=70%, a harder portrait case=50%, vs.
# random noise=38% for the same top-8 sum -- this ordering is correct
# where raw top-1 was not. Threshold set with margin below the weakest
# real case and above the noise floor observed; not exhaustively tuned
# across a large diverse image set, so revisit if false accepts/rejects
# show up in practice.
TOP_K_FOR_CONFIDENCE = 8
CONFIDENCE_THRESHOLD = 0.45


class SubjectClassifier:
    """Lazy-loaded zero-shot CLIP classifier: PIL image -> short subject
    label (e.g. "a cat"), or None if the top-K candidates' combined
    probability mass doesn't clear CONFIDENCE_THRESHOLD (see comment above
    it for why this is top-K mass rather than a single top-1 score)."""

    def __init__(self, device="cpu", model_id="openai/clip-vit-base-patch32"):
        from transformers import CLIPModel, CLIPProcessor

        self.device = device
        self.model = CLIPModel.from_pretrained(model_id).to(device)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)

    @torch.no_grad()
    def __call__(self, image: Image.Image):
        with _CLIP_LOCK:
            inputs = self.processor(
                text=CANDIDATE_QUERIES, images=image.convert("RGB"),
                return_tensors="pt", padding=True,
            ).to(self.device)
            out = self.model(**inputs)
            probs = out.logits_per_image.softmax(dim=-1)[0]

        top_k = probs.topk(min(TOP_K_FOR_CONFIDENCE, len(CANDIDATE_VOCABULARY)))
        if float(top_k.values.sum()) < CONFIDENCE_THRESHOLD:
            return None
        return CANDIDATE_VOCABULARY[int(top_k.indices[0])]
