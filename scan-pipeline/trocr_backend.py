import os
import threading
from pathlib import Path

import cv2
import numpy as np

# TrOCR backend: reads the whole field crop, no separate text detector. The
# fine-tuned model (tools/train_trocr.py) is trained on whole fields, blank ones
# included, it returns "" for an empty box and ignores writing that spills
# in from neighboring fields. 
#
# TROCR_BEAMS=1 is ~2-3x faster than the default 4, usually for a small accuracy cost.

REPO_ROOT = Path(__file__).resolve().parent.parent

def model_source() -> str:
    model_dir = os.environ.get("TROCR_MODEL_DIR")
    if not model_dir:
        return os.environ.get("TROCR_MODEL", "microsoft/trocr-base-handwritten")
    path = Path(model_dir).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.is_dir():
        raise FileNotFoundError(f"TROCR_MODEL_DIR is not a directory: {path}")
    return str(path)


MODEL_NAME = model_source()
NUM_BEAMS = int(os.environ.get("TROCR_BEAMS", "4"))
MAX_TOKENS = 48
BATCH = int(os.environ.get("TROCR_BATCH", "1"))    # fields per GPU call; batching was slower on an RTX 4060
INSET = 0.03         # trimmed from each side of the field crop, clears most printed borders
EDGE_JUNK = " .,:;'\"()[]"

_lock = threading.Lock()
_model = None


# Loads TrOCR once. Uses the GPU when PyTorch can see one.
def get_model():
    global _model
    if _model is None:
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
        model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME).to(device).eval()
        _model = (processor, model, device)
    return _model


# The image TrOCR sees for a field, used for both training and reading. BGR in, BGR out.
def field_image(crop: np.ndarray) -> np.ndarray | None:
    if crop is None or crop.size == 0 or min(crop.shape[:2]) < 4:
        return None
    h, w = crop.shape[:2]
    dy, dx = int(h * INSET), int(w * INSET)
    return crop[dy:h - dy, dx:w - dx]


# Greedy: the least confident token, ignoring the padding after a sequence ends.
def confidences(out, eos_token_id: int) -> list[float]:
    import torch

    if NUM_BEAMS > 1:
        return torch.exp(out.sequences_scores).tolist()
    tokens = out.sequences[:, 1:]
    probs = torch.stack(out.scores, 1).softmax(-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
    is_eos = tokens == eos_token_id
    probs[(is_eos.cumsum(1) - is_eos.int()) > 0] = 1.0
    return probs.min(1).values.tolist()


# Text and confidence for many fields, read in batches on one device. Without
# with_confidence the per-token scores aren't kept, saves a lot of GPU memory.
def read_many(crops: list[np.ndarray], field_keys: list[str] | None = None,
              with_confidence: bool = True) -> list[tuple[str, float]]:
    import torch

    results = [("", 1.0)] * len(crops)
    images = [(i, image) for i, image in enumerate(map(field_image, crops)) if image is not None]
    if not images:
        return results
    processor, model, device = get_model()
    for start in range(0, len(images), BATCH):
        chunk = images[start:start + BATCH]
        with _lock, torch.no_grad():
            pixels = processor(images=[cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for _, image in chunk],
                               return_tensors="pt").pixel_values.to(device)
            out = model.generate(pixels, num_beams=NUM_BEAMS, max_new_tokens=MAX_TOKENS,
                                 return_dict_in_generate=True, output_scores=with_confidence)
            scores = (confidences(out, model.generation_config.eos_token_id) if with_confidence
                      else [1.0] * len(chunk))
        texts = processor.batch_decode(out.sequences, skip_special_tokens=True)
        for (i, _), text, score in zip(chunk, texts, scores):
            results[i] = (text.strip(EDGE_JUNK), float(score))
    return results


def read(crop: np.ndarray, field_key: str = "") -> tuple[str, float]:
    return read_many([crop], [field_key])[0]


def ocr(crop: np.ndarray, field_key: str = "") -> str:
    return read(crop, field_key)[0]



# One model on one device: calls are serialized, PyTorch uses the cores itself.
def workers() -> int:
    return 1