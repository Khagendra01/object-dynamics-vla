"""Frozen image+instruction hidden-state cache. No fallback to generated text.

Model execution is integration code, not verified in the packaging environment.
It explicitly fails when the installed Transformers/backend lacks the model API.
"""
import argparse
import copy
from pathlib import Path
import numpy as np
from PIL import Image
from odvla.data import load_manifest, load_episode, safe_path, write_json


def pick_last_hidden(outputs):
    states = getattr(outputs, "hidden_states", None)
    if states is None:
        nested = getattr(outputs, "language_model_outputs", None)
        states = getattr(nested, "hidden_states", None)
    if states is None or not len(states):
        raise RuntimeError("Backend returned no language hidden states. Add a verified backbone hook; do not substitute text.")
    hidden = states[-1]
    if hidden.ndim != 3:
        raise RuntimeError("Expected hidden state [batch, sequence, width]")
    return hidden


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True); p.add_argument("--out", required=True)
    p.add_argument("--model", default="meta-models/Muse-Glimmer-30B")
    p.add_argument("--model-path", help="Optional local snapshot; provenance retains --model and --revision")
    p.add_argument("--revision", required=True, help="Pinned Hugging Face model revision/commit")
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--max-image-edge", type=int, default=384)
    a = p.parse_args()
    import torch
    import transformers
    from transformers import AutoProcessor
    if not torch.cuda.is_available(): raise RuntimeError("Use configured CUDA GPU; try an 80GB cloud GPU if BF16 does not fit")
    if a.stride < 1 or a.max_image_edge < 1: raise ValueError("stride and max-image-edge must be positive")
    out = Path(a.out)
    if out.exists() and any(out.iterdir()): raise FileExistsError("Use a new output dataset")
    factory = getattr(transformers, "AutoModelForMultimodalLM", None)
    if factory is None:
        factory = getattr(transformers, "AutoModelForImageTextToText", None)
    if factory is None: raise RuntimeError("Install a Transformers version with Muse Glimmer support")
    model_source = a.model_path or a.model
    processor = AutoProcessor.from_pretrained(model_source, revision=a.revision, trust_remote_code=False)
    model = factory.from_pretrained(model_source, revision=a.revision, dtype=torch.bfloat16,
                                    device_map="auto", trust_remote_code=False)
    model.eval(); model.requires_grad_(False)
    m = copy.deepcopy(load_manifest(a.dataset))
    (out / "episodes").mkdir(parents=True, exist_ok=True)
    width = None
    for entry in m["episodes"]:
        ep = load_episode(a.dataset, entry, m)
        if "rgb" not in ep: raise ValueError("RGB observations required")
        cached, rows = None, []
        for t, rgb in enumerate(ep["rgb"]):
            if t % a.stride == 0:
                image = Image.fromarray(rgb)
                image.thumbnail((a.max_image_edge, a.max_image_edge))
                messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                                            {"type": "text", "text": entry["instruction"]}]}]
                batch = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                       return_dict=True, return_tensors="pt")
                batch = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in batch.items()}
                with torch.inference_mode():
                    outputs = model(**batch, output_hidden_states=True, use_cache=False, return_dict=True)
                    hidden = pick_last_hidden(outputs)
                    # Single unpadded sample: final input token summarizes image+instruction.
                    cached = hidden[0, -1].float().cpu().numpy()
                width = len(cached)
                print(entry["id"], t, "hidden_width", width, flush=True)
            rows.append(cached.copy())  # Carry past context forward; never interpolate from future frames.
        ep["context"] = np.stack(rows).astype(np.float32)
        destination = safe_path(out, entry["file"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, **ep)
    m["context_provenance"] = {"source": a.model, "revision": a.revision,
        "local_snapshot": a.model_path,
        "transformers_version": transformers.__version__, "width": width,
        "pooling": "last language input token", "stride": a.stride,
        "causal": True, "stale_context_policy": "carry forward from most recent processed frame",
        "max_image_edge": a.max_image_edge, "backbone_frozen": True}
    write_json(out / "manifest.json", m)


if __name__ == "__main__": main()
