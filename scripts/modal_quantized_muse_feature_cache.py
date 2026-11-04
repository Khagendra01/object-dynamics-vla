"""Resumable frozen-NF4 Muse feature cache for the aligned demonstration dataset.

The cache stores exactly the BF16 hidden-state bits that the training bridge
consumes, so later training/evaluation jobs never load the 30B Muse model again.
No simulator actions are executed and the untouched test split is rejected.
"""
import modal
from scripts.modal_quantized_muse_preflight import image, volume, MUSE_REVISION

app = modal.App("odvla-quantized-muse-feature-cache")

ALIGNED_FINGERPRINT = "86d9fdf00329e8f18df23bab67d70dbfae914bc68c6884cdd8938f8518386bed"
REFERENCE_REPORT = "/eval/quantized_muse_aligned7500_v1/report.json"
CHAT_DATE = "2026-10-30"


def validate_request(name, dataset, split, max_runtime_seconds):
    if not name or len(name) > 80 or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
        raise ValueError("Invalid cache name")
    if not dataset or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in dataset):
        raise ValueError("Invalid dataset name")
    if split not in ("train", "val", "all"):
        raise ValueError("split must be train, val or all (train+val; test is never read)")
    if not 60 <= max_runtime_seconds <= 4800:
        raise ValueError("Runtime cap must be 60..4800 seconds")


@app.function(image=image, gpu="H200", cpu=4, memory=131072, volumes={"/data": volume},
              timeout=5400, max_containers=1, single_use_containers=True)
def build(name: str = "muse_nf4_aligned_train_v1", dataset: str = "muse_aligned_development_v1",
          split: str = "train", max_runtime_seconds: int = 2400):
    import hashlib
    import importlib.metadata
    import json
    import os
    import time
    from pathlib import Path
    import numpy as np
    import torch
    import transformers
    from PIL import Image
    from odvla.data import file_hash, load_episode, read_json, validate_splits
    from odvla.vla.muse_context import FrozenMuseTokens, backend_snapshot

    validate_request(name, dataset, split, max_runtime_seconds)
    started = time.monotonic()
    root = Path("/data/libero") / dataset
    splits = read_json(root / "splits.json")
    manifest = validate_splits(root, splits)
    if splits["fingerprint"] != ALIGNED_FINGERPRINT:
        raise ValueError("Refusing to cache a dataset that is not the audited aligned development set")
    if splits["test"]:
        raise ValueError("The untouched test split must remain empty")
    selected_ids = list(splits["train"])
    if split in ("val", "all"):
        selected_ids += list(splits["val"])
    selected_ids = list(dict.fromkeys(selected_ids))
    entries = {e["id"]: e for e in manifest["episodes"]}
    renders_root = Path("/data/libero/milk_sam_native_v1")
    render_report = read_json(renders_root / "provenance.json")
    if render_report.get("test_episodes_rendered") is not False:
        raise ValueError("Rendered provenance must declare that no test episode was rendered")
    renders = {record["episode"]: record for record in render_report["entries"]}
    for episode in selected_ids:
        if episode not in renders:
            raise FileNotFoundError(f"No rendered RGB provenance for {episode}")

    out = Path("/data/libero") / name
    identity = {"schema_version": 1, "dataset": dataset, "dataset_fingerprint": ALIGNED_FINGERPRINT,
                "split": split, "episode_ids": selected_ids, "muse_revision": MUSE_REVISION,
                "layer_index": 26, "chat_current_date": CHAT_DATE,
                "dtype": "BF16 bit pattern stored as uint16", "fps": 20, "test_used": False,
                "quantization": "nf4 compute/storage bf16, double quant, vision excluded",
                "feature_source": "FrozenMuseTokens.encode; no fallback and no generated text"}
    manifest_path = out / "manifest.json"
    prior = read_json(manifest_path) if manifest_path.exists() else None
    if prior is not None:
        for key, value in identity.items():
            if prior.get(key) != value:
                raise ValueError(f"Existing cache has different {key}; use a new cache name")
        if prior.get("complete") and all((out / episode / "manifest.json").exists() for episode in selected_ids):
            return prior
    out.mkdir(parents=True, exist_ok=True)
    if prior is None:
        manifest_path.write_text(json.dumps(dict(identity, complete=False, complete_episodes=[],
            episode_manifests={}, missing_episodes=selected_ids, frames=0, bytes=0), indent=2) + "\n")
        volume.commit()

    reference_path = Path("/data") / REFERENCE_REPORT.lstrip("/")
    reference = read_json(reference_path) if reference_path.exists() else None
    if reference is None and "episode_0000" in selected_ids:
        raise FileNotFoundError("Missing aligned fit report; cannot verify NF4 parity")
    reference_hashes = {}
    if reference is not None:
        if reference.get("dataset_fingerprint") != ALIGNED_FINGERPRINT or reference.get("episode") != "episode_0000":
            raise ValueError("Reference report is not for the aligned episode_0000 fit")
        if reference.get("muse_revision") != MUSE_REVISION:
            raise ValueError("Reference report uses a different Muse revision")
        reference_hashes = {int(row["frame"]): row["feature_sha256"] for row in reference["frames"]}
        reference_rgb = {int(row["frame"]): row["rgb_sha256"] for row in reference["frames"]}
        if not reference_hashes:
            raise ValueError("Reference report has no frame hashes")

    model_root = Path("/data/checkpoints/muse-glimmer-30b") / MUSE_REVISION
    if not (model_root / "config.json").exists():
        raise FileNotFoundError("Missing existing Muse checkpoint; no implicit download in GPU job")
    muse = None
    try:
        muse = FrozenMuseTokens(model_root, layer_index=26, quantization="nf4")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        quantized = [n for n, m in muse.model.named_modules() if type(m).__name__ == "Linear4bit"]
        if not quantized:
            raise ValueError("No Linear4bit modules found")
        if any("vision" in name.lower() for name in quantized):
            raise ValueError("Vision modules were quantized unexpectedly")
        outside = [name for name in quantized if not name.startswith(muse.module_name + ".")]
        if outside:
            raise ValueError(f"NF4 modules outside the text decoder: {outside[:8]}")
        if torch.is_autocast_enabled("cuda"):
            raise ValueError("Muse cache must be built without CUDA autocast")
        backend = backend_snapshot(torch, muse.model)
        if backend["matmul_allow_tf32"] or backend["cudnn_allow_tf32"]:
            raise ValueError("TF32 must be disabled while extracting reference features")
        completed, skipped, missing = [], [], []
        frames_written = bytes_written = 0
        reference_verified = 0
        for episode in selected_ids:
            entry = entries[episode]
            episode_dir = out / episode
            episode_manifest = episode_dir / "manifest.json"
            timestamps = load_episode(root, entry, manifest, include_images=False)["timestamps"]
            image_paths = sorted((renders_root / episode).glob("*.jpg"))
            expected = renders[episode]
            if len(image_paths) != expected["frames"] or len(timestamps) != expected["frames"]:
                raise ValueError(f"{episode}: render/timestamp frame count mismatch")
            digest = hashlib.sha256()
            for path in image_paths:
                digest.update(path.read_bytes())
            if digest.hexdigest() != expected["jpeg_sequence_sha256"]:
                raise ValueError(f"{episode}: rendered JPEG sequence does not match provenance")
            if episode_manifest.exists():
                record = read_json(episode_manifest)
                if record["episode"] != episode or record["frames"] != len(image_paths) or record[
                        "instruction_sha256"] != hashlib.sha256(entry["instruction"].encode()).hexdigest():
                    raise ValueError(f"{episode}: existing cache metadata mismatch")
                if record.get("processor_config_sha256") != muse.processor_hash:
                    raise ValueError(f"{episode}: cached tokens used a different processor")
                if (record.get("dataset_fingerprint") != ALIGNED_FINGERPRINT
                        or record.get("muse_revision") != MUSE_REVISION
                        or record.get("layer_index") != 26
                        or record.get("chat_current_date") != CHAT_DATE):
                    raise ValueError(f"{episode}: existing cache identity mismatch")
                for item in record["records"]:
                    path = episode_dir / f'{item["frame"]:06}.npz'
                    if file_hash(path) != item["sha256"]:
                        raise ValueError(f"{episode}: cached token changed")
                    if episode == "episode_0000" and reference_hashes:
                        if item["frame"] not in reference_hashes:
                            raise ValueError("Reference report does not cover every episode_0000 frame")
                        if item.get("feature_sha256") != reference_hashes[item["frame"]]:
                            raise ValueError(f"episode_0000 frame {item['frame']}: cached feature differs from fit")
                        if item.get("image_sha256") != reference_rgb[item["frame"]]:
                            raise ValueError(f"episode_0000 frame {item['frame']}: RGB differs from fit")
                        reference_verified += 1
                completed.append(episode)
                skipped.append(episode)
                frames_written += len(record["records"])
                bytes_written += sum((episode_dir / f'{item["frame"]:06}.npz').stat().st_size
                                     for item in record["records"])
                continue
            if time.monotonic() - started >= max_runtime_seconds:
                missing = selected_ids[selected_ids.index(episode):]
                break
            episode_dir.mkdir(parents=True, exist_ok=True)
            for stale in episode_dir.glob("*.npz"):
                stale.unlink()
            records = []
            for frame, image_path in enumerate(image_paths):
                with Image.open(image_path) as rgb:
                    tokens = muse.encode(rgb, entry["instruction"], current_date=CHAT_DATE)
                feature_hash = hashlib.sha256(tokens["features_bfloat16_bits"].tobytes()).hexdigest()
                image_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
                if episode == "episode_0000" and reference_hashes:
                    if frame not in reference_hashes:
                        raise ValueError("Reference report does not cover every episode_0000 frame")
                    if image_hash != reference_rgb[frame]:
                        raise ValueError(f"episode_0000 frame {frame}: RGB parity gate failed")
                    if feature_hash != reference_hashes[frame]:
                        raise ValueError(f"episode_0000 frame {frame}: NF4 feature parity gate failed")
                tmp = episode_dir / f"{frame:06}.tmp.npz"
                final = episode_dir / f"{frame:06}.npz"
                np.savez(tmp, **tokens)
                os.replace(tmp, final)
                records.append({"frame": frame, "timestamp": float(timestamps[frame]),
                                "image_sha256": image_hash,
                                "feature_sha256": feature_hash, "tokens": int(tokens["attention_mask"].sum()),
                                "bytes": final.stat().st_size, "sha256": file_hash(final)})
                frames_written += 1
                bytes_written += final.stat().st_size
                if episode == "episode_0000" and reference_hashes:
                    reference_verified += 1
            record = {"episode": episode, "frames": len(records),
                      "dataset_fingerprint": ALIGNED_FINGERPRINT,
                      "instruction_sha256": hashlib.sha256(entry["instruction"].encode()).hexdigest(),
                      "processor_config_sha256": muse.processor_hash, "muse_revision": MUSE_REVISION,
                      "layer_index": 26, "chat_current_date": CHAT_DATE, "records": records}
            episode_manifest.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
            completed.append(episode)
            (out / "progress.json").write_text(json.dumps({"completed": completed, "selected": selected_ids},
                                                          indent=2) + "\n")
            volume.commit()
            print(json.dumps({"episode": episode, "completed": len(completed), "frames": frames_written,
                              "seconds": time.monotonic() - started}), flush=True)
        complete = len(completed) == len(selected_ids)
        episode_manifests = {e: file_hash(out / e / "manifest.json") for e in completed}
        result = dict(identity, complete=complete, complete_episodes=completed,
                      episode_manifests=episode_manifests,
                      missing_episodes=[e for e in selected_ids if e not in completed],
                      frames=frames_written, bytes=bytes_written, processor_config_sha256=muse.processor_hash,
                      processor_backend="Transformers 5.21.0; chat date pinned",
                      transformers=transformers.__version__, torch=str(torch.__version__),
                      cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                      backend_before_extraction=backend, quantized_module_count=len(quantized),
                      quantized_module_scope=muse.module_name,
                      reference_report=REFERENCE_REPORT if reference else None,
                      reference_report_sha256=file_hash(reference_path) if reference else None,
                      reference_frames_verified=reference_verified,
                      elapsed_seconds=time.monotonic() - started,
                      peak_gpu_gib=torch.cuda.max_memory_allocated() / 1024 ** 3,
                      packages={k: importlib.metadata.version(k)
                                for k in ("torch", "transformers", "bitsandbytes", "accelerate")})
        manifest_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        volume.commit()
        return result
    except Exception as exc:
        (out / "failure.json").write_text(json.dumps({"identity": identity, "error_type": type(exc).__name__,
                                                      "error": str(exc), "elapsed_seconds": time.monotonic() - started},
                                                     indent=2) + "\n")
        volume.commit()
        raise
    finally:
        if muse is not None:
            muse.close()


@app.local_entrypoint()
def main(name: str = "muse_nf4_aligned_train_v1", dataset: str = "muse_aligned_development_v1",
         split: str = "train", max_runtime_seconds: int = 2400):
    print(build.remote(name, dataset, split, max_runtime_seconds))
