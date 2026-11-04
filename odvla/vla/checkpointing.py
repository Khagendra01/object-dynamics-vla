"""Complete resumable checkpoint payloads for cached-feature training runs."""
SCHEMA_VERSION = 1

REQUIRED_KEYS = ("schema_version", "completed_updates", "prefix", "trainable_policy", "optimizer",
                 "torch_rng", "cuda_rng", "sampler_state", "normalize_queries", "dynamics_objective",
                 "dataset", "dataset_fingerprint", "cache", "cache_manifest_sha256", "stats_sha256",
                 "batch_size", "lr", "seed")


def checkpoint_payload(*, completed_updates, prefix, trainable_policy, optimizer, torch_rng, cuda_rng,
                       sampler_state, provenance):
    """Assemble a checkpoint that can resume model, optimizer, RNG and sample order."""
    if int(completed_updates) < 1:
        raise ValueError("A resumable checkpoint requires at least one completed update")
    if not isinstance(sampler_state, dict) or not sampler_state:
        raise ValueError("A complete resumable checkpoint requires sampler state")
    missing = {"normalize_queries", "dynamics_objective", "dataset", "dataset_fingerprint",
               "cache", "cache_manifest_sha256", "stats_sha256", "batch_size", "lr", "seed"} - set(provenance)
    if missing:
        raise ValueError(f"Checkpoint provenance is missing {sorted(missing)}")
    payload = {"schema_version": SCHEMA_VERSION, "completed_updates": int(completed_updates),
               "prefix": prefix, "trainable_policy": trainable_policy, "optimizer": optimizer,
               "torch_rng": torch_rng, "cuda_rng": cuda_rng, "sampler_state": sampler_state}
    overlap = set(provenance).intersection(payload)
    if overlap:
        raise ValueError(f"Checkpoint provenance cannot replace {sorted(overlap)}")
    payload.update(provenance)
    return payload


def validate_checkpoint(checkpoint, *, require_sampler=True, expected=None):
    """Check required fields and, optionally, immutable run identity."""
    missing = [key for key in REQUIRED_KEYS if key not in checkpoint]
    if missing:
        raise ValueError(f"Checkpoint is missing {missing}")
    if checkpoint["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported checkpoint schema")
    if require_sampler and (not checkpoint["sampler_state"] or "permutation" not in checkpoint["sampler_state"]):
        raise ValueError("Checkpoint has no resumable sampler state")
    if checkpoint["normalize_queries"] is not True:
        raise ValueError("Expected query-normalized conditioning")
    if checkpoint["dynamics_objective"] is not False:
        raise ValueError("Expected no dynamics objective")
    for key, value in (expected or {}).items():
        if checkpoint.get(key) != value:
            raise ValueError(f"Checkpoint {key} mismatch: {checkpoint.get(key)!r} != {value!r}")
    return checkpoint
