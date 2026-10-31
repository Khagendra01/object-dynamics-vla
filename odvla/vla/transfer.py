"""Conservative explicit cross-dataset motor-transfer audit.

Contracts are human-audited declarations, not automatic proof of controller
semantics or of the absence of semantically duplicated demonstrations.
"""

SEMANTIC_FIELDS = ("embodiment", "controller", "action_channels", "action_frame",
                   "action_units", "gripper_convention", "robot_channels",
                   "camera", "rgb_orientation", "control_hz", "metrics",
                   "metric_units", "slot_roles", "muse_revision", "muse_layer",
                   "muse_processor_sha256")


def validate_contract(contract, splits):
    if contract.get("schema_version") != 1 or not contract.get("audit_evidence"):
        raise ValueError("Transfer requires a versioned audit and evidence")
    if contract.get("dataset_fingerprint") != splits["fingerprint"]:
        raise ValueError("Audit fingerprint mismatch")
    semantics = contract.get("semantics", {})
    if any(key not in semantics or semantics[key] in (None, "", []) for key in SEMANTIC_FIELDS):
        raise ValueError("Incomplete controller, feature, or Muse compatibility audit")
    identities = contract.get("episode_sources", {})
    ids = set(splits["train"]) | set(splits["val"]) | set(splits["test"])
    if set(identities) != ids or any(not isinstance(v, str) or not v for v in identities.values()):
        raise ValueError("Every split episode needs a stable source identity")
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if {identities[i] for i in splits[left]} & {identities[i] for i in splits[right]}:
            raise ValueError("Source demonstrations overlap across splits")
    return contract


def audit_transfer(source, destination):
    for key in SEMANTIC_FIELDS:
        if source["semantics"][key] != destination["semantics"][key]:
            raise ValueError(f"Incompatible transfer semantics: {key}")
    overlap = set(source["training_sources"]) & set(destination["protected_sources"])
    if overlap:
        raise ValueError("Motor training data overlaps destination validation/test sources")
    return {"normalization": "reuse motor training statistics unchanged",
            "source_training_count": len(set(source["training_sources"])),
            "destination_protected_count": len(set(destination["protected_sources"])),
            "overlap_count": 0, "semantics": source["semantics"]}


def summarize_contract(contract, splits):
    validate_contract(contract, splits)
    sources = contract["episode_sources"]
    return {"semantics": contract["semantics"], "audit_evidence": contract["audit_evidence"],
            "training_sources": sorted({sources[i] for i in splits["train"]}),
            "protected_sources": sorted({sources[i] for i in splits["val"] + splits["test"]})}


def validate_artifacts(contract, manifest, cache):
    semantics = contract["semantics"]
    checks = {"metrics": manifest["metric_names"], "metric_units": manifest["metric_units"],
              "muse_revision": cache.get("muse_revision"), "muse_layer": cache.get("layer_index"),
              "muse_processor_sha256": cache.get("processor_config_sha256")}
    for key, value in checks.items():
        if value is None or semantics[key] != value:
            raise ValueError(f"Audit disagrees with dataset/cache artifact: {key}")
