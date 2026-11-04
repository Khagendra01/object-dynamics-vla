"""Single-worker, byte-bounded LRU storage for frozen Muse token arrays."""
import hashlib
import io
import json
import os
import re
import time
from pathlib import Path

import numpy as np


def frozen_feature_key(*, task_id, episode_id, frame, instruction, image_sha256,
                       muse_revision, processor_sha256, backend, chat_date):
    """Identity includes image, instruction, frozen model, processor and backend."""
    fields = dict(task_id=task_id, episode_id=episode_id, frame=int(frame), instruction=instruction,
                  image_sha256=image_sha256, muse_revision=muse_revision, processor_sha256=processor_sha256,
                  backend=backend, chat_date=chat_date)
    if frame < 0 or any(not isinstance(fields[k], str) or not fields[k] for k in (
            "task_id", "episode_id", "instruction", "image_sha256", "muse_revision", "processor_sha256", "chat_date")):
        raise ValueError("Incomplete frozen feature identity")
    if not isinstance(backend, dict) or not backend:
        raise ValueError("Backend identity is required")
    return hashlib.sha256(json.dumps(fields, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class BoundedFeatureCache:
    """Persist most recently accessed NPZ entries under a strict byte budget.

    One process owns a cache directory. On misses the caller recomputes frozen
    Muse features; the cache never freezes or modifies the trainable adapter.
    """

    def __init__(self, root, max_bytes):
        self.root, self.max_bytes = Path(root), int(max_bytes)
        if self.max_bytes < 1:
            raise ValueError("Cache budget must be positive")
        self.root.mkdir(parents=True, exist_ok=True)
        for path in self.root.glob(".*.tmp"):
            if re.fullmatch(r"\.[0-9a-f]{64}\.\d+\.\d+\.tmp", path.name):
                path.unlink()
        self._evict_to_fit(0)

    def _path(self, key):
        if not isinstance(key, str) or len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("Cache key must be a lowercase SHA-256")
        return self.root / f"{key}.npz"

    def _entries(self):
        return sorted((p for p in self.root.glob("*.npz") if p.is_file()),
                      key=lambda p: (p.stat().st_mtime_ns, p.name))

    @property
    def size_bytes(self):
        return sum(p.stat().st_size for p in self._entries())

    def _evict_to_fit(self, needed, exclude=None):
        entries = self._entries()
        total = sum(p.stat().st_size for p in entries)
        for path in entries:
            if total + needed <= self.max_bytes:
                break
            if path == exclude:
                continue
            size = path.stat().st_size
            path.unlink()
            total -= size
        return total + needed <= self.max_bytes

    def get(self, key):
        path = self._path(key)
        if not path.is_file():
            return None
        with np.load(path, allow_pickle=False) as arrays:
            result = {name: arrays[name].copy() for name in arrays.files}
        os.utime(path, None)
        return result

    def put(self, key, arrays):
        path = self._path(key)
        if not isinstance(arrays, dict) or not arrays:
            raise ValueError("Expected nonempty feature arrays")
        if any(np.asarray(value).dtype.hasobject for value in arrays.values()):
            raise ValueError("Object arrays are not supported")
        buffer = io.BytesIO()
        np.savez(buffer, **arrays)
        payload = buffer.getvalue()
        if len(payload) > self.max_bytes:
            return False
        # Replacing an existing entry first removes its accounted bytes. The
        # temporary file is within the same byte budget and renamed atomically.
        if path.exists():
            path.unlink()
        if not self._evict_to_fit(len(payload)):
            return False
        temporary = self.root / f".{key}.{os.getpid()}.{time.time_ns()}.tmp"
        try:
            temporary.write_bytes(payload)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return True
