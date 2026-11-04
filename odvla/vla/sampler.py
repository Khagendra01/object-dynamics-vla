"""Deterministic, resumable shuffled minibatch sampling for cached demonstrations."""
import numpy as np


class ShuffledBatchSampler:
    """One permutation per epoch, with exact checkpoint/resume state."""

    def __init__(self, size, batch_size, seed=0):
        self.size, self.batch_size, self.seed = int(size), int(batch_size), int(seed)
        if self.size < 1 or self.batch_size < 1:
            raise ValueError("size and batch_size must be positive")
        self.epoch, self.cursor = 0, 0
        self.rng = np.random.default_rng(self.seed)
        self.permutation = self.rng.permutation(self.size)

    @classmethod
    def from_state(cls, state):
        required = {"size", "batch_size", "seed", "epoch", "cursor", "permutation", "rng_state"}
        if set(state) != required:
            raise ValueError(f"Sampler state keys must be {sorted(required)}")
        sampler = cls.__new__(cls)
        sampler.size, sampler.batch_size, sampler.seed = int(state["size"]), int(state["batch_size"]), int(state["seed"])
        sampler.epoch, sampler.cursor = int(state["epoch"]), int(state["cursor"])
        if sampler.size < 1 or sampler.batch_size < 1 or sampler.epoch < 0 or not 0 <= sampler.cursor <= sampler.size:
            raise ValueError("Invalid sampler state values")
        sampler.permutation = np.asarray(state["permutation"], dtype=np.int64)
        if sampler.permutation.shape != (sampler.size,) or not np.array_equal(
                np.sort(sampler.permutation), np.arange(sampler.size)):
            raise ValueError("Sampler permutation is not a complete permutation")
        sampler.rng = np.random.default_rng()
        sampler.rng.bit_generator.state = state["rng_state"]
        return sampler

    def state_dict(self):
        return {"size": int(self.size), "batch_size": int(self.batch_size), "seed": int(self.seed),
                "epoch": int(self.epoch), "cursor": int(self.cursor),
                "permutation": [int(x) for x in self.permutation],
                "rng_state": self.rng.bit_generator.state}

    def next_batch(self):
        if self.cursor == self.size:
            self.epoch += 1
            self.permutation = self.rng.permutation(self.size)
            self.cursor = 0
        end = min(self.cursor + self.batch_size, self.size)
        batch = self.permutation[self.cursor:end].copy()
        self.cursor = end
        return batch
