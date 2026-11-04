"""Reproducible shuffled windows across episode boundaries."""
import copy
import numpy as np


def validate_resume_contract(saved, requested):
    """Permit extending the target, while rejecting changed training semantics."""
    if (set(saved) != set(requested) or
            any(saved[k] != requested[k] for k in saved if k != "updates") or
            requested["updates"] < saved["updates"]):
        raise ValueError("Resume contract changed or update target decreased")


class ShuffledWindows:
    def __init__(self, count, seed=7):
        if count < 1:
            raise ValueError("Expected at least one training window")
        self.count = int(count)
        self.rng = np.random.default_rng(seed)
        self.order = self.rng.permutation(self.count)
        self.cursor = 0
        self.epochs = 0

    def take(self, size):
        if size < 1:
            raise ValueError("Batch size must be positive")
        result = []
        while len(result) < size:
            if self.cursor == self.count:
                self.order = self.rng.permutation(self.count)
                self.cursor = 0
                self.epochs += 1
            length = min(size - len(result), self.count - self.cursor)
            result.extend(self.order[self.cursor:self.cursor + length].tolist())
            self.cursor += length
        return result

    def state_dict(self):
        return {"count": self.count, "order": self.order.tolist(), "cursor": self.cursor,
                "epochs": self.epochs, "rng": copy.deepcopy(self.rng.bit_generator.state)}

    def load_state_dict(self, state):
        order = np.asarray(state["order"], dtype=np.int64)
        if (state["count"] != self.count or order.shape != (self.count,) or
                not np.array_equal(np.sort(order), np.arange(self.count)) or
                not 0 <= state["cursor"] <= self.count or state["epochs"] < 0):
            raise ValueError("Invalid shuffled-window state")
        self.order, self.cursor, self.epochs = order.copy(), state["cursor"], state["epochs"]
        self.rng.bit_generator.state = copy.deepcopy(state["rng"])
