"""Task-then-episode-then-window sampling with exact resumable RNG state."""
import copy
import numpy as np


class TaskBalancedSampler:
    """Sample tasks uniformly, then episodes uniformly, then windows uniformly."""

    def __init__(self, task_episodes, seed=7):
        if not isinstance(task_episodes, dict) or not task_episodes:
            raise ValueError("Expected nonempty task-to-episode mapping")
        self.tasks = sorted(task_episodes)
        self.episodes = {}
        for task in self.tasks:
            episodes = task_episodes[task]
            if not isinstance(episodes, dict) or not episodes:
                raise ValueError(f"Task {task} has no training episodes")
            self.episodes[task] = {episode: int(n) for episode, n in sorted(episodes.items())}
            if any(n < 1 for n in self.episodes[task].values()):
                raise ValueError(f"Task {task} has an empty episode")
        self.rng = np.random.default_rng(seed)
        self.examples_per_task = {task: 0 for task in self.tasks}

    def take(self, size):
        if size < 1:
            raise ValueError("Batch size must be positive")
        result = []
        for _ in range(size):
            task = self.tasks[int(self.rng.integers(len(self.tasks)))]
            episodes = list(self.episodes[task])
            episode = episodes[int(self.rng.integers(len(episodes)))]
            frame = int(self.rng.integers(self.episodes[task][episode]))
            result.append((task, episode, frame))
            self.examples_per_task[task] += 1
        return result

    def state_dict(self):
        return {"episodes": copy.deepcopy(self.episodes), "rng": copy.deepcopy(self.rng.bit_generator.state),
                "examples_per_task": dict(self.examples_per_task)}

    def load_state_dict(self, state):
        if state.get("episodes") != self.episodes or set(state.get("examples_per_task", {})) != set(self.tasks):
            raise ValueError("Sampler state does not match training tasks and episodes")
        counts = state["examples_per_task"]
        if any(not isinstance(counts[t], int) or counts[t] < 0 for t in self.tasks):
            raise ValueError("Invalid per-task counts")
        self.rng.bit_generator.state = copy.deepcopy(state["rng"])
        self.examples_per_task = dict(counts)
