"""Pinned ten-task LIBERO-Object integration registry.

Tasks differ by target object, not just initial placement. Tracking prompts are
deliberately left unassigned until inspected on actual RGB observations.
"""

_SOURCES = (
    ("alphabet_soup", "alphabet soup", "42189d4415d4c51aaaf0708300653fccc39239cd3f2709079a713cd8d1678a8d"),
    ("bbq_sauce", "bbq sauce", "90a722f2243a72878baff494d07ce7f4206cfc10307a4c82dcda5f2a5fabc060"),
    ("butter", "butter", "07f48f1c0c06b5e10bc219e63701f0240d8205f85ccd0daab3ff9bd27bc35fb9"),
    ("chocolate_pudding", "chocolate pudding", "b5b57f36cd78e598e258e0e2a1a9d4e8d37d387901487eecd22d503e0e222302"),
    ("cream_cheese", "cream cheese", "7ae50ed3a64bab8418fd6c8e346ba1c39da0fdaa62916a87b9d20f3745b45406"),
    ("ketchup", "ketchup", "e0814effbe3a46777952950be0565577c348dad211327bc5d2a16f4eacc55ba1"),
    ("milk", "milk", "e56d954928e613f9db58085b7677ccf9c4824f1ab982265f71405d5bb14cfb4a"),
    ("orange_juice", "orange juice", "53e40490c62e94c678233eefbfd874a61c93d92c8088f59d426ae6c3fd90d288"),
    ("salad_dressing", "salad dressing", "45960606a85b9c68980eb8130e0fe16bf3621539cf5d4147f35aa8139573c5ac"),
    ("tomato_sauce", "tomato sauce", "43c52bdaa78b4c6aff3afb54d932d6a0e065f5c30c8e30a6c17f78749d487ea9"),
)

LIBERO_REVISION = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
DATASET_BASE = "https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/resolve/main/libero_object"
CONTROLLER = {
    "embodiment": "LIBERO Panda single arm",
    "name": "robosuite 1.4 OSC_POSE",
    "action_channels": ["translation_x", "translation_y", "translation_z", "rotation_x", "rotation_y", "rotation_z", "gripper"],
    "units": "normalized [-1,1], OSC translation scale 0.05, rotation scale 0.5",
    "frame": "robosuite 1.4 default OSC_POSE controller configuration",
    "gripper": "PandaGripper: negative open, positive close",
    "control_hz": 20,
    "robot_observation": "joint position 7, end-effector position and axis angle 6, gripper qpos 2",
    "timing": "RGB and robot regenerated from simulator states[t]; action[t] follows state[t]",
    "camera": "agentview; native 512x512 JPEG quality 98, unflipped MuJoCo render",
}


def registry():
    tasks = []
    for slug, label, digest in _SOURCES:
        task_name = f"pick_up_the_{slug}_and_place_it_in_the_basket"
        tasks.append({
            "task_id": f"libero_object_{slug}",
            "suite": "libero_object",
            "instruction": f"pick up the {label} and place it in the basket",
            "bddl": f"libero_object/{task_name}.bddl",
            "source_url": f"{DATASET_BASE}/{task_name}_demo.hdf5",
            "source_sha256": digest,
            "source_path": f"libero/ten_task_sources/{slug}.hdf5",
            "object_slots": [{"role": "target", "identity": label}, {"role": "destination", "identity": "basket"}],
            "sam_initialization": {"method": "unassigned; requires first-frame RGB inspection", "points": None},
            "controller": dict(CONTROLLER),
            "split_policy": "episode-level train/dev/eval; milk retains previously used development split",
        })
    return {"schema_version": 1, "libero_revision": LIBERO_REVISION, "tasks": tasks}
