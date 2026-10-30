# Prefix-only SAM closed-loop development pilot

This is a bounded simulator integration pilot, not a success-rate comparison or
real-time deployment. The existing frozen milk checkpoints and perception
configuration in `MILK_SAM_DEVELOPMENT.md` were not retrained or reweighted.

## Prefix replay benchmark

`odvla.sam_prefix.replay_current` starts a new tracker session containing exactly
the contiguous JPEG frames observed so far, with the frozen frame-zero points.
The same guard, pointer scaling, merge fix and resize as the cache are used.
It rejects extra images in the prefix directory and returns empty masks and ID
status explicitly; it never substitutes simulator masks.

On reused validation recording `episode_0014`, sampled prefix lengths 1, 2, 6,
11, 16, 17, 21, 31 and 51 produced **pixel-identical** 128×128 milk and basket
masks to the frozen cache at their final frames, all with unique IDs and nonempty
masks. This is a sampled check on one recording, not a universal equivalence
claim. On Modal A100-80GB, session load/prompt/replay/close took 0.48 s at
length 2, 2.84 s at length 17, 5.24 s at length 31 and 8.50 s at length 51.
The first single-frame session took 1.89 s including warm-up. Timing excludes
simulator rendering and policy inference; repeated full-prefix replay scales
roughly quadratically with trajectory length. The simulator is paused while
tracking and uses a 20 Hz simulated control interval.

## First action, offline validation

`scripts/modal_first_action_eval.py` uses the exact frozen evaluation's paired
seed-921 noise schedule, reports only chunk index zero, and evaluates 1,155
windows per checkpoint from reused validation demonstrations. The table shows
first-action **normalized MSE** for action-only / joint by channel; lower is
better. Full per-channel source-unit MAEs and metadata remain at
`/data/eval/milk_sam_first_action_v1.json` on Modal.

| Channel | Seed 7 | Seed 19 | Seed 31 |
|---|---|---|---|
| translation x | .71585 / .71794 | .70510 / .70321 | .71747 / .70669 |
| translation y | .14810 / .14531 | .15217 / .14980 | .15371 / .15328 |
| translation z | .18935 / .18874 | .16248 / .16237 | .17626 / .17354 |
| rotation x | .72285 / .70684 | .54460 / .52666 | .83329 / .83013 |
| rotation y | 1.28144 / 1.29387 | 1.19027 / 1.20174 | 1.35081 / 1.35639 |
| rotation z | 1.69206 / 1.68882 | 1.37249 / 1.35324 | 1.76429 / 1.75891 |
| gripper | .25813 / .25183 | .24719 / .24756 | .22952 / .22905 |
| mean over channels | .71540 / .71334 | .62490 / .62065 | .74648 / .74400 |

These are reused validation windows, not closed-loop outcomes. The changes are
mixed by channel even where the aggregate favors joint.

## Paired 30-step loop

`scripts/modal_sam_closed_loop_pilot.py` ran action-only and joint **seed-7**
frozen checkpoints from the same already-used validation initial state
`episode_0014` / `demo_21`, with matching initial RGB hashes and sampling
seed `7 + step`. Both used the same frozen two RGB points. Each simulator step
re-rendered RGB, replayed only frames already observed, constructed observation
history, sampled a fresh action chunk and executed its first action clipped to
`[-1,1]`. Initial observation history was repeated four times, a known training
distribution mismatch. SAM outputs were the only object measurements; no oracle
masks were read or used. The one-step monitor was disabled and passive.

Both policies executed all **30/30 steps**, with no missing, duplicated or empty
object IDs at the current frame. Neither achieved task success within the
30-step cap. This short pilot does not establish whether either policy would
succeed with a normal task horizon or whether one controls better. The first
action-only mask pair was pixel-identical to the cached initial frame even in
the combined simulator/GPU environment. Three overlay frames per policy were
visually inspected: both tracked objects remained in place while the arm moved;
no grasp or placement occurred in those sampled views.

| Seed 7 variant | Clipped steps | Clipped channel counts (x, y, z, rx, ry, rz, gripper) | Total tracking time |
|---|---:|---|---:|
| action-only | 14/30 | 0, 2, 0, 0, 0, 0, 14 | 86.60 s |
| joint | 13/30 | 0, 3, 0, 0, 0, 0, 13 | 86.14 s |

The longest 30-frame replay took approximately 5.21 s for each variant. Per-step
IDs, nonempty status, sampled/executed actions, clipping, timing and simulator
success flags, plus masks and three-frame montages, are at
`/data/eval/milk_sam_closed_loop_pilot_v2/`. This is one reused initialization,
one seed pair and a deliberately truncated horizon. It is development evidence;
fresh initializations remain reserved for a later frozen evaluation.

An earlier pilot attempt failed before executing an action because SAM left CUDA
autocast active and the policy's bfloat16 dynamics tensor could not be converted
to NumPy. Policy inference now explicitly disables autocast and converts outputs
to float32. The completed pilot used that corrected inference path.
