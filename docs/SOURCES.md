# Official references checked during authoring

Checked 2026-10-28. Upstream APIs, requirements, prices and checkpoints can change.
Pin actual revisions in your environment; these links are references, not a lockfile.

- SAM official code, setup and checkpoints:
  https://github.com/facebookresearch/sam3
- SAM 3.1 release notes:
  https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md
- Actual SAM 3.1 multiplex notebook:
  https://github.com/facebookresearch/sam3/blob/main/examples/sam3.1_video_predictor_example.ipynb
- SAM model builders and session API:
  https://github.com/facebookresearch/sam3/blob/main/sam3/model_builder.py
  https://github.com/facebookresearch/sam3/blob/main/sam3/model/sam3_base_predictor.py
- Muse model card and config:
  https://huggingface.co/meta-models/Muse-Glimmer-30B
  https://huggingface.co/meta-models/Muse-Glimmer-30B/blob/main/config.json
- Muse quantized inference artifacts (different from the BF16 cache script):
  https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF
- LIBERO setup, demonstrations and evaluation:
  https://github.com/Lifelong-Robot-Learning/LIBERO
- Isaac Lab imitation learning:
  https://isaac-sim.github.io/IsaacLab/main/source/features/imitation-learning/index.html
- Isaac camera observations:
  https://docs.isaacsim.omniverse.nvidia.com/latest/sensors/isaacsim_sensors_camera.html
- Isaac requirements:
  https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html
- Flow-based VLA precedent and implementations:
  https://www.physicalintelligence.company/download/pi0.pdf
  https://github.com/Physical-Intelligence/openpi
- CUDA PyTorch install selector:
  https://pytorch.org/get-started/locally/
- Modal pricing:
  https://modal.com/pricing

The reference action head uses the mathematical flow-matching interpolation, not
an imported pretrained pi0 policy or an assertion that swapping in Muse reproduces pi0.
