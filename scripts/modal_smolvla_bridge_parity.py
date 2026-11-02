"""Check the native expert wrapper against official full-prefix computation."""
import modal
from scripts.modal_smolvla_reference import image as base_image

image = (base_image.add_local_file("scripts/__init__.py", "/root/scripts/__init__.py")
         .add_local_file("scripts/modal_smolvla_reference.py", "/root/scripts/modal_smolvla_reference.py"))

app = modal.App("odvla-smolvla-bridge-parity")
volume = modal.Volume.from_name("object-dynamics-vla-data", create_if_missing=False)


@app.function(image=image, volumes={"/data": volume}, cpu=4, memory=16384, timeout=600,
              max_containers=1, single_use_containers=True)
def run():
    import torch
    from pathlib import Path
    from odvla.data import file_hash
    from odvla.vla.smolvla_bridge import expert_velocity
    from lerobot.configs import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, make_att_2d_masks

    torch.set_num_threads(4)
    source = Path("/data/checkpoints/smolvla-base/5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6")
    if file_hash(source / "model.safetensors") != "7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb":
        raise ValueError("Checkpoint hash mismatch")
    cfg = PreTrainedConfig.from_pretrained(str(source))
    cfg.device = "cpu"; cfg.vlm_model_name = "/data/checkpoints/smolvlm2-reference/7b375e1b73b11138ff12fe22c8f2822d8fe03467"
    cfg.load_vlm_weights = False
    torch.manual_seed(7)
    policy = SmolVLAPolicy.from_pretrained(str(source), config=cfg, strict=True).eval()
    model = policy.model
    model.requires_grad_(False)
    prefix = torch.randn(1, 7, 960)
    valid = torch.tensor([[True, True, True, True, True, False, False]])
    prefix_ar = torch.tensor([[False, False, False, False, True, False, False]])
    prefix_mask = make_att_2d_masks(valid, prefix_ar)
    prefix_positions = torch.cumsum(valid, dim=1) - 1
    with torch.no_grad():
        _, cache = model.vlm_with_expert.forward(attention_mask=prefix_mask, position_ids=prefix_positions,
            inputs_embeds=[prefix, None], past_key_values=None, use_cache=True)
    keys = [layer.keys.detach().clone().requires_grad_(True) for layer in cache.layers]
    values = [layer.values.detach().clone().requires_grad_(True) for layer in cache.layers]
    noisy = torch.randn(1, 50, 32); t = torch.tensor([.5])
    suffix, suffix_valid, suffix_ar = model.embed_suffix(noisy, t)
    all_valid = torch.cat((valid, suffix_valid), dim=1)
    all_ar = torch.cat((prefix_ar, suffix_ar.bool()), dim=1)
    with torch.no_grad():
        (_, official), _ = model.vlm_with_expert.forward(
            attention_mask=make_att_2d_masks(all_valid, all_ar), position_ids=torch.cumsum(all_valid, dim=1) - 1,
            inputs_embeds=[prefix, suffix], past_key_values=None, use_cache=False)
        expected = model.action_out_proj(official[:, -50:].float())
    actual = expert_velocity(model, keys, values, valid, noisy, t)
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
    actual.square().mean().backward()
    norms = [float(x.grad.norm()) if x.grad is not None else 0.0 for x in keys + values]
    if not all(x > 0 for x in norms): raise AssertionError("Missing K/V gradients")
    return {"max_abs_error": float((actual - expected).abs().max()), "layer_count": len(keys),
            "kv_shape": list(keys[0].shape), "minimum_kv_gradient_norm": min(norms),
            "strict_released_checkpoint": True, "prefix": "synthetic embeddings processed by official prefix prefill"}


@app.local_entrypoint()
def main():
    print(run.remote())
