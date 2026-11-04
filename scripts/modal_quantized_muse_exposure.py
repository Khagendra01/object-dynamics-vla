"""Continue the query-normalized NF4 Muse/SAM expert; evaluation is separate."""
import modal
from scripts.modal_quantized_muse_preflight import image, volume, MUSE_REVISION, SMOL_CONFIG_REVISION

app = modal.App("odvla-quantized-muse-exposure")


@app.function(image=image, gpu="H200", cpu=4, memory=65536, volumes={"/data": volume},
              timeout=1200, max_containers=1, single_use_containers=True)
def train(name: str = "quantized_muse_exposure_v1", max_train_seconds: int = 900):
    import hashlib
    import json
    import sys
    import time
    from pathlib import Path
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as qwen
    if not hasattr(qwen, "apply_multimodal_rotary_pos_emb"):
        def apply_multimodal_rotary_pos_emb(q,k,cos,sin,mrope_section,unsqueeze_dim=1):
            sections=mrope_section*2
            cos=torch.cat([m[i%3] for i,m in enumerate(cos.split(sections,dim=-1))],dim=-1).unsqueeze(unsqueeze_dim)
            sin=torch.cat([m[i%3] for i,m in enumerate(sin.split(sections,dim=-1))],dim=-1).unsqueeze(unsqueeze_dim)
            return (q*cos)+(qwen.rotate_half(q)*sin), (k*cos)+(qwen.rotate_half(k)*sin)
        qwen.apply_multimodal_rotary_pos_emb=apply_multimodal_rotary_pos_emb
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.constants import ACTION, OBS_STATE
    from odvla.data import file_hash, load_episode, read_json, validate_splits
    from odvla.vla.dataset import observation_window
    from odvla.vla.muse_context import FrozenMuseTokens
    from odvla.vla.smolvla_bridge import MuseSamPrefix, expert_velocity
    sys.path.insert(0, "/root")
    from train_muse_vla import make_batch

    if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
        raise ValueError("Invalid output name")
    if not 60 <= max_train_seconds <= 900: raise ValueError("Training budget must be 60..900 seconds")
    out=Path("/data/eval")/name; out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic(); muse=None
    report={"status":"started", "episode":"episode_0000", "starting_updates":3000, "updates_reached":3000,
        "milestones_requested":[5000,7500,10000], "committed_checkpoints":[], "batch_size":1,
        "optimizer":"AdamW lr=1e-4 betas=(0.9,0.95) default weight_decay=0.01",
        "optimizer_restart_at_3000":True, "optimizer_restart_reason":"source checkpoint did not contain optimizer state",
        "rng_restart_at_3000":True, "rng_restart_seed":8008, "sampling":"noise-at-one flow; same per-frame order and time sampling",
        "normalize_queries":True, "dynamics_objective":False, "test_accessed":False,
        "evaluation":"separate functions after each committed checkpoint", "max_train_seconds":max_train_seconds}
    try:
        fit=Path("/data/eval/quantized_muse_full_prenorm_v1"); prior=read_json(fit/"report.json")
        bank=Path("/data/eval/native_exposure_curve_v1"); bank_meta=read_json(bank/"report.json")
        if prior["status"]!="completed_full_episode_fit" or not prior["normalize_queries"] or len(prior["updates"])!=3000:
            raise ValueError("Expected the 3000-update query-normalized checkpoint")
        if file_hash(bank/"noise.npy")!=bank_meta["noise_sha256"]: raise ValueError("Saved evaluation noise mismatch")
        root=Path("/data/libero/milk_sam_development_v1"); splits=read_json(root/"splits.json"); manifest=validate_splits(root,splits)
        if splits["fingerprint"]!=prior["dataset_fingerprint"] or splits["fingerprint"]!=bank_meta["dataset_fingerprint"]:
            raise ValueError("Dataset mismatch")
        entry=next(e for e in manifest["episodes"] if e["id"]=="episode_0000")
        if entry["id"] not in splits["train"]: raise ValueError("Expected training episode")
        ep=load_episode(root,entry,manifest,include_images=False)
        if len(ep["timestamps"])!=150: raise ValueError("Expected 150 observations")
        with np.load(fit/"stats.npz",allow_pickle=False) as z: stats={k:z[k].copy() for k in z.files}
        with np.load(bank/"stats.npz",allow_pickle=False) as z:
            if any(not np.array_equal(stats[k],z[k]) for k in stats): raise ValueError("Normalization mismatch")
        np.savez(out/"stats.npz",**stats)
        cfg=SmolVLAConfig(device="cuda",vlm_model_name=str(Path("/data/checkpoints/smolvlm2-reference")/SMOL_CONFIG_REVISION),
            load_vlm_weights=False,input_features={OBS_STATE:PolicyFeature(type=FeatureType.STATE,shape=(15,)),
                "observation.images.agentview":PolicyFeature(type=FeatureType.VISUAL,shape=(3,512,512))},
            output_features={ACTION:PolicyFeature(type=FeatureType.ACTION,shape=(7,))})
        torch.manual_seed(7); policy=SmolVLAPolicy(cfg).cuda(); model=policy.model
        model.vlm_with_expert.vlm.eval().requires_grad_(False); tc=model.vlm_with_expert.config.text_config
        source=fit/"preflight_weights.pt"; ck=torch.load(source,map_location="cpu",weights_only=True)
        if ck["completed_updates"]!=3000 or not ck.get("normalize_queries",False): raise ValueError("Checkpoint mismatch")
        prefix=MuseSamPrefix(muse_width=6656,layers=model.vlm_with_expert.num_vlm_layers,heads=tc.num_key_value_heads,
            head_dim=tc.head_dim,normalize_queries=True).cuda()
        prefix.load_state_dict(ck["prefix"],strict=True); params=dict(model.named_parameters())
        if set(ck["trainable_policy"])!={n for n,p in params.items() if p.requires_grad}: raise ValueError("Trainable set mismatch")
        with torch.no_grad():
            for k,v in ck["trainable_policy"].items(): params[k].copy_(v)
        muse=FrozenMuseTokens(Path("/data/checkpoints/muse-glimmer-30b")/MUSE_REVISION,26,quantization="nf4")
        quantized=[n for n,m in muse.model.named_modules() if type(m).__name__=="Linear4bit"]
        if not quantized or any(not n.startswith(muse.module_name+".") for n in quantized): raise ValueError("NF4 module mismatch")
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        samples=[]; feature_started=time.monotonic(); rgb_root=Path("/data/libero/milk_sam_native_v1")/entry["id"]
        for t in range(150):
            path=rgb_root/f"{t:06}.jpg"
            if hashlib.sha256(path.read_bytes()).hexdigest()!=prior["frames"][t]["rgb_sha256"]: raise ValueError(f"RGB mismatch at {t}")
            with Image.open(path) as rgb: tokens=muse.encode(rgb,entry["instruction"],current_date="2026-10-30")
            sample=observation_window(ep,t,stats,history=4,horizon=50)
            sample.update(muse_bits=tokens["features_bfloat16_bits"],muse_valid=tokens["attention_mask"])
            samples.append(make_batch([sample],"cuda"))
            if (t+1)%50==0: print(json.dumps({"features":t+1,"seconds":time.monotonic()-feature_started}),flush=True)
        report.update(dataset_fingerprint=splits["fingerprint"],source_checkpoint_sha256=file_hash(source),
            evaluation_noise_sha256=bank_meta["noise_sha256"],feature_extraction_seconds=time.monotonic()-feature_started,
            muse_revision=MUSE_REVISION,smol_config_revision=SMOL_CONFIG_REVISION,frame_count=len(samples))
        # The source did not save training RNG state. Restart explicitly after
        # constructing/loading all modules and extracting frozen features.
        torch.manual_seed(8008)
        trainable=list(prefix.parameters())+[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(trainable,lr=1e-4,betas=(.9,.95))
        prefix.train(); model.vlm_with_expert.lm_expert.train()
        step=3000; losses=[]; train_start=time.monotonic()
        while step<10000:
            if time.monotonic()-train_start>=max_train_seconds: break
            b=samples[step%150]; optimizer.zero_grad(set_to_none=True)
            target=F.pad(b["actions"],(0,cfg.max_action_dim-7)); noise=model.sample_noise(target.shape,"cuda")
            t=model.sample_time(len(target),"cuda"); xt=t[:,None,None]*noise+(1-t[:,None,None])*target
            with torch.autocast("cuda",dtype=torch.bfloat16):
                state=F.pad(b["robot"][:,-1],(0,cfg.max_state_dim-15))
                keys,values,mask=prefix(b["muse"],b["muse_valid"],model.state_proj(state),b["sam_values"],b["sam_deltas"],
                    b["sam_valid"],b["sam_age"],b["history_valid"],use_sam=True)
                prediction=expert_velocity(model,keys,values,mask,xt,t); valid=b["action_valid"][...,None]
                loss=((prediction-(noise-target)).square()[:,:,:7]*valid).sum()/(valid.sum()*7).clamp_min(1)
            if not torch.isfinite(loss): raise ValueError("Nonfinite flow loss")
            loss.backward(); grad=torch.nn.utils.clip_grad_norm_(trainable,10,error_if_nonfinite=True); optimizer.step()
            step+=1; losses.append(float(loss.detach()))
            if step%500==0:
                print(json.dumps({"step":step,"recent_loss_mean":float(np.mean(losses[-100:])),"training_seconds":time.monotonic()-train_start}),flush=True)
            if step in (5000,7500,10000):
                checkpoint={"completed_updates":step,"prefix":{k:v.detach().cpu() for k,v in prefix.state_dict().items()},
                    "trainable_policy":{n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad},
                    "optimizer":optimizer.state_dict(),"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all(),
                    "frame_schedule_next_index":step%150,"normalize_queries":True,"dynamics_objective":False,
                    "source_checkpoint_sha256":report["source_checkpoint_sha256"],"evaluation_noise_sha256":report["evaluation_noise_sha256"],
                    "optimizer_restart_at_3000":True,"rng_restart_seed":8008,"dataset_fingerprint":splits["fingerprint"],
                    "muse_revision":MUSE_REVISION,"smol_config_revision":SMOL_CONFIG_REVISION}
                path=out/f"checkpoint_{step}.pt"; torch.save(checkpoint,path)
                report["updates_reached"]=step; report["committed_checkpoints"].append({"step":step,"sha256":file_hash(path),
                    "recent_loss_mean":float(np.mean(losses[-100:])),"training_seconds":time.monotonic()-train_start})
                (out/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n"); volume.commit()
                print(json.dumps({"committed":step}),flush=True)
        if step not in (5000,7500,10000):
            torch.save({"completed_updates":step,"prefix":{k:v.detach().cpu() for k,v in prefix.state_dict().items()},
                "trainable_policy":{n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad},
                "optimizer":optimizer.state_dict(),"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all(),
                "frame_schedule_next_index":step%150,"normalize_queries":True,"dynamics_objective":False,
                "source_checkpoint_sha256":report["source_checkpoint_sha256"],"evaluation_noise_sha256":report["evaluation_noise_sha256"],
                "optimizer_restart_at_3000":True,"rng_restart_seed":8008,"dataset_fingerprint":splits["fingerprint"]},out/f"checkpoint_{step}.pt")
            report["updates_reached"]=step
        report["status"]="completed" if step==10000 else "budget_stopped"
        report["training_seconds"]=time.monotonic()-train_start
        return report
    except Exception as exc:
        report.update(status="failed",error_type=type(exc).__name__,error=str(exc)); raise
    finally:
        if muse is not None: muse.close()
        report["remote_function_seconds"]=time.monotonic()-started
        report["requested_resource_cost_estimate_usd"]=report["remote_function_seconds"]*(4.54+4*.0473+64*.008)/3600
        (out/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n"); volume.commit()


@app.function(image=image, cpu=2, memory=16384, volumes={"/data": volume}, timeout=600, max_containers=1)
def audit(name: str = "quantized_muse_exposure_v1", output: str = "quantized_exposure_checkpoint_audit_v1"):
    """Verify checkpoint parameters, optimizer restoration and saved RNG metadata on CPU."""
    import json
    from pathlib import Path
    import torch
    import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as qwen
    if not hasattr(qwen, "apply_multimodal_rotary_pos_emb"):
        def apply_multimodal_rotary_pos_emb(q,k,cos,sin,mrope_section,unsqueeze_dim=1):
            sections=mrope_section*2
            cos=torch.cat([m[i%3] for i,m in enumerate(cos.split(sections,dim=-1))],dim=-1).unsqueeze(unsqueeze_dim)
            sin=torch.cat([m[i%3] for i,m in enumerate(sin.split(sections,dim=-1))],dim=-1).unsqueeze(unsqueeze_dim)
            return (q*cos)+(qwen.rotate_half(q)*sin), (k*cos)+(qwen.rotate_half(k)*sin)
        qwen.apply_multimodal_rotary_pos_emb=apply_multimodal_rotary_pos_emb
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.constants import ACTION, OBS_STATE
    from odvla.data import file_hash, read_json
    from odvla.vla.smolvla_bridge import MuseSamPrefix
    for value in (name,output):
        if not value or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in value): raise ValueError("Invalid name")
    run=Path("/data/eval")/name; report=read_json(run/"report.json")
    dest=Path("/data/eval")/output; dest.mkdir(parents=True,exist_ok=False)
    result={"status":"started","checkpoints":[],"scope":"CPU restore audit; no numerical next-update replay"}
    try:
        cfg=SmolVLAConfig(device="cpu",vlm_model_name=str(Path("/data/checkpoints/smolvlm2-reference")/SMOL_CONFIG_REVISION),
            load_vlm_weights=False,input_features={OBS_STATE:PolicyFeature(type=FeatureType.STATE,shape=(15,)),
                "observation.images.agentview":PolicyFeature(type=FeatureType.VISUAL,shape=(3,512,512))},
            output_features={ACTION:PolicyFeature(type=FeatureType.ACTION,shape=(7,))})
        model=SmolVLAPolicy(cfg).model; model.vlm_with_expert.vlm.requires_grad_(False)
        tc=model.vlm_with_expert.config.text_config
        prefix=MuseSamPrefix(muse_width=6656,layers=model.vlm_with_expert.num_vlm_layers,heads=tc.num_key_value_heads,
            head_dim=tc.head_dim,normalize_queries=True)
        trainable=list(prefix.parameters())+[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(trainable,lr=1e-4,betas=(.9,.95))
        params=dict(model.named_parameters())
        for item in report["committed_checkpoints"]:
            step=item["step"]; path=run/f"checkpoint_{step}.pt"
            if file_hash(path)!=item["sha256"]: raise ValueError(f"Checkpoint hash mismatch: {step}")
            ck=torch.load(path,map_location="cpu",weights_only=True)
            if ck["completed_updates"]!=step or ck["frame_schedule_next_index"]!=step%150 or not ck["normalize_queries"]:
                raise ValueError("Checkpoint metadata mismatch")
            if set(ck["trainable_policy"])!={n for n,p in params.items() if p.requires_grad}: raise ValueError("Trainable parameter set mismatch")
            prefix.load_state_dict(ck["prefix"],strict=True)
            with torch.no_grad():
                for k,v in ck["trainable_policy"].items(): params[k].copy_(v)
            optimizer.load_state_dict(ck["optimizer"])
            if not optimizer.state or not ck["cuda_rng"] or not isinstance(ck["torch_rng"],torch.Tensor):
                raise ValueError("Missing optimizer or RNG state")
            if ck["evaluation_noise_sha256"]!=report["evaluation_noise_sha256"]: raise ValueError("Noise hash mismatch")
            result["checkpoints"].append({"step":step,"sha256":item["sha256"],"optimizer_state_entries":len(optimizer.state),
                "cpu_rng_bytes":int(ck["torch_rng"].numel()),"cuda_rng_states":len(ck["cuda_rng"]),
                "next_frame":ck["frame_schedule_next_index"]})
            del ck
        result["status"]="passed"
        return result
    finally:
        (dest/"report.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n"); volume.commit()


@app.local_entrypoint()
def main(name: str = "quantized_muse_exposure_v1", max_train_seconds: int = 900, stage: str = "train", output: str = "quantized_exposure_checkpoint_audit_v1"):
    if stage == "train": print(train.remote(name,max_train_seconds))
    elif stage == "audit": print(audit.remote(name,output))
    else: raise ValueError("Unknown stage")
