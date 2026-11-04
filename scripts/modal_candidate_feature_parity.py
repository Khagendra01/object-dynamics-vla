"""Action-free H200/A100 NF4 Muse feature and precision parity probe."""
import modal
from scripts.modal_quantized_muse_preflight import image as reference_image
from scripts.modal_muse_smolvla_live import vision_image

volume=modal.Volume.from_name("object-dynamics-vla-data",create_if_missing=False)
app=modal.App("odvla-candidate-feature-parity")


@app.function(image=reference_image,gpu="H200",cpu=2,memory=65536,volumes={"/data":volume},timeout=300,max_containers=1,single_use_containers=True)
def reference(name: str="muse_candidate_feature_reference_v1"):
    import hashlib,json,time
    from pathlib import Path
    import numpy as np
    import torch
    from PIL import Image
    from odvla.data import read_json
    from odvla.vla.muse_context import FrozenMuseTokens,backend_snapshot
    if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name): raise ValueError("Invalid name")
    out=Path("/data/eval")/name; out.mkdir(parents=True,exist_ok=False); start=time.monotonic(); muse=None
    report={"status":"started","actions_executed":0,"gpu":"H200"}
    try:
        prior=read_json("/data/eval/quantized_muse_full_prenorm_v1/report.json")
        muse=FrozenMuseTokens("/data/checkpoints/muse-glimmer-30b/cec94fcfd6512f2cfe708138c32533b346ea4f69",26,quantization="nf4")
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        report["backend"]=backend_snapshot(torch,muse.model)
        frame=Path("/data/libero/milk_sam_native_v1/episode_0000/000000.jpg")
        with Image.open(frame) as im: token=muse.encode(im,"pick up the milk and place it in the basket",current_date="2026-10-30")
        digest=hashlib.sha256(token["features_bfloat16_bits"].tobytes()).hexdigest()
        report.update(status="completed",rgb_sha256=hashlib.sha256(frame.read_bytes()).hexdigest(),feature_sha256=digest,
            training_feature_sha256=prior["frames"][0]["feature_sha256"],matches_training=digest==prior["frames"][0]["feature_sha256"])
        np.savez(out/"token.npz",**token)
        return report
    except Exception as exc:
        report.update(status="failed",error_type=type(exc).__name__,error=str(exc)); raise
    finally:
        if muse is not None: muse.close()
        report["seconds"]=time.monotonic()-start
        (out/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n"); volume.commit()


@app.function(image=vision_image,gpu="A100-80GB",cpu=4,memory=131072,volumes={"/data":volume},timeout=300,max_containers=1,single_use_containers=True)
def probe(name: str="muse_candidate_feature_a100_v1"):
    import hashlib,json,sys,time,tempfile
    from pathlib import Path
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from odvla.data import file_hash,read_json
    from odvla.vla.muse_context import FrozenMuseTokens,backend_snapshot
    from odvla.sam_incremental import IncrementalPointTracker
    from odvla.sam_prefix import replay_current
    from sam3.model_builder import build_sam3_multiplex_video_predictor
    sys.path.insert(0,"/root")
    from extract_sam31 import _patch_multiplex_init_state
    if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name): raise ValueError("Invalid name")
    out=Path("/data/eval")/name; out.mkdir(parents=True,exist_ok=False); start=time.monotonic(); muse=None
    report={"status":"started","actions_executed":0,"gpu":torch.cuda.get_device_name(0)}
    try:
        reference=Path("/data/eval/muse_candidate_feature_reference_v1")
        ref_report=read_json(reference/"report.json")
        if not ref_report["matches_training"]: raise ValueError("H200 reference did not reproduce training hash")
        config=read_json("/root/sam31_milk_development_config.json"); sam=Path("/data/checkpoints/sam3.1/sam3.1_multiplex.pt")
        if file_hash(sam)!=config["checkpoint_sha256"]: raise ValueError("SAM checkpoint mismatch")
        torch.manual_seed(2468); np.random.seed(2468)
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        tracker=build_sam3_multiplex_video_predictor(checkpoint_path=str(sam),use_fa3=False,compile=False,warm_up=False,async_loading_frames=False)
        _patch_multiplex_init_state(tracker)
        muse=FrozenMuseTokens("/data/checkpoints/muse-glimmer-30b/cec94fcfd6512f2cfe708138c32533b346ea4f69",26,quantization="nf4")
        report["after_constructors"]=backend_snapshot(torch,muse.model)
        frame=Path("/data/libero/milk_sam_native_v1/episode_0000/000000.jpg")
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp); first=folder/"000000.jpg"; first.write_bytes(frame.read_bytes())
            incremental=IncrementalPointTracker(tracker,first,config["first_frame_points_xy_pixels_512"],config["object_ids"],capacity=200)
            try:
                masks,ids=incremental.step(first)
                ref_masks,ref_ids=replay_current(tracker,folder,1,config["first_frame_points_xy_pixels_512"],config["object_ids"])
                if not np.array_equal(masks,ref_masks) or not np.array_equal(ids,ref_ids): raise ValueError("SAM initial replay mismatch")
            finally: incremental.close()
        report["after_sam_inference"]=backend_snapshot(torch,muse.model)
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        report["after_reset"]=backend_snapshot(torch,muse.model)
        with Image.open(frame) as im:
            with torch.autocast("cuda",enabled=False): token=muse.encode(im,"pick up the milk and place it in the basket",current_date="2026-10-30")
        with np.load(reference/"token.npz",allow_pickle=False) as z: ref={k:z[k].copy() for k in z.files}
        a=torch.from_numpy(token["features_bfloat16_bits"].view(np.int16)).view(torch.bfloat16).float()
        b=torch.from_numpy(ref["features_bfloat16_bits"].view(np.int16)).view(torch.bfloat16).float()
        diff=(a-b).abs()
        report.update(status="completed",feature_sha256=hashlib.sha256(token["features_bfloat16_bits"].tobytes()).hexdigest(),
            reference_sha256=ref_report["feature_sha256"],feature_bits_exact=bool(np.array_equal(token["features_bfloat16_bits"],ref["features_bfloat16_bits"])),
            mask_exact=bool(np.array_equal(token["attention_mask"],ref["attention_mask"])),ids_exact=bool(np.array_equal(token["input_ids"],ref["input_ids"])),
            max_abs=float(diff.max()),mean_abs=float(diff.mean()),cosine=float(F.cosine_similarity(a.flatten()[None],b.flatten()[None])))
        np.savez(out/"token.npz",**token)
        return report
    except Exception as exc:
        report.update(status="failed",error_type=type(exc).__name__,error=str(exc)); raise
    finally:
        if muse is not None: muse.close()
        report["seconds"]=time.monotonic()-start
        (out/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n"); volume.commit()


@app.function(image=vision_image,gpu="H200",cpu=4,memory=131072,volumes={"/data":volume},timeout=300,max_containers=1,single_use_containers=True)
def probe_h200(name: str="muse_candidate_feature_h200_sam_v1"):
    return probe.local(name)


@app.local_entrypoint()
def main(stage: str="reference",name: str=""):
    if stage=="reference": print(reference.remote(name or "muse_candidate_feature_reference_v1"))
    elif stage=="probe": print(probe.remote(name or "muse_candidate_feature_a100_v1"))
    elif stage=="probe-h200": print(probe_h200.remote(name or "muse_candidate_feature_h200_sam_v1"))
    else: raise ValueError("Unknown stage")
