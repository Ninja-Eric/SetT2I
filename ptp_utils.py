"""SET encoder-based input-level backdoor detection utilities."""

import os
import gc
import time
import math

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel
import matplotlib.pyplot as plt
from safetensors.torch import save_file, load_file
from datetime import datetime
import traceback
import subprocess
import tempfile
import pickle
import re
from sklearn.metrics import roc_curve, auc, confusion_matrix
from concurrent.futures import ProcessPoolExecutor, as_completed

GPU_ID = '0'
if '--gpu' in sys.argv:
    try:
        idx = sys.argv.index('--gpu')
        if idx + 1 < len(sys.argv):
            GPU_ID = sys.argv[idx + 1].strip()
    except ValueError:
        pass

os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

print(f"[Config] GPU ID: {GPU_ID}")
print(f"[Config] Main device: {DEVICE}")

BASE_SD_PATH = os.environ.get("SET_BASE_MODEL_PATH", "stable-diffusion-v1-4")
BASE_OUTPUT_DIR = os.environ.get("SET_OUTPUT_DIR", ".")
V4_PARENT_DIR = os.environ.get("SET_RUN_DIR", os.path.join(BASE_OUTPUT_DIR, "outputs"))

################################################################################
################################################################################

BACKDOOR_CONFIGS = {
    "rickrolling": {
        "poisoned_model_path": "checkpoints/rickrolling",
        "train_prompt_file": "data/train_prompts.txt",
        "prompt_file": "data/eval_prompts/rickrolling.txt",
        "output_model": f"{BASE_OUTPUT_DIR}/models/detector_encoder_rickrolling_v4.safetensors",
        "output_dir": f"{BASE_OUTPUT_DIR}/train_encoder_rickrolling",
        "model_type": "text_encoder",
    },
    "twt": {
        "poisoned_model_path": "checkpoints/twt",
        "train_prompt_file": "data/train_prompts.txt",
        "prompt_file": "data/eval_prompts/twt.txt",
        "output_model": f"{BASE_OUTPUT_DIR}/models/detector_encoder_twt_v4.safetensors",
        "output_dir": f"{BASE_OUTPUT_DIR}/train_encoder_twt",
        "model_type": "text_encoder",
    },
    "villan_mignneko": {
        "poisoned_model_path": "checkpoints/villan_mignneko.safetensors",
        "train_prompt_file": "data/train_prompts.txt",
        "prompt_file": "data/eval_prompts/villan_mignneko.txt",
        "output_model": f"{BASE_OUTPUT_DIR}/models/detector_encoder_villan_mignneko_v4.safetensors",
        "output_dir": f"{BASE_OUTPUT_DIR}/train_encoder_villan_mignneko",
        "model_type": "lora",
    },
    "villan_github": {
        "poisoned_model_path": "checkpoints/villan_github",
        "train_prompt_file": "data/train_prompts.txt",
        "prompt_file": "data/eval_prompts/villan_github.txt",
        "output_model": f"{BASE_OUTPUT_DIR}/models/detector_encoder_villan_github_v4.safetensors",
        "output_dir": f"{BASE_OUTPUT_DIR}/train_encoder_villan_github",
        "model_type": "lora",
    },
    "villan_anonymous": {
        "poisoned_model_path": "checkpoints/villan_anonymous",
        "train_prompt_file": "data/train_prompts.txt",
        "prompt_file": "data/eval_prompts/villan_anonymous.txt",
        "output_model": f"{BASE_OUTPUT_DIR}/models/detector_encoder_villan_anonymous_v4.safetensors",
        "output_dir": f"{BASE_OUTPUT_DIR}/train_encoder_villan_anonymous",
        "model_type": "lora",
    },
    "eviledit": {
        "poisoned_model_path": "checkpoints/eviledit.pt",
        "train_prompt_file": "data/train_prompts.txt",
        "prompt_file": "data/eval_prompts/eviledit.txt",
        "output_model": f"{BASE_OUTPUT_DIR}/models/detector_encoder_eviledit_v4.safetensors",
        "output_dir": f"{BASE_OUTPUT_DIR}/train_encoder_eviledit",
        "model_type": "unet_state_dict",
    },
    "pixel": {
        "poisoned_model_path": "checkpoints/pixel",
        "train_prompt_file": "data/train_prompts.txt",
        "prompt_file": "data/eval_prompts/pixel.txt",
        "output_model": f"{BASE_OUTPUT_DIR}/models/detector_encoder_pixel_v4.safetensors",
        "output_dir": f"{BASE_OUTPUT_DIR}/train_encoder_pixel",
        "model_type": "unet",   
    },
    "personal": {
        "poisoned_model_path": "checkpoints/personal_cat2dog",
        "train_prompt_file": "data/train_prompts.txt",
        "prompt_file": "data/eval_prompts/personal.txt",
        "output_model": f"{BASE_OUTPUT_DIR}/models/detector_encoder_personal_v4.safetensors",
        "output_dir": f"{BASE_OUTPUT_DIR}/train_encoder_personal",
        "model_type": "textual_inversion",
    },
    "clean": {
        "poisoned_model_path": None,
        "train_prompt_file": "data/train_prompts.txt",
        "prompt_file": "data/eval_prompts/clean.txt",
        "output_model": f"{BASE_OUTPUT_DIR}/models/detector_encoder_clean_v4.safetensors",
        "output_dir": f"{BASE_OUTPUT_DIR}/train_encoder_clean",
        "model_type": "clean",
    },
}

# ============================================================================
# ============================================================================
LAMBDA_VALUES = [0.2, 0.3, 7, 10, 20]
SCALING_FACTORS = LAMBDA_VALUES
NUM_SCALES = len(SCALING_FACTORS)
TARGET_SEQ_Q = 256
ALLOW_STAGES = {"down", "up"}
FIRST_STEP_ONLY = False
NUM_PROBE_STEPS = 5

RANDOM_SEED = 42
MAX_SAMPLES = None
NU = 0.05
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 100
DEFAULT_BATCH_SIZE = 20
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5

THRESHOLD_PERCENTILE = 0.99

class SyncCompareHook:
    """
    """
    def __init__(self, unet, lam=2.0):
        self.unet = unet
        self.lam = lam

        self.baseline_pv = {}
        self.current_step_pv_mse = {}

        self.cross_attn_counter = 0
        self.is_baseline = False

        self._original_forwards = {}
        self._attn_modules = []  # [(module, is_cross, stage)]
        self._identify_and_patch()


    def _identify_and_patch(self):
        self._attn_modules = []
        for name, child in self.unet.named_children():
            if "down" in name:
                stage = "down"
            elif "mid" in name:
                stage = "mid"
            elif "up" in name:
                stage = "up"
            else:
                continue
            self._collect_recursive(child, stage)

        # Patch forward
        for mod, is_cross, stage in self._attn_modules:
            self._original_forwards[id(mod)] = mod.forward
            mod.forward = self._make_forward(mod, is_cross, stage)

    def _collect_recursive(self, net, stage):
        if net.__class__.__name__ == "Attention":
            is_cross = net.to_q.in_features != net.to_k.in_features
            self._attn_modules.append((net, is_cross, stage))
        for c in net.children():
            self._collect_recursive(c, stage)

    def _make_forward(self, attn, is_cross, stage):
        hook = self
        orig = self._original_forwards[id(attn)]

        def fwd(hidden_states, encoder_hidden_states=None, attention_mask=None, **kw):
            # ============================================================
            # ============================================================
            if not is_cross:
                if not hook.is_baseline:
                    batch_size_sa = hidden_states.shape[0]
                    seq_len_sa = hidden_states.shape[1]

                    has_lora_sa = hasattr(attn, "processor") and hasattr(attn.processor, "to_q_lora")
                    if has_lora_sa:
                        ls_sa = kw.get("scale", 1.0) if kw else 1.0
                        q_sa = attn.to_q(hidden_states) + ls_sa * attn.processor.to_q_lora(hidden_states)
                        k_sa = attn.to_k(hidden_states) + ls_sa * attn.processor.to_k_lora(hidden_states)
                        v_sa = attn.to_v(hidden_states) + ls_sa * attn.processor.to_v_lora(hidden_states)
                    else:
                        q_sa = attn.to_q(hidden_states)
                        k_sa = attn.to_k(hidden_states)
                        v_sa = attn.to_v(hidden_states)

                    q_sa = attn.head_to_batch_dim(q_sa)
                    k_sa = attn.head_to_batch_dim(k_sa)
                    v_sa = attn.head_to_batch_dim(v_sa)

                    am_sa = attn.prepare_attention_mask(attention_mask, seq_len_sa, batch_size_sa)

                    scores_sa = torch.bmm(q_sa, k_sa.transpose(-1, -2)) * attn.scale
                    if am_sa is not None:
                        scores_sa = scores_sa + am_sa

                    probs_sa = (scores_sa * hook.lam).softmax(dim=-1)

                    pv_sa = torch.bmm(probs_sa, v_sa)
                    out_sa = attn.batch_to_head_dim(pv_sa)

                    if has_lora_sa:
                        ls_sa = kw.get("scale", 1.0) if kw else 1.0
                        out_sa = attn.to_out[0](out_sa) + ls_sa * attn.processor.to_out_lora(out_sa)
                    else:
                        out_sa = attn.to_out[0](out_sa)
                    return attn.to_out[1](out_sa)
                else:
                    return orig(hidden_states, encoder_hidden_states, attention_mask, **kw)

            # ============================================================
            # ============================================================
            hook.cross_attn_counter += 1
            lid = hook.cross_attn_counter
            batch_2B = hidden_states.shape[0]
            seq_q = hidden_states.shape[1]

            is_target = (stage in ALLOW_STAGES) and (seq_q == TARGET_SEQ_Q)

            ehs = (attn.norm_encoder_hidden_states(encoder_hidden_states)
                   if getattr(attn, "norm_cross", False) else encoder_hidden_states)
            bs, sl, _ = encoder_hidden_states.shape
            am = attn.prepare_attention_mask(attention_mask, sl, bs)

            has_lora = hasattr(attn, "processor") and hasattr(attn.processor, "to_q_lora")
            if has_lora:
                ls = kw.get("scale", 1.0) if kw else 1.0
                q = attn.to_q(hidden_states) + ls * attn.processor.to_q_lora(hidden_states)
                k = attn.to_k(ehs) + ls * attn.processor.to_k_lora(ehs)
                v = attn.to_v(ehs) + ls * attn.processor.to_v_lora(ehs)
            else:
                q, k, v = attn.to_q(hidden_states), attn.to_k(ehs), attn.to_v(ehs)

            q, k, v = attn.head_to_batch_dim(q), attn.head_to_batch_dim(k), attn.head_to_batch_dim(v)

            scores = torch.bmm(q, k.transpose(-1, -2)) * attn.scale
            if am is not None:
                scores = scores + am

            if is_target and not hook.is_baseline:
                probs = (scores * hook.lam).softmax(dim=-1)
            else:
                probs = scores.softmax(dim=-1)

            pv = torch.bmm(probs, v)  # (batch_2B*H, seq_q, d_v)

            if is_target:
                H = attn.heads
                B = batch_2B // 2
                cond_start = B * H
                pv_cond = pv[cond_start:]
                pv_4d = pv_cond.reshape(B, H, seq_q, pv_cond.shape[2])  # (B, H, seq_q, d_v)

                layer_key = (stage, lid)

                if hook.is_baseline:
                    hook.baseline_pv[layer_key] = pv_4d.detach()
                else:
                    if layer_key in hook.baseline_pv:
                        bl_pv = hook.baseline_pv[layer_key]
                        pv_mse_per_pos = ((pv_4d.float() - bl_pv.float()) ** 2).mean(dim=-1)
                        pv_mse_spatial = pv_mse_per_pos.mean(dim=1)  # (B, seq_q)
                        hook.current_step_pv_mse[layer_key] = pv_mse_spatial.cpu().numpy()

            out = attn.batch_to_head_dim(pv)
            if has_lora:
                ls = kw.get("scale", 1.0) if kw else 1.0
                out = attn.to_out[0](out) + ls * attn.processor.to_out_lora(out)
            else:
                out = attn.to_out[0](out)
            return attn.to_out[1](out)

        return fwd

    def reset_step_counter(self):
        self.cross_attn_counter = 0
        self.current_step_pv_mse = {}

    def clear_baseline(self):
        self.baseline_pv = {}

    def get_step_pv_mse(self):
        return dict(self.current_step_pv_mse)
    
    def restore(self):
        for mod, is_cross, stage in self._attn_modules:
            mid = id(mod)
            if mid in self._original_forwards:
                mod.forward = self._original_forwards[mid]
        self._original_forwards.clear()
        self._attn_modules.clear()

################################################################################
################################################################################

class CounterfactualFingerprintGenerator:
    """

    """
    def __init__(self, pipeline, device: str = "cuda:0", lambda_values: list = None):
        self.pipe = pipeline
        self.device = device
        self.lambda_values = lambda_values or LAMBDA_VALUES
        self.num_inference_steps = NUM_INFERENCE_STEPS
        self.guidance_scale = GUIDANCE_SCALE
        self.num_scales = len(self.lambda_values)

        self.num_selected_layers, self.layer_stages = self._probe_selected_layers()
        print(f"[Generator] Selected layers (down/up, seq_q={TARGET_SEQ_Q}): {self.num_selected_layers}, Scales: {self.num_scales}, Mode: PV-only")
        print(f"[Generator] Layer stages: {self.layer_stages}")
        print(f"[Generator] Lambda values: {self.lambda_values}")
        print(f"[Generator] Fingerprint dim per sample: {self.num_scales * self.num_selected_layers}")

    def _probe_selected_layers(self):
        hook = SyncCompareHook(self.pipe.unet, lam=1.0)
        try:
            device = self.device
            dtype = torch.float16
            g = torch.Generator(device=device).manual_seed(0)
            C = self.pipe.unet.config.in_channels
            lat = torch.randn((1, C, 64, 64), generator=g, device=device, dtype=dtype)

            ui = self.pipe.tokenizer([""], padding="max_length",
                                      max_length=self.pipe.tokenizer.model_max_length, return_tensors="pt")
            uncond = self.pipe.text_encoder(ui.input_ids.to(device))[0]
            ctx = torch.cat([uncond, uncond], dim=0)

            self.pipe.scheduler.set_timesteps(self.num_inference_steps, device=device)
            lat = lat * self.pipe.scheduler.init_noise_sigma
            t0 = self.pipe.scheduler.timesteps[0]

            hook.is_baseline = True
            hook.reset_step_counter()
            li = self.pipe.scheduler.scale_model_input(torch.cat([lat, lat], dim=0), t0)
            with torch.no_grad():
                _ = self.pipe.unet(li, t0, encoder_hidden_states=ctx).sample

            n = len(hook.baseline_pv)
            if n == 0:
                print(f"[Generator] WARNING: No layers passed filter (ALLOW_STAGES={ALLOW_STAGES}, TARGET_SEQ_Q={TARGET_SEQ_Q})")
                print(f"[Generator] Hook had {hook.cross_attn_counter} cross-attn forward calls")
                return n, []
            sorted_keys = sorted(hook.baseline_pv.keys(), key=lambda k: k[1])
            layer_stages = [k[0] for k in sorted_keys]  # e.g. ["down", "down", "up", "up", "up"]
            return n, layer_stages
        finally:
            hook.restore()
    
    @torch.no_grad()
    def generate_fingerprint(self, prompts: list, seed: int = 42) -> np.ndarray:
        """

        """
        if isinstance(prompts, str):
            prompts = [prompts]

        B = len(prompts)
        device = self.device
        dtype = torch.float16
        num_steps = NUM_PROBE_STEPS

        ui = self.pipe.tokenizer([""] * B, padding="max_length",
                                  max_length=self.pipe.tokenizer.model_max_length, return_tensors="pt")
        uncond_emb = self.pipe.text_encoder(ui.input_ids.to(device))[0]

        ti = self.pipe.tokenizer(prompts, padding="max_length",
                                  max_length=self.pipe.tokenizer.model_max_length,
                                  truncation=True, return_tensors="pt")
        cond_emb = self.pipe.text_encoder(ti.input_ids.to(device))[0]

        ctx = torch.cat([uncond_emb, cond_emb], dim=0)  # (2B, 77, 768)

        C = self.pipe.unet.config.in_channels
        lats = []
        for i in range(B):
            g = torch.Generator(device=device).manual_seed(seed + i)
            lats.append(torch.randn((1, C, 64, 64), generator=g, device=device, dtype=dtype))
        lat_init = torch.cat(lats, dim=0)  # (B, 4, 64, 64)

        self.pipe.scheduler.set_timesteps(self.num_inference_steps, device=device)
        lat_init = lat_init * self.pipe.scheduler.init_noise_sigma
        timesteps = self.pipe.scheduler.timesteps[:num_steps]

        all_scale_fingerprints = []
        canonical_keys = None

        try:
            #    baseline_pvs_per_step[step_idx] = {layer_key: pv_tensor}
            baseline_pvs_per_step = {}
            baseline_hook = SyncCompareHook(self.pipe.unet, lam=1.0)
            try:
                lat_bl = lat_init.clone()
                for step_idx, t in enumerate(timesteps):
                    baseline_hook.is_baseline = True
                    baseline_hook.reset_step_counter()
                    baseline_hook.clear_baseline()

                    li = self.pipe.scheduler.scale_model_input(torch.cat([lat_bl, lat_bl], dim=0), t)
                    noise_pred = self.pipe.unet(li, t, encoder_hidden_states=ctx).sample

                    # Classifier-free guidance
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred_cfg = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

                    # Scheduler step
                    lat_bl = self.pipe.scheduler.step(noise_pred_cfg, t, lat_bl).prev_sample

                    baseline_pvs_per_step[step_idx] = dict(baseline_hook.baseline_pv)
            finally:
                baseline_hook.restore()
            del lat_bl

            for scale_idx, lam in enumerate(self.lambda_values):
                step_fps = []
                lat_probe = lat_init.clone()

                hook = SyncCompareHook(self.pipe.unet, lam=lam)
                try:
                    for step_idx, t in enumerate(timesteps):
                        hook.baseline_pv = baseline_pvs_per_step[step_idx]
                        hook.is_baseline = False
                        hook.reset_step_counter()

                        li = self.pipe.scheduler.scale_model_input(torch.cat([lat_probe, lat_probe], dim=0), t)
                        noise_pred = self.pipe.unet(li, t, encoder_hidden_states=ctx).sample

                        # CFG
                        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                        noise_pred_cfg = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

                        lat_probe = self.pipe.scheduler.step(noise_pred_cfg, t, lat_probe).prev_sample

                        pv_mse_dict = hook.get_step_pv_mse()

                        if canonical_keys is None:
                            canonical_keys = sorted(pv_mse_dict.keys(), key=lambda k: k[1])
                            stages_found = set(k[0] for k in canonical_keys)
                            print(f"[Generator] Multi-step probe: {len(canonical_keys)} layers passed filter")
                            print(f"[Generator]   Stages: {stages_found}")
                            print(f"[Generator]   Layer keys: {canonical_keys}")
                            print(f"[Generator]   Steps: {num_steps}, Total dim: {self.num_scales * len(canonical_keys) * num_steps}")

                        n_layers = len(canonical_keys)
                        step_fp = np.zeros((B, n_layers), dtype=np.float32)
                        for layer_idx, key in enumerate(canonical_keys):
                            if key in pv_mse_dict:
                                spatial_mse = pv_mse_dict[key]  # (B, 256)
                                step_fp[:, layer_idx] = np.mean(spatial_mse, axis=-1)

                        step_fps.append(step_fp)
                finally:
                    hook.restore()

                scale_fp = np.concatenate(step_fps, axis=1)
                all_scale_fingerprints.append(scale_fp)

            del baseline_pvs_per_step

            fingerprints = np.concatenate(all_scale_fingerprints, axis=1)
            return fingerprints

        except Exception as e:
            print(f"[Generator] Error during fingerprint generation: {e}")
            raise e

################################################################################
################################################################################

class Encoder(nn.Module):
    """
    Response encoder: phi(per-layer) -> aggregate -> rho(global)
    
    
    """
    def __init__(self, num_layers: int, num_scales: int = 3, num_steps: int = 1,
                 phi_hidden: int = 64, embedding_dim: int = 32,
                 d_model: int = 128, nhead: int = 4,
                 num_transformer_layers: int = 2, dropout: float = 0.1,
                 layer_stages: list = None):
        super().__init__()
        self.num_layers = num_layers
        self.num_scales = num_scales
        self.num_steps = num_steps
        
        token_input_dim = num_scales * num_steps
        
        self.phi = nn.Sequential(
            nn.Linear(token_input_dim, phi_hidden),
            nn.GELU(),
            nn.Linear(phi_hidden, phi_hidden),
            nn.GELU(),
        )
        
        self.rho = nn.Sequential(
            nn.Linear(phi_hidden * 2, phi_hidden),
            nn.GELU(),
            nn.Linear(phi_hidden, embedding_dim, bias=False),
        )
    
    def forward(self, x):
        """
        Args:
        Returns:
            embedding: (B, embedding_dim)
        """
        B = x.shape[0]
        S = self.num_scales
        T = self.num_steps
        L = self.num_layers
        
        # (B, S*T*L) → (B, S, T, L) → (B, L, S*T)
        x = x.view(B, S, T, L).permute(0, 3, 1, 2).reshape(B, L, S * T)
        
        # φ: shared MLP per layer → (B, L, phi_hidden)
        h = self.phi(x)
        
        # Aggregate: mean + std over layers (permutation-invariant)
        h_mean = torch.mean(h, dim=1)   # (B, phi_hidden)
        h_std = torch.sqrt(torch.var(h, dim=1) + 1e-6)
        h_global = torch.cat([h_mean, h_std], dim=-1)  # (B, 2*phi_hidden)
        
        # ρ: global MLP → (B, embedding_dim)
        embedding = self.rho(h_global)
        return embedding


################################################################################
# Encoder Detector
################################################################################

class EncoderDetector:
    def __init__(self, num_layers: int, num_scales: int = NUM_SCALES, num_steps: int = NUM_PROBE_STEPS,
                 embedding_dim: int = 32, nu: float = 0.05,
                 lr: float = 1e-3, weight_decay: float = 1e-4,
                 batch_size: int = 256, num_epochs: int = 100, device: str = "cuda",
                 warmup_epochs: int = 5, transformer_config: dict = None, input_dim: int = 1,
                 layer_stages: list = None):
        self.num_layers = num_layers
        self.num_scales = num_scales
        self.num_steps = num_steps
        self.embedding_dim = embedding_dim
        self.nu = nu
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.warmup_epochs = warmup_epochs
        self.device = device
        self.input_dim = input_dim  # 1 for PV-only, 2 for PV+Attn
        self.layer_stages = layer_stages  # e.g. ["down", "down", "up", "up", "up"]
        
        self.transformer_config = transformer_config or {
            'd_model': 128,
            'nhead': 4,
            'num_transformer_layers': 2,
            'dropout': 0.1
        }
        
        self.encoder = self._build_network().to(self.device)
        self.center = None
        self.radius = torch.tensor(0.0, device=self.device)
        self.mean = None
        self.std = None

    def _build_network(self):
        return Encoder(
            num_layers=self.num_layers,
            num_scales=self.num_scales,
            num_steps=self.num_steps,
            phi_hidden=64,
            embedding_dim=self.embedding_dim,
        )

    def fit(self, fingerprints_high_dim: np.ndarray, prompts: list = None):
        print(f"Encoder: fitting on data shape {fingerprints_high_dim.shape}...")
        self.mean = np.mean(fingerprints_high_dim, axis=0)
        self.std = np.std(fingerprints_high_dim, axis=0)
        self.std[self.std < 1e-8] = 1.0

        X_std = (fingerprints_high_dim - self.mean) / self.std
        X_tensor = torch.tensor(X_std, dtype=torch.float32).to(self.device)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.encoder.eval()
        with torch.no_grad():
            z_sum = torch.zeros(self.embedding_dim, device=self.device)
            n_samples = 0
            init_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
            for (batch_x,) in init_loader:
                batch_x = batch_x.to(self.device)
                z_batch = self.encoder(batch_x)
                z_sum += torch.sum(z_batch, dim=0)
                n_samples += batch_x.size(0)
            
            self.center = z_sum / n_samples
            self.center.requires_grad = False
        print(f"  Initialized Center. Shape: {self.center.shape}")

        R = torch.tensor(0.1, device=self.device, requires_grad=True)
        optimizer = optim.Adam([
            {'params': self.encoder.parameters(), 'weight_decay': self.weight_decay},
            {'params': [R], 'weight_decay': 0.0},
        ], lr=self.lr)

        print(f"  Starting Warmup ({self.warmup_epochs} epochs)...")
        self.encoder.train()
        for epoch in range(self.warmup_epochs):
            total_loss = 0.0
            for (batch_x,) in dataloader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()
                outputs = self.encoder(batch_x)
                dist = torch.sum((outputs - self.center) ** 2, dim=1)
                r_sq = R ** 2
                penalty = torch.relu(dist - r_sq)
                loss = r_sq + (1.0 / self.nu) * torch.mean(penalty)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"    Warmup Epoch {epoch + 1}/{self.warmup_epochs} | Loss: {total_loss / len(dataloader):.6f} | R: {R.item():.4f}")

        print("  Re-estimating Center using Trimmed Mean (removing top 1% outliers)...")
        self.encoder.eval()
        with torch.no_grad():
            z_all = []
            temp_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
            for (batch_x,) in temp_loader:
                z_batch = self.encoder(batch_x.to(self.device))
                z_all.append(z_batch)
            z_all = torch.cat(z_all, dim=0)
            
            c_tmp = torch.mean(z_all, dim=0)
            dists = torch.sum((z_all - c_tmp) ** 2, dim=1)
            
            threshold = torch.quantile(dists, 0.99)
            mask = dists <= threshold
            
            removed_indices = torch.nonzero(~mask).squeeze(1).cpu().numpy()
            removed_dists = dists[~mask].cpu().numpy()
            
            if prompts is not None and len(removed_indices) > 0:
                print(f"\n[Removed Outliers] Top {len(removed_indices)} samples removed during Center Re-estimation:")
                for i, idx in enumerate(removed_indices):
                    if idx < len(prompts):
                        prompt_text = prompts[idx].replace('\n', ' ')[:100] 
                        print(f"  - Index {idx} | Distance: {removed_dists[i]:.4e} | Prompt: {prompt_text}...")
                    else:
                        print(f"  - Index {idx} | Distance: {removed_dists[i]:.4e} | Prompt: [Prompt not available/out of range]...")

            new_center = torch.mean(z_all[mask], dim=0)
            self.center = new_center.detach()
            print(f"  Center updated. Used {mask.sum().item()}/{len(dists)} samples.")

        R = torch.tensor(0.1, device=self.device, requires_grad=True)
        optimizer = optim.Adam([
            {'params': self.encoder.parameters(), 'weight_decay': self.weight_decay},
            {'params': [R], 'weight_decay': 0.0},
        ], lr=self.lr)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=1e-5)

        print(f"  Starting Main Training ({self.num_epochs} epochs)...")
        self.encoder.train()
        history = {'loss': [], 'r_sq': [], 'penalty': [], 'avg_dist': []}
        
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            epoch_r_sq = 0.0
            epoch_penalty = 0.0
            epoch_dist = 0.0
            n_batches = 0
            
            for (batch_x,) in dataloader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()
                outputs = self.encoder(batch_x)
                dist = torch.sum((outputs - self.center) ** 2, dim=1)
                r_sq = R ** 2
                penalty = torch.relu(dist - r_sq)
                mean_penalty = torch.mean(penalty)
                
                loss = r_sq + (1.0 / self.nu) * mean_penalty
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_r_sq += r_sq.item()
                epoch_penalty += mean_penalty.item()
                epoch_dist += torch.mean(dist).item()
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            avg_r_sq = epoch_r_sq / n_batches
            avg_penalty = epoch_penalty / n_batches
            avg_dist = epoch_dist / n_batches
            
            history['loss'].append(avg_loss)
            history['r_sq'].append(avg_r_sq)
            history['penalty'].append(avg_penalty)
            history['avg_dist'].append(avg_dist)
            
            scheduler.step()
            
            print(f"  Epoch {epoch + 1}/{self.num_epochs} | Loss: {avg_loss:.6f} | R^2: {avg_r_sq:.6f} | Penalty: {avg_penalty:.6f} | Avg Dist: {avg_dist:.6f}")

        self.encoder.eval()
        
        train_R = float(R.item())
        train_R_sq = train_R ** 2
        
        with torch.no_grad():
            z_eval = []
            eval_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
            for (batch_x,) in eval_loader:
                z_batch = self.encoder(batch_x.to(self.device))
                z_eval.append(z_batch)
            z_eval = torch.cat(z_eval, dim=0)
            eval_dists = torch.sum((z_eval - self.center) ** 2, dim=1)
            
            eval_r_sq = float(torch.quantile(eval_dists, 1.0 - self.nu).item())
            self.radius = float(math.sqrt(eval_r_sq))
            self.threshold_percentile = float(torch.quantile(eval_dists, THRESHOLD_PERCENTILE).item())
            
            eval_avg_dist = float(torch.mean(eval_dists).item())
            eval_min_dist = float(torch.min(eval_dists).item())
            eval_max_dist = float(torch.max(eval_dists).item())
        
        print(f"✓ Encoder training finished.")
        print(f"  - Train-mode R: {train_R:.4f}, R²: {train_R_sq:.4e}")
        print(f"  - Eval-mode recalibration:")
        print(f"    Avg Dist: {eval_avg_dist:.4e}, Min: {eval_min_dist:.4e}, Max: {eval_max_dist:.4e}")
        print(f"    R (eval, {(1-self.nu)*100:.0f}%ile): {self.radius:.4f}, R²: {self.radius**2:.4e}")
        print(f"    {THRESHOLD_PERCENTILE*100:.0f}% Percentile Threshold: {self.threshold_percentile:.4e}")
        return history

    def score_samples(self, fingerprints_high_dim: np.ndarray) -> np.ndarray:
        X_std = (fingerprints_high_dim - self.mean) / self.std
        X_tensor = torch.tensor(X_std, dtype=torch.float32).to(self.device)
        self.encoder.eval()
        with torch.no_grad():
            z = self.encoder(X_tensor)
            dist = torch.sum((z - self.center) ** 2, dim=1)
        return dist.cpu().numpy()

    def save(self, filepath: str, layer_stages: list = None):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        tensors = {"center": self.center.cpu(), "mean": torch.tensor(self.mean, dtype=torch.float32),
                   "std": torch.tensor(self.std, dtype=torch.float32)}
        for k, v in self.encoder.state_dict().items():
            tensors[f"encoder.{k}"] = v.cpu()
        save_file(tensors, filepath)
        
        meta = {
            "radius": self.radius, 
            "threshold_percentile": getattr(self, 'threshold_percentile', None),
            "threshold_percentile_value": THRESHOLD_PERCENTILE,
            "lambda_values": LAMBDA_VALUES,
            "num_scales": NUM_SCALES,
            "layer_stages": layer_stages,  # e.g. ["down", "down", "up", "up", "up"]
            "hparams": {
                "num_layers": self.num_layers,
                "num_scales": self.num_scales,
                "num_steps": self.num_steps,
                "embedding_dim": self.embedding_dim,
                "nu": self.nu, 
                "arch": "encoder",
                "transformer_config": self.transformer_config,
                "input_dim": self.input_dim
            }
        }
        with open(filepath.replace(".safetensors", "_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"✓ Model saved to {filepath}")

    def load(self, filepath: str):
        meta_path = filepath.replace(".safetensors", "_meta.json")
        with open(meta_path, "r") as f:
            meta = json.load(f)
        
        self.radius = meta["radius"]
        self.threshold_percentile = meta.get("threshold_percentile", None)
        self.threshold_percentile_value = meta.get("threshold_percentile_value", 0.99)
        
        hparams = meta["hparams"]
        self.num_layers = hparams["num_layers"]
        self.num_scales = hparams.get("num_scales", meta.get("num_scales", NUM_SCALES))
        self.num_steps = hparams.get("num_steps", 1)
        self.embedding_dim = hparams["embedding_dim"]
        self.nu = hparams["nu"]
        self.input_dim = hparams.get("input_dim", 1)
        
        if "transformer_config" in hparams:
            self.transformer_config = hparams["transformer_config"]
        
        self.layer_stages = meta.get("layer_stages", None)
        
        self.encoder = self._build_network().to(self.device)
        
        tensors = load_file(filepath)
        self.center = tensors["center"].to(self.device)
        self.mean = tensors["mean"].numpy()
        self.std = tensors["std"].numpy()
        
        encoder_state = {}
        for k, v in tensors.items():
            if k.startswith("encoder."):
                encoder_state[k.replace("encoder.", "")] = v
        self.encoder.load_state_dict(encoder_state)
        self.encoder.to(self.device)
        
        print(f"✓ Model loaded from {filepath}")
        print(f"  - Num layers: {self.num_layers}, Embedding dim: {self.embedding_dim}")
        print(f"  - Radius: {self.radius:.4f}, R²: {self.radius**2:.4e}")
        if self.threshold_percentile is not None:
            print(f"  - {self.threshold_percentile_value*100:.0f}% Percentile Threshold: {self.threshold_percentile:.4e}")
        print(f"  - Architecture: {meta['hparams'].get('arch', 'unknown')}")


################################################################################
################################################################################

def load_prompts_from_file(filepath: str) -> list:
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]


def compute_per_scale_distances(fingerprints: np.ndarray, detector, scaling_factors: list) -> dict:
    """
    
    Args:
    
    Returns:
        dict: {scale: distances_array}
    """
    num_scales = len(scaling_factors)
    total_dim = fingerprints.shape[1]
    dim_per_scale = total_dim // num_scales
    
    X_std = (fingerprints - detector.mean) / detector.std
    X_tensor = torch.tensor(X_std, dtype=torch.float32).to(detector.device)
    
    scale_distances = {}
    
    detector.encoder.eval()
    with torch.no_grad():
        for i, scale in enumerate(scaling_factors):
            start_idx = i * dim_per_scale
            end_idx = (i + 1) * dim_per_scale
            
            fp_scale = X_tensor[:, start_idx:end_idx]
            
            fp_full = torch.zeros_like(X_tensor)
            fp_full[:, start_idx:end_idx] = fp_scale
            
            z = detector.encoder(fp_full)
            
            dist = torch.sum((z - detector.center) ** 2, dim=1)
            scale_distances[scale] = dist.cpu().numpy()
    
    return scale_distances

def plot_mse_heatmap(fingerprints, scaling_factors, output_path):
    """
    """
    try:
        if fingerprints is None or len(fingerprints) == 0:
            raise ValueError("fingerprints is empty")

        avg_fp = np.mean(fingerprints, axis=0)
        num_scales = len(scaling_factors)
        total_dim = len(avg_fp)
        
        dim_per_scale = total_dim // num_scales
        
        # total_dim=4896 => per_scale=816 => layers=16, steps=51
        candidate_layers = [16, 32]
        num_layers = None
        num_steps = None
        for cand in candidate_layers:
            if dim_per_scale % cand == 0:
                num_layers = cand
                num_steps = dim_per_scale // cand
                break

        if num_layers is None:
            num_steps = 1
            num_layers = dim_per_scale

        print(f"[Heatmap] total_dim={total_dim}, num_scales={num_scales}, dim_per_scale={dim_per_scale}, inferred_steps={num_steps}, inferred_layers={num_layers}")
        
        matrix_3d = avg_fp.reshape(num_scales, num_steps, num_layers)
        
        mse_matrix = np.mean(matrix_3d, axis=1)
        
        plt.figure(figsize=(14, 8))
        im = plt.imshow(mse_matrix, aspect='auto', cmap='YlOrRd')
        plt.colorbar(im, label='Average MSE')
        
        plt.yticks(range(num_scales), scaling_factors)
        plt.xticks(range(num_layers), [f'L{i}' for i in range(num_layers)])
        
        plt.title("Average MSE per Layer and Scaling Factor\n(Averaged over samples and time steps)")
        plt.xlabel("UNet Cross-Attention Layers")
        plt.ylabel("Scaling Factor")
        
        for i in range(num_scales):
            for j in range(num_layers):
                val = mse_matrix[i, j]
                color = "white" if val > np.max(mse_matrix) * 0.7 else "black"
                plt.text(j, i, f"{val:.1e}", 
                         ha="center", va="center", color=color, fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"✓ MSE Heatmap saved to {output_path}")

        txt_path = output_path.replace(".png", ".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("Detailed MSE Breakdown (Scale vs Layer)\n")
            f.write("-" * 50 + "\n")
            header = "Scale".ljust(10) + "".join([f"L{i}".rjust(12) for i in range(num_layers)])
            f.write(header + "\n")
            for i, scale in enumerate(scaling_factors):
                row = f"{scale}".ljust(10) + "".join([f"{mse_matrix[i, j]:.4e}".rjust(12) for j in range(num_layers)])
                f.write(row + "\n")
        print(f"✓ MSE Numerical breakdown saved to {txt_path}")

    except Exception as e:
        print(f"Warning: Failed to plot MSE heatmap: {e}")
        traceback.print_exc()

def load_poisoned_model(pipeline, config, device):
    model_type = config["model_type"]
    path = config["poisoned_model_path"]

    if model_type == "clean" or path is None:
        return pipeline

    if model_type == "text_encoder":
        encoder = CLIPTextModel.from_pretrained(path, local_files_only=True)
        pipeline.text_encoder = encoder.to(device).to(torch.float16)

    elif model_type == "lora":
        if os.path.isdir(path):
            files = os.listdir(path)
            safetensors_files = [f for f in files if f.endswith('.safetensors')]
            bin_files = [f for f in files if f.endswith('.bin')]
            if safetensors_files:
                target_file = os.path.join(path, safetensors_files[0])
                pipeline.load_lora_weights(pretrained_model_name_or_path_or_dict=target_file)
                pipeline.unet.load_attn_procs(target_file)
                pipeline.fuse_lora(lora_scale=1.0)
            elif bin_files:
                pipeline.unet.load_attn_procs(path)
                pipeline.fuse_lora(lora_scale=1.0)
        else:
            pipeline.load_lora_weights(pretrained_model_name_or_path_or_dict=path)
            pipeline.unet.load_attn_procs(path)
            pipeline.fuse_lora(lora_scale=1.0)

    elif model_type == "unet_state_dict":
        state_dict = torch.load(path, map_location=device)
        pipeline.unet.load_state_dict(state_dict)
        pipeline.unet.to(torch.float16)

    elif model_type == "unet":
        pipeline.unet = UNet2DConditionModel.from_pretrained(path, torch_dtype=torch.float16)
        pipeline.unet = pipeline.unet.to(device)

    elif model_type == "textual_inversion":
        import glob as _glob
        embed_path = os.path.join(path, "learned_embeds.bin")
        if not os.path.exists(embed_path):
            step_files = sorted(_glob.glob(os.path.join(path, "learned_embeds-step-*.bin")))
            if step_files:
                embed_path = step_files[-1]
                print(f"[Personal] Using step checkpoint: {embed_path}")
            else:
                raise FileNotFoundError(f"No learned_embeds*.bin found in {path}")

        embeds = torch.load(embed_path, weights_only=False, map_location=device)

        if "string_to_param" in embeds:
            token_name = list(embeds["string_to_param"].keys())[0]
            learned_embed = embeds["string_to_param"][token_name].squeeze()
        else:
            token_name = list(embeds.keys())[0]
            learned_embed = embeds[token_name].squeeze()

        print(f"[Personal] Trigger token: '{token_name}', embedding shape: {learned_embed.shape}")

        num_added = pipeline.tokenizer.add_tokens(token_name)
        token_id = pipeline.tokenizer.convert_tokens_to_ids(token_name)
        print(f"[Personal] Token '{token_name}' -> ID {token_id} (newly added: {num_added > 0})")

        pipeline.text_encoder.resize_token_embeddings(len(pipeline.tokenizer))
        pipeline.text_encoder.get_input_embeddings().weight.data[token_id] = learned_embed.to(
            dtype=pipeline.text_encoder.get_input_embeddings().weight.dtype
        )
        pipeline.text_encoder = pipeline.text_encoder.to(device).to(torch.float16)

    return pipeline


def generate_fingerprints_single(prompts, config, scaling_factors, seed, base_sd_path, output_dir, device, batch_size: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    print(f"[{device}] Loading model and extracting response-offset features for {len(prompts)} prompts...")

    pipeline = StableDiffusionPipeline.from_pretrained(
        base_sd_path, torch_dtype=torch.float16, local_files_only=True
    ).to(device)
    pipeline = load_poisoned_model(pipeline, config, device)
    pipeline.set_progress_bar_config(disable=True)

    generator = CounterfactualFingerprintGenerator(pipeline, device=device, lambda_values=scaling_factors)
    if generator.layer_stages:
        with open(os.path.join(output_dir, "layer_stages.json"), "w", encoding="utf-8") as f:
            json.dump(generator.layer_stages, f)

    fingerprints = []
    batch_size = max(1, batch_size)
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start:start + batch_size]
        fp_batch = generator.generate_fingerprint(batch_prompts, seed)
        fingerprints.extend(list(fp_batch))
        print(f"[{device}] Processed {min(start + batch_size, len(prompts))}/{len(prompts)} prompts", flush=True)

    del pipeline
    del generator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.array(fingerprints)

# ============================================================================
# Prompt-to-Prompt utility functions
# ============================================================================

# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import cv2
from typing import Optional, Union, Tuple, List, Callable, Dict
from IPython.display import display
from tqdm.notebook import tqdm


def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)):
    """
    """
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    # font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf", font_size)
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y ), font, 1, text_color, 2)
    return img


def view_images(images, num_rows=1, offset_ratio=0.02, save=False, id=0, save_path=None, save_dir=None):
    """
    """
    if type(images) is list:
        num_empCty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    if save:
        if save_path is None:
            save_path = os.path.join(save_dir or ".", f"{id}.png")
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        pil_img.save(save_path)
    else:
        display(pil_img)
    
    


def diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False):
    """
    """
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    latents = controller.step_callback(latents)
    return latents


def latent2image(vae, latents,id=0):
    """
    """
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).astype(np.uint8)
    # pil_images = [Image.fromarray(image) for image in images]
    # for num, im in enumerate(pil_images):
    #     im.save("outputs/images/{}.png".format(id))
    return images


def init_latent(latent, model, height, width, generator, batch_size):
    """
    """
    if latent is None:
        latent = torch.randn(
            (1, model.unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
    latents = latent.expand(batch_size,  model.unet.in_channels, height // 8, width // 8).to(model.device)
    return latent, latents


@torch.no_grad()
def text2image_ldm(
    model,
    prompt:  List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
):
    """
    """
    register_attention_control(model, controller)
    height = width = 256
    batch_size = len(prompt)
    
    uncond_input = model.tokenizer([""] * batch_size, padding="max_length", max_length=77, return_tensors="pt")
    uncond_embeddings = model.bert(uncond_input.input_ids.to(model.device))[0]
    
    text_input = model.tokenizer(prompt, padding="max_length", max_length=77, return_tensors="pt")
    text_embeddings = model.bert(text_input.input_ids.to(model.device))[0]
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    context = torch.cat([uncond_embeddings, text_embeddings])
    
    model.scheduler.set_timesteps(num_inference_steps)
    for t in tqdm(model.scheduler.timesteps):
        latents = diffusion_step(model, controller, latents, context, t, guidance_scale)
    
    image = latent2image(model.vqvae, latents)
   
    return image, latent


@torch.no_grad()
def text2image_ldm_stable(
    model,
    prompt: List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    low_resource: bool = False,
):
    """
    """
    register_attention_control(model, controller)
    height = width = 512
    batch_size = len(prompt)

    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    
    context = [uncond_embeddings, text_embeddings]
    if not low_resource:
        context = torch.cat(context)
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    # [1,4,64,64],[2,4,64,64]
    # set timesteps
    model.scheduler.set_timesteps(num_inference_steps)
    z_i = []
    for t in tqdm(model.scheduler.timesteps):
        latents = diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource)
        z_i.append(latents)
        
    # image = latent2image(model.vae, latents)
    z_i = torch.stack(z_i,dim=1)
    return z_i

@torch.no_grad()
def text2image_ldm_stable_v2(
    model,
    prompt: List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    low_resource: bool = False,
    id: int = 0
):
    """
    """
    # register_attention_control_v2(model, controller) # controller.num_attn = 32
    register_attention_control(model, controller)
    
    height = width = 512
    batch_size = len(prompt)

    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    
    context = [uncond_embeddings, text_embeddings]
    if not low_resource:
        context = torch.cat(context)
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    # [1,4,64,64],[2,4,64,64]
    # set timesteps
    model.scheduler.set_timesteps(num_inference_steps)
    for t in model.scheduler.timesteps:
        latents = diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource)
        # latents = scale_tensor(latents)
    
    image = latent2image(model.vae, latents,id)
  
    return image, latents

def scale_tensor(x):
    """
    """
    min_value = x.min()
    max_value = x.max()

    scale = 2 / (max_value - min_value)

    scaled_x = scale * (x - min_value) - 1
    
    return scaled_x

@torch.no_grad()
def text2image_ldm_stable_v3(
    model,
    prompt: List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    low_resource: bool = False,
    lora: bool = True,
    s: float = 1.0
):
    """
    """
    if lora:
        register_attention_control_v2(model, controller)
    else:
        register_attention_control(model, controller,s)
    
    height = width = 512
    batch_size = len(prompt)

    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    
    context = [uncond_embeddings, text_embeddings]
    if not low_resource:
        context = torch.cat(context)
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    # [1,4,64,64],[2,4,64,64]
    # set timesteps
    model.scheduler.set_timesteps(num_inference_steps)
    for t in model.scheduler.timesteps:
        latents = diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource)
    
    image = latent2image(model.vae, latents,id)

    return latents

_original_forwards = {}
_registered_models = set()


def clear_attention_control_cache():
    """
    
    """
    global _original_forwards, _registered_models
    _original_forwards.clear()
    _registered_models.clear()
    print("✓ Attention control cache cleared")


def register_attention_control_multi_scale(model, controller, scaling_factors: list):
    """
    
    """
    global _original_forwards
    
    def ca_forward_multi_scale(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
            """
            """
            x = hidden_states
            context = encoder_hidden_states
            is_cross = context is not None
            context = context if is_cross else x

            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )
            attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)

            if encoder_hidden_states is None:
                encoder_hidden_states_for_kv = hidden_states
            elif self.norm_cross:
                encoder_hidden_states_for_kv = self.norm_encoder_hidden_states(encoder_hidden_states)
            else:
                encoder_hidden_states_for_kv = encoder_hidden_states

            is_lora_active = hasattr(self, "processor") and hasattr(self.processor, "to_q_lora")
            
            if is_lora_active:
                lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs else 1.0
                query = self.to_q(hidden_states) + lora_scale * self.processor.to_q_lora(hidden_states)
                key = self.to_k(encoder_hidden_states_for_kv) + lora_scale * self.processor.to_k_lora(encoder_hidden_states_for_kv)
                value = self.to_v(encoder_hidden_states_for_kv) + lora_scale * self.processor.to_v_lora(encoder_hidden_states_for_kv)
            else:
                query = self.to_q(hidden_states)
                key = self.to_k(encoder_hidden_states_for_kv)
                value = self.to_v(encoder_hidden_states_for_kv)

            query = self.head_to_batch_dim(query)
            key = self.head_to_batch_dim(key)
            value = self.head_to_batch_dim(value)

            attention_scores_raw = torch.bmm(query, key.transpose(-1, -2))
            
            if is_cross:
                multi_scale_scores = {}
                for s in scaling_factors:
                    attention_scores_scaled = attention_scores_raw * s * self.scale
                    if attention_mask is not None:
                        attention_scores_scaled = attention_scores_scaled + attention_mask
                    multi_scale_scores[s] = attention_scores_scaled.detach().to(dtype=torch.float16).cpu()
                
                controller(
                    {
                        'multi_scale_scores': multi_scale_scores,  # dict: {scale: attention_scores}
                    },
                    is_cross,
                    place_in_unet
                )
            
            attention_scores = attention_scores_raw * self.scale
            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask
            
            attention_probs = attention_scores.softmax(dim=-1)
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = self.batch_to_head_dim(hidden_states)

            if is_lora_active:
                lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs else 1.0
                hidden_states = self.to_out[0](hidden_states) + lora_scale * self.processor.to_out_lora(hidden_states)
            else:
                hidden_states = self.to_out[0](hidden_states)
            
            # Dropout
            hidden_states = self.to_out[1](hidden_states)
            
            return hidden_states

        return forward

    class DummyController:
        def __call__(self, *args):
            return args[0]
        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'Attention':
            net_id = id(net_)
            if net_id not in _original_forwards:
                _original_forwards[net_id] = net_.forward
            net_.forward = ca_forward_multi_scale(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    controller.num_att_layers = cross_att_count


def unregister_attention_control(model):
    """
    
    
    
    """
    global _original_forwards, _registered_models
    
    restored_count = 0
    
    def restore_recr(net_):
        nonlocal restored_count
        if net_.__class__.__name__ == 'Attention':
            net_id = id(net_)
            if net_id in _original_forwards:
                net_.forward = _original_forwards[net_id]
                del _original_forwards[net_id]
                restored_count += 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                restore_recr(net__)
    
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        restore_recr(net[1])
    
    model_id = id(model)
    if model_id in _registered_models:
        _registered_models.remove(model_id)
    
    # if restored_count > 0:
    #     print(f"✓ Unregistered attention control: restored {restored_count} attention layers")


def register_attention_control_pv(model, controller, s: float = 1.0, collect_pv: bool = True, collect_attn: bool = False):
    """
    
    
    
        attention_scores = Q @ K^T / sqrt(d_k)
        attention_probs = softmax(attention_scores * s)
        attention_output = attention_probs @ V           # PV
    """
    global _original_forwards, _registered_models
    
    model_id = id(model)
    if model_id in _registered_models:
        pass
    
    _registered_models.add(model_id)
    
    def ca_forward_pv(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
            """
            """
            x = hidden_states
            context = encoder_hidden_states
            is_cross = context is not None
            context = context if is_cross else x

            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )
            attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)

            if encoder_hidden_states is None:
                encoder_hidden_states_for_kv = hidden_states
            elif self.norm_cross:
                encoder_hidden_states_for_kv = self.norm_encoder_hidden_states(encoder_hidden_states)
            else:
                encoder_hidden_states_for_kv = encoder_hidden_states

            is_lora_active = hasattr(self, "processor") and hasattr(self.processor, "to_q_lora")
            
            if is_lora_active:
                lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs else 1.0
                query = self.to_q(hidden_states) + lora_scale * self.processor.to_q_lora(hidden_states)
                key = self.to_k(encoder_hidden_states_for_kv) + lora_scale * self.processor.to_k_lora(encoder_hidden_states_for_kv)
                value = self.to_v(encoder_hidden_states_for_kv) + lora_scale * self.processor.to_v_lora(encoder_hidden_states_for_kv)
            else:
                query = self.to_q(hidden_states)
                key = self.to_k(encoder_hidden_states_for_kv)
                value = self.to_v(encoder_hidden_states_for_kv)

            query = self.head_to_batch_dim(query)
            key = self.head_to_batch_dim(key)
            value = self.head_to_batch_dim(value)

            attention_scores_raw = torch.bmm(query, key.transpose(-1, -2))
            
            attention_scores = attention_scores_raw * self.scale
            
            attention_scores_scaled = attention_scores * s
            
            if attention_mask is not None:
                attention_scores_scaled = attention_scores_scaled + attention_mask
            
            attention_probs = attention_scores_scaled.softmax(dim=-1)
            
            attention_output = torch.bmm(attention_probs, value)
            
            if is_cross:
                collected_data = {}
                if collect_pv:
                    collected_data['pv'] = attention_output.detach()  # (batch*heads, seq_q, head_dim)
                if collect_attn:
                    collected_data['attn'] = attention_probs.detach()  # (batch*heads, seq_q, seq_k)
                collected_data['place_in_unet'] = place_in_unet
                
                controller(collected_data, is_cross, place_in_unet)
            
            hidden_states = self.batch_to_head_dim(attention_output)

            if is_lora_active:
                lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs else 1.0
                hidden_states = self.to_out[0](hidden_states) + lora_scale * self.processor.to_out_lora(hidden_states)
            else:
                hidden_states = self.to_out[0](hidden_states)
            
            # Dropout
            hidden_states = self.to_out[1](hidden_states)
            
            return hidden_states

        return forward

    class DummyController:
        def __call__(self, *args):
            return args[0]
        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'Attention':
            net_id = id(net_)
            if net_id not in _original_forwards:
                _original_forwards[net_id] = net_.forward
            net_.forward = ca_forward_pv(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    controller.num_att_layers = cross_att_count


def register_attention_control(model, controller, s: float = 1.0):
    """
    
    
    
    """
    global _original_forwards, _registered_models
    
    model_id = id(model)
    if model_id in _registered_models:
        print(f"⚠ Warning: Model already registered. Re-registering will override previous registration.")
    
    _registered_models.add(model_id)
    def ca_forward(self, place_in_unet):
        """
        """
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
            """
            """
            x = hidden_states
            context = encoder_hidden_states
            is_cross = context is not None
            context = context if is_cross else x

            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )
            attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)

            if encoder_hidden_states is None:
                encoder_hidden_states_for_kv = hidden_states
            elif self.norm_cross:
                encoder_hidden_states_for_kv = self.norm_encoder_hidden_states(encoder_hidden_states)
            else:
                encoder_hidden_states_for_kv = encoder_hidden_states

            is_lora_active = hasattr(self, "processor") and hasattr(self.processor, "to_q_lora")
            
            if is_lora_active:
                lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs else 1.0
                query = self.to_q(hidden_states) + lora_scale * self.processor.to_q_lora(hidden_states)
                key = self.to_k(encoder_hidden_states_for_kv) + lora_scale * self.processor.to_k_lora(encoder_hidden_states_for_kv)
                value = self.to_v(encoder_hidden_states_for_kv) + lora_scale * self.processor.to_v_lora(encoder_hidden_states_for_kv)
            else:
                query = self.to_q(hidden_states)
                key = self.to_k(encoder_hidden_states_for_kv)
                value = self.to_v(encoder_hidden_states_for_kv)

            query = self.head_to_batch_dim(query)
            key = self.head_to_batch_dim(key)
            value = self.head_to_batch_dim(value)

            attention_scores_raw = torch.bmm(query, key.transpose(-1, -2))
            
            attention_scores_scaled = attention_scores_raw * s
            
            attention_scores = attention_scores_scaled * self.scale
            
            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask
            
            attention_probs = attention_scores.softmax(dim=-1)
            
            controller(
                {
                    'attention_scores': attention_scores,
                    'attention_probs': attention_probs
                },
                is_cross,
                place_in_unet
            )
            
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = self.batch_to_head_dim(hidden_states)

            if is_lora_active:
                lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs else 1.0
                hidden_states = self.to_out[0](hidden_states) + lora_scale * self.processor.to_out_lora(hidden_states)
            else:
                hidden_states = self.to_out[0](hidden_states)
            
            # Dropout
            hidden_states = self.to_out[1](hidden_states)
            
            return hidden_states

        return forward

    class DummyController:
        """
        """

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        """
        """
        if net_.__class__.__name__ == 'Attention':
            net_id = id(net_)
            if net_id not in _original_forwards:
                _original_forwards[net_id] = net_.forward
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    controller.num_att_layers = cross_att_count
    
    # print(f"✓ Registered attention control: {cross_att_count} attention layers (scale={s})")

    
def register_attention_control_v2(model, controller):
    """
    """
    def ca_forward(self, place_in_unet):
        """
        """
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
            """
            """
            scale = 1
            x = hidden_states
            context = encoder_hidden_states
            is_cross = context is not None
            context = context if is_cross else x

            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )
            attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)

            query = self.to_q(hidden_states) + scale * self.processor.to_q_lora(hidden_states)
            query = self.head_to_batch_dim(query)

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif self.norm_cross:
                encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

            key = self.to_k(encoder_hidden_states) + scale * self.processor.to_k_lora(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states) + scale * self.processor.to_v_lora(encoder_hidden_states)

            key = self.head_to_batch_dim(key)
            value = self.head_to_batch_dim(value)

            attention_probs = self.get_attention_scores(query, key, attention_mask)
            controller(attention_probs, is_cross, place_in_unet)
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = self.batch_to_head_dim(hidden_states)

            # linear proj
            hidden_states = self.to_out[0](hidden_states) + scale * self.processor.to_out_lora(hidden_states)
            # dropout
            hidden_states = self.to_out[1](hidden_states)
            
            return hidden_states

        return forward

    class DummyController:
        """
        """

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        """
        """
        if net_.__class__.__name__ == 'Attention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    controller.num_att_layers = cross_att_count
    
    print(f"✓ Registered attention control v2 (LoRA): {cross_att_count} attention layers")

    
def get_word_inds(text: str, word_place: int, tokenizer):
    """
    """
    split_text = text.split(" ")
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)


def update_alpha_time_word(alpha, bounds: Union[float, Tuple[float, float]], prompt_ind: int,
                           word_inds: Optional[torch.Tensor]=None):
    """
    """
    if type(bounds) is float:
        bounds = 0, bounds
    start, end = int(bounds[0] * alpha.shape[0]), int(bounds[1] * alpha.shape[0])
    if word_inds is None:
        word_inds = torch.arange(alpha.shape[2])
    alpha[: start, prompt_ind, word_inds] = 0
    alpha[start: end, prompt_ind, word_inds] = 1
    alpha[end:, prompt_ind, word_inds] = 0
    return alpha


def get_time_words_attention_alpha(prompts, num_steps,
                                   cross_replace_steps: Union[float, Dict[str, Tuple[float, float]]],
                                   tokenizer, max_num_words=77):
    """
    """
    if type(cross_replace_steps) is not dict:
        cross_replace_steps = {"default_": cross_replace_steps}
    if "default_" not in cross_replace_steps:
        cross_replace_steps["default_"] = (0., 1.)
    alpha_time_words = torch.zeros(num_steps + 1, len(prompts) - 1, max_num_words)
    for i in range(len(prompts) - 1):
        alpha_time_words = update_alpha_time_word(alpha_time_words, cross_replace_steps["default_"],
                                                  i)
    for key, item in cross_replace_steps.items():
        if key != "default_":
             inds = [get_word_inds(prompts[i], key, tokenizer) for i in range(1, len(prompts))]
             for i, ind in enumerate(inds):
                 if len(ind) > 0:
                    alpha_time_words = update_alpha_time_word(alpha_time_words, item, i, ind)
    alpha_time_words = alpha_time_words.reshape(num_steps + 1, len(prompts) - 1, 1, 1, max_num_words)
    return alpha_time_words

# ============================================================================
# SET paper terminology aliases
# ============================================================================
ATTACK_CONFIGS = BACKDOOR_CONFIGS
CrossAttentionScalingHook = SyncCompareHook
CSRDFeatureExtractor = CounterfactualFingerprintGenerator
ScalingResponseFeatureExtractor = CounterfactualFingerprintGenerator
BenignResponseSpaceDetector = EncoderDetector


def load_backdoored_t2i_model(pipeline, config, device):
    return load_poisoned_model(pipeline, config, device)


def extract_response_offset_features(prompts, config, scaling_factors, seed, base_model_path, output_dir, device, batch_size):
    return generate_fingerprints_single(
        prompts,
        config,
        scaling_factors,
        seed,
        base_model_path,
        output_dir,
        device,
        batch_size,
    )


def plot_csrd_response_heatmap(fingerprints_benign, fingerprints_backdoor, scaling_factors, output_path):
    """
    """
    try:
        if fingerprints_benign is None or len(fingerprints_benign) == 0:
            raise ValueError("fingerprints_benign is empty")
        if fingerprints_backdoor is None or len(fingerprints_backdoor) == 0:
            raise ValueError("fingerprints_backdoor is empty")

        avg_fp_benign = np.mean(fingerprints_benign, axis=0)
        avg_fp_backdoor = np.mean(fingerprints_backdoor, axis=0)
        
        num_scales = len(scaling_factors)
        total_dim = len(avg_fp_benign)
        
        dim_per_scale = total_dim // num_scales
        
        candidate_layers = [16, 32]
        num_layers = None
        num_steps = None
        for cand in candidate_layers:
            if dim_per_scale % cand == 0:
                num_layers = cand
                num_steps = dim_per_scale // cand
                break

        if num_layers is None:
            num_steps = 1
            num_layers = dim_per_scale

        print(f"[Difference Heatmap] total_dim={total_dim}, num_scales={num_scales}, dim_per_scale={dim_per_scale}, inferred_steps={num_steps}, inferred_layers={num_layers}")
        
        matrix_3d_benign = avg_fp_benign.reshape(num_scales, num_steps, num_layers)
        matrix_3d_backdoor = avg_fp_backdoor.reshape(num_scales, num_steps, num_layers)
        
        mse_matrix_benign = np.mean(matrix_3d_benign, axis=1)
        mse_matrix_backdoor = np.mean(matrix_3d_backdoor, axis=1)
        
        mse_diff = mse_matrix_benign - mse_matrix_backdoor
        
        plt.figure(figsize=(14, 8))
        im = plt.imshow(mse_diff, aspect='auto', cmap='RdBu_r', 
                       vmin=-np.max(np.abs(mse_diff)), vmax=np.max(np.abs(mse_diff)))
        plt.colorbar(im, label='MSE Difference (Benign - Backdoor)')
        
        plt.yticks(range(num_scales), scaling_factors)
        plt.xticks(range(num_layers), [f'L{i}' for i in range(num_layers)])
        
        plt.title("MSE Difference: Benign - Backdoor\n(Positive=Benign larger, Negative=Backdoor larger)")
        plt.xlabel("UNet Cross-Attention Layers")
        plt.ylabel("Scaling Factor")
        
        for i in range(num_scales):
            for j in range(num_layers):
                val = mse_diff[i, j]
                color = "white" if abs(val) > np.max(np.abs(mse_diff)) * 0.5 else "black"
                plt.text(j, i, f"{val:.1e}", 
                         ha="center", va="center", color=color, fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"✓ MSE Difference Heatmap saved to {output_path}")

        txt_path = output_path.replace(".png", ".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("MSE Difference Breakdown (Benign - Backdoor)\n")
            f.write("Positive values: Benign MSE > Backdoor MSE\n")
            f.write("Negative values: Backdoor MSE > Benign MSE\n")
            f.write("-" * 50 + "\n")
            header = "Scale".ljust(10) + "".join([f"L{i}".rjust(12) for i in range(num_layers)])
            f.write(header + "\n")
            for i, scale in enumerate(scaling_factors):
                row = f"{scale}".ljust(10) + "".join([f"{mse_diff[i, j]:.4e}".rjust(12) for j in range(num_layers)])
                f.write(row + "\n")
            
            f.write("\n" + "=" * 50 + "\n")
            f.write("Statistics:\n")
            f.write(f"Max Difference (Benign > Backdoor): {np.max(mse_diff):.4e}\n")
            f.write(f"Min Difference (Backdoor > Benign): {np.min(mse_diff):.4e}\n")
            f.write(f"Mean Absolute Difference: {np.mean(np.abs(mse_diff)):.4e}\n")
            
            max_idx = np.unravel_index(np.argmax(mse_diff), mse_diff.shape)
            min_idx = np.unravel_index(np.argmin(mse_diff), mse_diff.shape)
            f.write(f"\nLargest Positive Diff: Scale={scaling_factors[max_idx[0]]}, Layer={max_idx[1]}, Value={mse_diff[max_idx]:.4e}\n")
            f.write(f"Largest Negative Diff: Scale={scaling_factors[min_idx[0]]}, Layer={min_idx[1]}, Value={mse_diff[min_idx]:.4e}\n")
            
        print(f"✓ MSE Difference numerical breakdown saved to {txt_path}")

    except Exception as e:
        print(f"Warning: Failed to plot MSE difference heatmap: {e}")
        traceback.print_exc()


