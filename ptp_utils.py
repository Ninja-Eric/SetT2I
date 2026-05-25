import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from safetensors.torch import load_file, save_file
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import CLIPTextModel


normal_model_id = "CompVis/stable-diffusion-v1-4"
scale_values = [0.2, 0.3, 7.0, 10.0, 20.0]
target_query_tokens = 256
kept_unet_parts = {"down", "up"}
probe_steps = 5
sample_steps = 50
guidance_weight = 7.5


@dataclass
class AttackConfig:
    name: str
    model_path: str | None
    prompt_file: str
    model_kind: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(gpu: str | None) -> str:
    if gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def read_prompts(path: str, max_items: int | None = None) -> list[str]:
    with open(path, "r", encoding="utf-8") as handle:
        prompts = [line.strip() for line in handle if line.strip()]
    return prompts[:max_items] if max_items else prompts


def ensure_parent(path: str) -> None:
    parent = Path(path).parent
    if str(parent):
        parent.mkdir(parents=True, exist_ok=True)


def detector_meta_path(model_path: str) -> str:
    return str(Path(model_path).with_name("set_detector_meta.json"))


def default_attack_configs(args) -> dict[str, AttackConfig]:
    data_dir = Path("data")
    model_dir = Path("models")
    prompt_file = args.prompt_file
    model_path = args.backdoored_model_path
    defaults = {
        "rickrolling": ("text_encoder", str(model_dir / "rickrolling"), str(data_dir / "rickrolling.txt")),
        "twt": ("text_encoder", str(model_dir / "twt"), str(data_dir / "twt.txt")),
        "villan_mignneko": ("lora", str(model_dir / "villan_mignneko.safetensors"), str(data_dir / "villan_mignneko.txt")),
        "villan_github": ("lora", str(model_dir / "villan_github"), str(data_dir / "villan_github.txt")),
        "villan_anonymous": ("lora", str(model_dir / "villan_anonymous"), str(data_dir / "villan_anonymous.txt")),
        "eviledit": ("unet_state_dict", str(model_dir / "eviledit.pt"), str(data_dir / "eviledit.txt")),
        "pixel": ("unet", str(model_dir / "pixel"), str(data_dir / "pixel.txt")),
        "personal": ("textual_inversion", str(model_dir / "personal"), str(data_dir / "personal.txt")),
        "clean": ("clean", None, str(data_dir / "clean.txt")),
    }
    configs = {}
    for name, (kind, path, default_prompt) in defaults.items():
        configs[name] = AttackConfig(
            name=name,
            model_path=model_path if model_path and name == args.attack_method else path,
            prompt_file=prompt_file if prompt_file and name == args.attack_method else default_prompt,
            model_kind=kind,
        )
    return configs


def load_pipeline(base_model_path: str, config: AttackConfig, device: str) -> StableDiffusionPipeline:
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    return load_attack_weights(pipe, config, device)


def load_attack_weights(pipe: StableDiffusionPipeline, config: AttackConfig, device: str) -> StableDiffusionPipeline:
    path = config.model_path
    if config.model_kind == "clean" or path is None:
        return pipe
    if config.model_kind == "text_encoder":
        pipe.text_encoder = CLIPTextModel.from_pretrained(path, local_files_only=True).to(device).to(torch.float16)
        return pipe
    if config.model_kind == "lora":
        if os.path.isdir(path):
            files = os.listdir(path)
            safe_files = [item for item in files if item.endswith(".safetensors")]
            bin_files = [item for item in files if item.endswith(".bin")]
            if safe_files:
                target = os.path.join(path, safe_files[0])
                pipe.load_lora_weights(pretrained_model_name_or_path_or_dict=target)
                pipe.unet.load_attn_procs(target)
            elif bin_files:
                pipe.unet.load_attn_procs(path)
        else:
            pipe.load_lora_weights(pretrained_model_name_or_path_or_dict=path)
            pipe.unet.load_attn_procs(path)
        pipe.fuse_lora(lora_scale=1.0)
        return pipe
    if config.model_kind == "unet_state_dict":
        pipe.unet.load_state_dict(torch.load(path, map_location=device))
        pipe.unet.to(device).to(torch.float16)
        return pipe
    if config.model_kind == "unet":
        pipe.unet = UNet2DConditionModel.from_pretrained(path, torch_dtype=torch.float16).to(device)
        return pipe
    if config.model_kind == "textual_inversion":
        embed_path = Path(path) / "learned_embeds.bin"
        if not embed_path.exists():
            step_files = sorted(Path(path).glob("learned_embeds-step-*.bin"))
            if not step_files:
                raise FileNotFoundError(f"No learned embedding file found in {path}")
            embed_path = step_files[-1]
        embeds = torch.load(str(embed_path), weights_only=False, map_location=device)
        if "string_to_param" in embeds:
            token = next(iter(embeds["string_to_param"]))
            learned_embed = embeds["string_to_param"][token].squeeze()
        else:
            token = next(iter(embeds))
            learned_embed = embeds[token].squeeze()
        pipe.tokenizer.add_tokens(token)
        token_id = pipe.tokenizer.convert_tokens_to_ids(token)
        pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))
        weight = pipe.text_encoder.get_input_embeddings().weight
        weight.data[token_id] = learned_embed.to(dtype=weight.dtype, device=weight.device)
        pipe.text_encoder = pipe.text_encoder.to(device).to(torch.float16)
        return pipe
    raise ValueError(f"Unknown attack model kind: {config.model_kind}")


class AttentionProbeHook:
    def __init__(self, unet, scale: float):
        self.unet = unet
        self.scale = scale
        self.base_outputs = {}
        self.step_offsets = {}
        self.counter = 0
        self.collect_base = False
        self.original_forwards = {}
        self.modules = []
        self._patch()

    def _patch(self) -> None:
        for name, child in self.unet.named_children():
            if "down" in name:
                part = "down"
            elif "mid" in name:
                part = "mid"
            elif "up" in name:
                part = "up"
            else:
                continue
            self._walk(child, part)
        for module, is_cross, part in self.modules:
            self.original_forwards[id(module)] = module.forward
            module.forward = self._forward(module, is_cross, part)

    def _walk(self, module, part: str) -> None:
        if module.__class__.__name__ == "Attention":
            is_cross = module.to_q.in_features != module.to_k.in_features
            self.modules.append((module, is_cross, part))
        for child in module.children():
            self._walk(child, part)

    def _forward(self, attn, is_cross: bool, part: str):
        hook = self
        original = self.original_forwards[id(attn)]

        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
            if not is_cross:
                if hook.collect_base:
                    return original(hidden_states, encoder_hidden_states, attention_mask, **kwargs)
                batch_size = hidden_states.shape[0]
                seq_len = hidden_states.shape[1]
                q = attn.to_q(hidden_states)
                k = attn.to_k(hidden_states)
                v = attn.to_v(hidden_states)
                q = attn.head_to_batch_dim(q)
                k = attn.head_to_batch_dim(k)
                v = attn.head_to_batch_dim(v)
                mask = attn.prepare_attention_mask(attention_mask, seq_len, batch_size)
                scores = torch.bmm(q, k.transpose(-1, -2)) * attn.scale
                if mask is not None:
                    scores = scores + mask
                probs = (scores * hook.scale).softmax(dim=-1)
                out = attn.batch_to_head_dim(torch.bmm(probs, v))
                out = attn.to_out[0](out)
                return attn.to_out[1](out)
            hook.counter += 1
            layer_id = hook.counter
            query_tokens = hidden_states.shape[1]
            keep = part in kept_unet_parts and query_tokens == target_query_tokens
            ehs = attn.norm_encoder_hidden_states(encoder_hidden_states) if getattr(attn, "norm_cross", False) else encoder_hidden_states
            batch_size, seq_len, _ = encoder_hidden_states.shape
            mask = attn.prepare_attention_mask(attention_mask, seq_len, batch_size)
            q = attn.to_q(hidden_states)
            k = attn.to_k(ehs)
            v = attn.to_v(ehs)
            q = attn.head_to_batch_dim(q)
            k = attn.head_to_batch_dim(k)
            v = attn.head_to_batch_dim(v)
            scores = torch.bmm(q, k.transpose(-1, -2)) * attn.scale
            if mask is not None:
                scores = scores + mask
            probs = (scores * hook.scale).softmax(dim=-1) if keep and not hook.collect_base else scores.softmax(dim=-1)
            pv = torch.bmm(probs, v)
            if keep:
                heads = attn.heads
                half_batch = hidden_states.shape[0] // 2
                cond_start = half_batch * heads
                pv_cond = pv[cond_start:].reshape(half_batch, heads, query_tokens, pv.shape[-1])
                key = (part, layer_id)
                if hook.collect_base:
                    hook.base_outputs[key] = pv_cond.detach()
                elif key in hook.base_outputs:
                    base = hook.base_outputs[key]
                    mse = ((pv_cond.float() - base.float()) ** 2).mean(dim=-1).mean(dim=1)
                    hook.step_offsets[key] = mse.cpu().numpy()
            out = attn.batch_to_head_dim(pv)
            out = attn.to_out[0](out)
            return attn.to_out[1](out)

        return forward

    def reset(self) -> None:
        self.counter = 0
        self.step_offsets = {}

    def clear_base(self) -> None:
        self.base_outputs = {}

    def offsets(self) -> dict:
        return dict(self.step_offsets)

    def restore(self) -> None:
        for module, _, _ in self.modules:
            original = self.original_forwards.get(id(module))
            if original is not None:
                module.forward = original
        self.original_forwards.clear()
        self.modules.clear()


class ScalingProbe:
    def __init__(self, pipe: StableDiffusionPipeline, device: str, scales: list[float] | None = None):
        self.pipe = pipe
        self.device = device
        self.scales = scales or scale_values
        self.layer_count, self.layer_parts = self._find_layers()

    def _find_layers(self) -> tuple[int, list[str]]:
        hook = AttentionProbeHook(self.pipe.unet, 1.0)
        try:
            generator = torch.Generator(device=self.device).manual_seed(0)
            channels = self.pipe.unet.config.in_channels
            latents = torch.randn((1, channels, 64, 64), generator=generator, device=self.device, dtype=torch.float16)
            tokens = self.pipe.tokenizer([""], padding="max_length", max_length=self.pipe.tokenizer.model_max_length, return_tensors="pt")
            embeds = self.pipe.text_encoder(tokens.input_ids.to(self.device))[0]
            context = torch.cat([embeds, embeds], dim=0)
            self.pipe.scheduler.set_timesteps(sample_steps, device=self.device)
            latents = latents * self.pipe.scheduler.init_noise_sigma
            t0 = self.pipe.scheduler.timesteps[0]
            hook.collect_base = True
            hook.reset()
            latent_input = self.pipe.scheduler.scale_model_input(torch.cat([latents, latents], dim=0), t0)
            with torch.no_grad():
                self.pipe.unet(latent_input, t0, encoder_hidden_states=context).sample
            keys = sorted(hook.base_outputs.keys(), key=lambda item: item[1])
            return len(keys), [part for part, _ in keys]
        finally:
            hook.restore()

    @torch.no_grad()
    def collect(self, prompts: list[str], seed: int) -> np.ndarray:
        if isinstance(prompts, str):
            prompts = [prompts]
        batch = len(prompts)
        channels = self.pipe.unet.config.in_channels
        dtype = torch.float16
        uncond = self.pipe.tokenizer([""] * batch, padding="max_length", max_length=self.pipe.tokenizer.model_max_length, return_tensors="pt")
        cond = self.pipe.tokenizer(prompts, padding="max_length", max_length=self.pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        uncond_emb = self.pipe.text_encoder(uncond.input_ids.to(self.device))[0]
        cond_emb = self.pipe.text_encoder(cond.input_ids.to(self.device))[0]
        context = torch.cat([uncond_emb, cond_emb], dim=0)
        starts = []
        for index in range(batch):
            generator = torch.Generator(device=self.device).manual_seed(seed + index)
            starts.append(torch.randn((1, channels, 64, 64), generator=generator, device=self.device, dtype=dtype))
        latents_start = torch.cat(starts, dim=0)
        self.pipe.scheduler.set_timesteps(sample_steps, device=self.device)
        latents_start = latents_start * self.pipe.scheduler.init_noise_sigma
        timesteps = self.pipe.scheduler.timesteps[:probe_steps]
        base_by_step = {}
        base_hook = AttentionProbeHook(self.pipe.unet, 1.0)
        try:
            latents = latents_start.clone()
            for step_index, timestep in enumerate(timesteps):
                base_hook.collect_base = True
                base_hook.reset()
                base_hook.clear_base()
                latent_input = self.pipe.scheduler.scale_model_input(torch.cat([latents, latents], dim=0), timestep)
                noise = self.pipe.unet(latent_input, timestep, encoder_hidden_states=context).sample
                noise_uncond, noise_cond = noise.chunk(2)
                guided = noise_uncond + guidance_weight * (noise_cond - noise_uncond)
                latents = self.pipe.scheduler.step(guided, timestep, latents).prev_sample
                base_by_step[step_index] = dict(base_hook.base_outputs)
        finally:
            base_hook.restore()
        scale_features = []
        keys = None
        for scale in self.scales:
            latents = latents_start.clone()
            hook = AttentionProbeHook(self.pipe.unet, scale)
            step_features = []
            try:
                for step_index, timestep in enumerate(timesteps):
                    hook.base_outputs = base_by_step[step_index]
                    hook.collect_base = False
                    hook.reset()
                    latent_input = self.pipe.scheduler.scale_model_input(torch.cat([latents, latents], dim=0), timestep)
                    noise = self.pipe.unet(latent_input, timestep, encoder_hidden_states=context).sample
                    noise_uncond, noise_cond = noise.chunk(2)
                    guided = noise_uncond + guidance_weight * (noise_cond - noise_uncond)
                    latents = self.pipe.scheduler.step(guided, timestep, latents).prev_sample
                    offsets = hook.offsets()
                    if keys is None:
                        keys = sorted(offsets.keys(), key=lambda item: item[1])
                    step_array = np.zeros((batch, len(keys)), dtype=np.float32)
                    for layer_index, key in enumerate(keys):
                        if key in offsets:
                            step_array[:, layer_index] = offsets[key].mean(axis=-1)
                    step_features.append(step_array)
            finally:
                hook.restore()
            scale_features.append(np.concatenate(step_features, axis=1))
        return np.concatenate(scale_features, axis=1)


def collect_response_features(pipe: StableDiffusionPipeline, prompts: list[str], seed: int, batch_size: int, device: str) -> tuple[np.ndarray, list[str]]:
    probe = ScalingProbe(pipe, device=device, scales=scale_values)
    chunks = []
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        chunks.append(probe.collect(batch_prompts, seed + start))
        print(f"processed {min(start + batch_size, len(prompts))}/{len(prompts)}")
    return np.concatenate(chunks, axis=0), probe.layer_parts


class ResponseEncoder(nn.Module):
    def __init__(self, num_layers: int, num_scales: int, num_steps: int, hidden_dim: int = 64, embedding_dim: int = 32):
        super().__init__()
        token_dim = num_scales * num_steps
        self.num_layers = num_layers
        self.num_scales = num_scales
        self.num_steps = num_steps
        self.phi = nn.Sequential(nn.Linear(token_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.rho = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, embedding_dim, bias=False))

    def forward(self, x):
        batch = x.shape[0]
        x = x.view(batch, self.num_scales, self.num_steps, self.num_layers)
        x = x.permute(0, 3, 1, 2).reshape(batch, self.num_layers, self.num_scales * self.num_steps)
        h = self.phi(x)
        h_mean = h.mean(dim=1)
        h_std = torch.sqrt(torch.var(h, dim=1) + 1e-6)
        return self.rho(torch.cat([h_mean, h_std], dim=-1))


class ResponseSpace:
    def __init__(
        self,
        num_layers: int,
        num_scales: int = len(scale_values),
        num_steps: int = probe_steps,
        embedding_dim: int = 32,
        nu: float = 0.05,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        epochs: int = 100,
        device: str = "cuda",
        warmup_epochs: int = 5,
    ):
        self.num_layers = num_layers
        self.num_scales = num_scales
        self.num_steps = num_steps
        self.embedding_dim = embedding_dim
        self.nu = nu
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.warmup_epochs = warmup_epochs
        self.encoder = ResponseEncoder(num_layers, num_scales, num_steps, embedding_dim=embedding_dim).to(device)
        self.center = None
        self.radius = 0.0
        self.mean = None
        self.std = None

    def fit(self, features: np.ndarray) -> dict[str, list[float]]:
        self.mean = features.mean(axis=0)
        self.std = features.std(axis=0)
        self.std[self.std < 1e-8] = 1.0
        x = torch.tensor((features - self.mean) / self.std, dtype=torch.float32, device=self.device)
        data = TensorDataset(x)
        loader = DataLoader(data, batch_size=self.batch_size, shuffle=True)
        self.encoder.eval()
        with torch.no_grad():
            z_items = []
            for (batch_x,) in DataLoader(data, batch_size=self.batch_size, shuffle=False):
                z_items.append(self.encoder(batch_x))
            z = torch.cat(z_items, dim=0)
            self.center = z.mean(dim=0).detach()
            dist = torch.sum((z - self.center) ** 2, dim=1)
        radius = torch.tensor(0.1, device=self.device, requires_grad=True)
        optimizer = optim.Adam(
            [{"params": self.encoder.parameters(), "weight_decay": self.weight_decay}, {"params": [radius], "weight_decay": 0.0}],
            lr=self.lr,
        )
        history = {"loss": [], "radius_sq": [], "penalty": [], "mean_dist": []}
        total_epochs = self.warmup_epochs + self.epochs
        self.encoder.train()
        for epoch in range(total_epochs):
            loss_sum = 0.0
            radius_sum = 0.0
            penalty_sum = 0.0
            dist_sum = 0.0
            steps = 0
            for (batch_x,) in loader:
                optimizer.zero_grad()
                out = self.encoder(batch_x)
                dist = torch.sum((out - self.center) ** 2, dim=1)
                radius_sq = radius ** 2
                penalty = torch.relu(dist - radius_sq).mean()
                loss = radius_sq + penalty / self.nu
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.item())
                radius_sum += float(radius_sq.item())
                penalty_sum += float(penalty.item())
                dist_sum += float(dist.mean().item())
                steps += 1
            history["loss"].append(loss_sum / steps)
            history["radius_sq"].append(radius_sum / steps)
            history["penalty"].append(penalty_sum / steps)
            history["mean_dist"].append(dist_sum / steps)
            print(f"epoch {epoch + 1}/{total_epochs} loss={history['loss'][-1]:.6f}")
        self.encoder.eval()
        with torch.no_grad():
            z_items = []
            for (batch_x,) in DataLoader(data, batch_size=self.batch_size, shuffle=False):
                z_items.append(self.encoder(batch_x))
            z = torch.cat(z_items, dim=0)
            dist = torch.sum((z - self.center) ** 2, dim=1)
            radius_sq = torch.quantile(dist, 1.0 - self.nu).item()
            self.radius = math.sqrt(float(radius_sq))
        return history

    def score(self, features: np.ndarray) -> np.ndarray:
        x = torch.tensor((features - self.mean) / self.std, dtype=torch.float32, device=self.device)
        self.encoder.eval()
        scores = []
        with torch.no_grad():
            for start in range(0, x.shape[0], self.batch_size):
                out = self.encoder(x[start : start + self.batch_size])
                scores.append(torch.sum((out - self.center) ** 2, dim=1).cpu().numpy())
        return np.concatenate(scores, axis=0)

    def save(self, model_path: str, layer_parts: list[str]) -> None:
        ensure_parent(model_path)
        tensors = {
            "center": self.center.detach().cpu(),
            "mean": torch.tensor(self.mean, dtype=torch.float32),
            "std": torch.tensor(self.std, dtype=torch.float32),
        }
        for name, value in self.encoder.state_dict().items():
            tensors[f"encoder.{name}"] = value.detach().cpu()
        save_file(tensors, model_path)
        meta = {
            "radius": self.radius,
            "scale_values": scale_values,
            "layer_parts": layer_parts,
            "hparams": {
                "num_layers": self.num_layers,
                "num_scales": self.num_scales,
                "num_steps": self.num_steps,
                "embedding_dim": self.embedding_dim,
                "nu": self.nu,
                "arch": "response_encoder",
            },
        }
        with open(detector_meta_path(model_path), "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)

    def load(self, model_path: str) -> dict:
        with open(detector_meta_path(model_path), "r", encoding="utf-8") as handle:
            meta = json.load(handle)
        hparams = meta["hparams"]
        self.num_layers = hparams["num_layers"]
        self.num_scales = hparams["num_scales"]
        self.num_steps = hparams["num_steps"]
        self.embedding_dim = hparams["embedding_dim"]
        self.nu = hparams["nu"]
        self.radius = meta["radius"]
        self.encoder = ResponseEncoder(self.num_layers, self.num_scales, self.num_steps, embedding_dim=self.embedding_dim).to(self.device)
        tensors = load_file(model_path)
        self.center = tensors["center"].to(self.device)
        self.mean = tensors["mean"].numpy()
        self.std = tensors["std"].numpy()
        state = {name.replace("encoder.", ""): value for name, value in tensors.items() if name.startswith("encoder.")}
        self.encoder.load_state_dict(state)
        self.encoder.to(self.device)
        return meta


def plot_training_scores(scores: np.ndarray, path: str, radius: float) -> None:
    ensure_parent(path)
    plt.figure(figsize=(9, 5))
    plt.hist(scores, bins=50, alpha=0.75, color="tab:green")
    plt.axvline(radius ** 2, color="tab:red", linestyle="--", linewidth=2)
    plt.xlabel("Distance to center")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_detection_report(scores_benign: np.ndarray, scores_backdoor: np.ndarray, fpr, tpr, roc_auc: float, matrix: np.ndarray, path: str) -> None:
    ensure_parent(path)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].hist(scores_benign, bins=50, alpha=0.7, label="Benign", color="tab:blue", density=True)
    axes[0].hist(scores_backdoor, bins=50, alpha=0.7, label="Backdoor", color="tab:red", density=True)
    axes[0].set_xlabel("Distance to center")
    axes[0].legend()
    axes[1].plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="black")
    axes[1].legend()
    axes[2].imshow(matrix, cmap="Blues")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            axes[2].text(col, row, str(matrix[row, col]), ha="center", va="center")
    axes[2].set_xticks([0, 1])
    axes[2].set_yticks([0, 1])
    axes[2].set_xticklabels(["Benign", "Backdoor"])
    axes[2].set_yticklabels(["Benign", "Backdoor"])
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
