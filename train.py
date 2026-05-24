"""Train the SET benign response space detector."""

import argparse
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch

from ptp_utils import (
    ATTACK_CONFIGS,
    BASE_SD_PATH,
    DEFAULT_BATCH_SIZE,
    DEVICE,
    EncoderDetector,
    MAX_SAMPLES,
    NU,
    NUM_EPOCHS,
    NUM_PROBE_STEPS,
    NUM_SCALES,
    RANDOM_SEED,
    SCALING_FACTORS,
    THRESHOLD_PERCENTILE,
    WEIGHT_DECAY,
    extract_response_offset_features,
    load_prompts_from_file,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SET by learning a benign response space from clean prompt responses."
    )
    parser.add_argument("--attack_method", "--backdoor_method", dest="attack_method", required=True, choices=list(ATTACK_CONFIGS.keys()))
    parser.add_argument("--base_model_path", default=None, help="Path to the clean Stable Diffusion base model.")
    parser.add_argument("--prompt_file", default=None, help="Training prompt file; one prompt per line.")
    parser.add_argument("--backdoored_model_path", default=None, help="Path to the backdoored T2I model/checkpoint.")
    parser.add_argument("--detector_save_path", default=None, help="Where to save the trained SET detector.")
    parser.add_argument("--output_dir", default=None, help="Directory for fingerprints, logs, and training figures.")
    parser.add_argument("--model_type", default=None, help="Backdoored model type: text_encoder, lora, unet, unet_state_dict, textual_inversion, or clean.")
    parser.add_argument("--seed", default=RANDOM_SEED, type=int)
    parser.add_argument("--max_samples", default=MAX_SAMPLES, type=int)
    parser.add_argument("--nu", default=NU, type=float)
    parser.add_argument("--encoder", action="store_true", help="Reuse existing response-offset features and train only the encoder detector.")
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch_size", "--batchsize", dest="batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--gpu", type=str, default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    return parser.parse_args()


def apply_path_overrides(args):
    config = ATTACK_CONFIGS[args.attack_method]
    if args.base_model_path:
        import ptp_utils
        ptp_utils.BASE_SD_PATH = args.base_model_path
        os.environ["SET_BASE_MODEL_PATH"] = args.base_model_path
    if args.output_dir:
        config["output_dir"] = args.output_dir
    else:
        config["output_dir"] = os.path.join("outputs", "train", f"{args.attack_method}_seed{args.seed}")
    if args.prompt_file:
        config["train_prompt_file"] = args.prompt_file
    if args.backdoored_model_path:
        config["poisoned_model_path"] = args.backdoored_model_path
    if args.detector_save_path:
        config["output_model"] = args.detector_save_path
    else:
        config["output_model"] = os.path.join("models", f"set_detector_{args.attack_method}.safetensors")
    if args.model_type:
        config["model_type"] = args.model_type
    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(config["output_model"]) or ".", exist_ok=True)


def load_training_prompts(prompt_file, max_samples):
    prompts = load_prompts_from_file(prompt_file)
    if not prompts:
        raise ValueError(f"No prompts found in {prompt_file}.")
    return prompts[:max_samples] if max_samples else prompts


def extract_clean_response_features(args, config, prompts):
    output_dir = config["output_dir"]
    fingerprint_path = os.path.join(output_dir, f"{args.attack_method}_fingerprints.npy")
    if args.encoder:
        if not os.path.exists(fingerprint_path):
            raise FileNotFoundError(f"Feature file not found for --encoder: {fingerprint_path}")
        return np.load(fingerprint_path), fingerprint_path

    features = extract_response_offset_features(
        prompts=prompts,
        config=config,
        scaling_factors=SCALING_FACTORS,
        seed=args.seed,
        base_model_path=args.base_model_path or BASE_SD_PATH,
        output_dir=output_dir,
        device=DEVICE,
        batch_size=args.batch_size,
    )
    np.save(fingerprint_path, features)
    return features, fingerprint_path


def fit_benign_response_space(features, prompts, args, layer_stages):
    total_dim = features.shape[1]
    num_layers = total_dim // (NUM_SCALES * NUM_PROBE_STEPS)
    detector = EncoderDetector(
        num_layers=num_layers,
        num_scales=NUM_SCALES,
        num_steps=NUM_PROBE_STEPS,
        embedding_dim=16,
        nu=args.nu,
        batch_size=256,
        num_epochs=args.epochs,
        device=DEVICE,
        weight_decay=args.weight_decay,
        warmup_epochs=5,
        input_dim=1,
        layer_stages=layer_stages,
    )
    history = detector.fit(features, prompts=prompts)
    return detector, history


def save_training_outputs(args, config, prompts, features, detector, history, fingerprint_path, start_time):
    output_dir = config["output_dir"]
    end_time = datetime.now()
    scores = detector.score_samples(features)
    result_path = os.path.join(output_dir, "result.txt")

    with open(result_path, "w", encoding="utf-8") as f:
        f.write("Training Configuration\n")
        f.write(f"Prompt File: {config['train_prompt_file']}\n")
        f.write(f"Feature File: {fingerprint_path}\n")
        f.write(f"Scaling Factors: {SCALING_FACTORS}\n")
        f.write(f"NU: {args.nu}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Start Time: {start_time}\n")
        f.write(f"End Time: {end_time}\n")
        f.write(f"Duration: {end_time - start_time}\n")
        f.write(f"Radius (R): {detector.radius:.4f}\n")
        f.write(f"Threshold R²: {detector.radius**2:.4e}\n")
        if getattr(detector, "threshold_percentile", None) is not None:
            f.write(f"Threshold {THRESHOLD_PERCENTILE*100:.0f}% Percentile: {detector.threshold_percentile:.4e}\n")
        f.write("\nTop 10 Largest Distances\n")
        for idx in np.argsort(scores)[-10:][::-1]:
            prompt_text = prompts[idx].replace("\n", " ") if idx < len(prompts) else "Prompt not found"
            f.write(f"Idx {idx} | Distance: {scores[idx]:.4e} | Prompt: {prompt_text}\n")

    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=50, alpha=0.75, color="green", label="Training Distances")
    plt.axvline(detector.radius ** 2, color="red", linestyle="--", linewidth=2, label=f"R²={detector.radius**2:.4e}")
    if getattr(detector, "threshold_percentile", None) is not None:
        plt.axvline(detector.threshold_percentile, color="blue", linestyle="-.", linewidth=2, label=f"{THRESHOLD_PERCENTILE*100:.0f}%={detector.threshold_percentile:.2e}")
    plt.title(f"Encoder Distances - {args.attack_method}")
    plt.xlabel("Distance to Center")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_encoder_histogram_overall.png"))
    plt.close()

    if history:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history["loss"], label="Total Loss")
        plt.plot(history["r_sq"], label="R²")
        plt.plot(history["penalty"], label="Penalty")
        plt.xlabel("Epochs")
        plt.ylabel("Value")
        plt.title("Loss Components")
        plt.legend()
        plt.grid(True)
        plt.subplot(1, 2, 2)
        plt.plot(history["r_sq"], label="R²", linestyle="--")
        plt.plot(history["avg_dist"], label="Avg Distance")
        plt.xlabel("Epochs")
        plt.ylabel("Squared Distance")
        plt.title("Radius vs Data Distribution")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "training_loss_curves.png"))
        plt.close()

    layer_stages_path = os.path.join(output_dir, "layer_stages.json")
    layer_stages = None
    if os.path.exists(layer_stages_path):
        with open(layer_stages_path, "r", encoding="utf-8") as f:
            layer_stages = json.load(f)

    local_model_path = os.path.join(output_dir, "detector.safetensors")
    detector.save(local_model_path, layer_stages=layer_stages)
    detector.save(config["output_model"], layer_stages=layer_stages)
    print(f"Training completed. Results: {result_path}")


def run_training(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    start_time = datetime.now()
    config = ATTACK_CONFIGS[args.attack_method]
    prompts = load_training_prompts(config["train_prompt_file"], args.max_samples)
    print(f"Loaded {len(prompts)} clean prompts for benign response space training.")

    features, fingerprint_path = extract_clean_response_features(args, config, prompts)
    layer_stages_path = os.path.join(config["output_dir"], "layer_stages.json")
    layer_stages = None
    if os.path.exists(layer_stages_path):
        with open(layer_stages_path, "r", encoding="utf-8") as f:
            layer_stages = json.load(f)

    detector, history = fit_benign_response_space(features, prompts, args, layer_stages)
    save_training_outputs(args, config, prompts, features, detector, history, fingerprint_path, start_time)


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    apply_path_overrides(args)
    run_training(args)


if __name__ == "__main__":
    main()
