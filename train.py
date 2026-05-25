import argparse
import os
from datetime import datetime
from pathlib import Path

import numpy as np


os.environ["TOKENIZERS_PARALLELISM"] = "false"
normal_model_id = "CompVis/stable-diffusion-v1-4"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack_method", type=str, default="rickrolling")
    parser.add_argument("--base_model_path", type=str, default=normal_model_id)
    parser.add_argument("--prompt_file", type=str, default=None)
    parser.add_argument("--backdoored_model_path", type=str, default=None)
    parser.add_argument("--detector_save_path", type=str, default="models/set_detector.safetensors")
    parser.add_argument("--output_dir", type=str, default="outputs/train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--nu", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--gpu", type=str, default="0")
    return parser.parse_args()


def fit_response_space(args) -> None:
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    from ptp_utils import (
        ResponseSpace,
        collect_response_features,
        default_attack_configs,
        ensure_parent,
        load_pipeline,
        plot_training_scores,
        probe_steps,
        read_prompts,
        scale_values,
        select_device,
        set_seed,
    )

    start_time = datetime.now()
    set_seed(args.seed)
    device = select_device(args.gpu)
    configs = default_attack_configs(args)
    if args.attack_method not in configs:
        raise ValueError(f"Unknown attack method: {args.attack_method}")
    config = configs[args.attack_method]
    output_dir = Path(args.output_dir) / args.attack_method
    output_dir.mkdir(parents=True, exist_ok=True)
    detector_path = args.detector_save_path
    prompt_file = args.prompt_file or str(Path("data") / "train_prompts.txt")
    prompts = read_prompts(prompt_file, args.max_samples)
    if not prompts:
        raise ValueError(f"No prompts found in {prompt_file}")
    pipe = load_pipeline(args.base_model_path, config, device)
    features, layer_parts = collect_response_features(pipe, prompts, args.seed, args.batch_size, device)
    feature_path = output_dir / "response_features.npy"
    np.save(feature_path, features)
    num_layers = features.shape[1] // (len(scale_values) * probe_steps)
    space = ResponseSpace(
        num_layers=num_layers,
        nu=args.nu,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        device=device,
    )
    history = space.fit(features)
    scores = space.score(features)
    ensure_parent(detector_path)
    space.save(detector_path, layer_parts)
    result_path = output_dir / "training_result.txt"
    with open(result_path, "w", encoding="utf-8") as handle:
        handle.write(f"Attack method: {args.attack_method}\n")
        handle.write(f"Prompt file: {prompt_file}\n")
        handle.write(f"Feature shape: {features.shape}\n")
        handle.write(f"Scale values: {scale_values}\n")
        handle.write(f"Radius: {space.radius:.6f}\n")
        handle.write(f"Radius squared: {space.radius ** 2:.6e}\n")
        handle.write(f"Started: {start_time}\n")
        handle.write(f"Finished: {datetime.now()}\n")
        handle.write(f"Final loss: {history['loss'][-1]:.6f}\n")
    plot_training_scores(scores, str(output_dir / "training_response_histogram.png"), space.radius)
    print(f"saved detector to {detector_path}")
    print(f"saved features to {feature_path}")
    print(f"saved result to {result_path}")


def main():
    fit_response_space(parse_args())


if __name__ == "__main__":
    main()
