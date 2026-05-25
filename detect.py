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
    parser.add_argument("--detector_path", type=str, default="models/set_detector.safetensors")
    parser.add_argument("--output_dir", type=str, default="outputs/detect")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--gpu", type=str, default="0")
    return parser.parse_args()


def run_detection(args) -> None:
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    from sklearn.metrics import auc, confusion_matrix, roc_curve

    from ptp_utils import (
        ResponseSpace,
        collect_response_features,
        default_attack_configs,
        load_pipeline,
        plot_detection_report,
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
    prompt_file = args.prompt_file or config.prompt_file
    prompts = read_prompts(prompt_file)
    if len(prompts) < 1000:
        raise ValueError("Detection prompt file must contain at least 1000 prompts")
    prompts_backdoor = prompts[:500]
    prompts_benign = prompts[500:1000]
    space = ResponseSpace(num_layers=1, device=device)
    space.load(args.detector_path)
    pipe = load_pipeline(args.base_model_path, config, device)
    backdoor_features, _ = collect_response_features(pipe, prompts_backdoor, args.seed, args.batch_size, device)
    benign_features, _ = collect_response_features(pipe, prompts_benign, args.seed + 500, args.batch_size, device)
    np.save(output_dir / "response_features_backdoor.npy", backdoor_features)
    np.save(output_dir / "response_features_benign.npy", benign_features)
    scores_backdoor = space.score(backdoor_features)
    scores_benign = space.score(benign_features)
    labels = np.concatenate([np.ones(len(scores_backdoor)), np.zeros(len(scores_benign))])
    scores = np.concatenate([scores_backdoor, scores_benign])
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    radius_sq = space.radius ** 2
    predictions = (scores > radius_sq).astype(int)
    matrix = confusion_matrix(labels, predictions)
    accuracy = float((predictions == labels).mean())
    result_path = output_dir / "detection_result.txt"
    with open(result_path, "w", encoding="utf-8") as handle:
        handle.write(f"Attack method: {args.attack_method}\n")
        handle.write(f"Prompt file: {prompt_file}\n")
        handle.write(f"Detector: {args.detector_path}\n")
        handle.write(f"Scale values: {scale_values}\n")
        handle.write(f"Backdoor features: {backdoor_features.shape}\n")
        handle.write(f"Benign features: {benign_features.shape}\n")
        handle.write(f"AUROC: {roc_auc:.6f}\n")
        handle.write(f"Accuracy: {accuracy:.6f}\n")
        handle.write(f"Radius squared: {radius_sq:.6e}\n")
        handle.write(f"Confusion matrix: {matrix.tolist()}\n")
        handle.write(f"Started: {start_time}\n")
        handle.write(f"Finished: {datetime.now()}\n")
    plot_detection_report(scores_benign, scores_backdoor, fpr, tpr, roc_auc, matrix, str(output_dir / "detection_report.png"))
    print(f"AUROC: {roc_auc:.4f}")
    print(f"accuracy: {accuracy:.4f}")
    print(f"saved result to {result_path}")


def main():
    run_detection(parse_args())


if __name__ == "__main__":
    main()
