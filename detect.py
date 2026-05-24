"""Run SET input-level backdoor detection."""

import argparse
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import auc, confusion_matrix, roc_curve

from ptp_utils import (
    ATTACK_CONFIGS,
    BASE_SD_PATH,
    DEFAULT_BATCH_SIZE,
    DEVICE,
    EncoderDetector,
    NUM_PROBE_STEPS,
    NUM_SCALES,
    RANDOM_SEED,
    SCALING_FACTORS,
    extract_response_offset_features,
    load_prompts_from_file,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate SET by probing response-offset features and scoring distance from the benign response space."
    )
    parser.add_argument("--attack_method", "--backdoor_method", dest="attack_method", required=True, choices=list(ATTACK_CONFIGS.keys()))
    parser.add_argument("--base_model_path", default=None, help="Path to the clean Stable Diffusion base model.")
    parser.add_argument("--prompt_file", default=None, help="Evaluation prompt file; first 500 prompts are backdoor and next 500 are benign.")
    parser.add_argument("--backdoored_model_path", default=None, help="Path to the backdoored T2I model/checkpoint.")
    parser.add_argument("--detector_path", "--model_path", dest="detector_path", default=None, help="Path to a trained SET detector.")
    parser.add_argument("--output_dir", default=None, help="Directory for detection reports and features.")
    parser.add_argument("--model_type", default=None, help="Backdoored model type override.")
    parser.add_argument("--seed", default=RANDOM_SEED, type=int)
    parser.add_argument("--gpu", type=str, default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--batch_size", "--batchsize", dest="batch_size", type=int, default=None)
    parser.add_argument("--encoder", action="store_true", default=False, help="Reuse existing response-offset features and only run scoring/reporting.")
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
        config["output_dir"] = os.path.join("outputs", "detect", f"{args.attack_method}_seed{args.seed}")
    if args.prompt_file:
        config["prompt_file"] = args.prompt_file
    if args.backdoored_model_path:
        config["poisoned_model_path"] = args.backdoored_model_path
    if args.model_type:
        config["model_type"] = args.model_type
    os.makedirs(config["output_dir"], exist_ok=True)


def load_detection_prompts(prompt_file):
    prompts = load_prompts_from_file(prompt_file)
    if len(prompts) < 1000:
        raise ValueError("Evaluation prompt file must contain at least 1000 prompts: 500 backdoor followed by 500 benign.")
    return prompts[:500], prompts[500:1000]


def load_encoder_detector(model_path):
    meta_path = model_path.replace(".safetensors", "_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Meta file not found: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    hparams = meta["hparams"]
    detector = EncoderDetector(
        num_layers=hparams["num_layers"],
        num_scales=hparams.get("num_scales", meta.get("num_scales", len(SCALING_FACTORS))),
        num_steps=hparams.get("num_steps", NUM_PROBE_STEPS),
        embedding_dim=hparams["embedding_dim"],
        device=DEVICE,
        input_dim=hparams.get("input_dim", 1),
    )
    detector.load(model_path)
    return detector, meta


def extract_detection_features(args, config, prompts_backdoor, prompts_benign):
    output_dir = config["output_dir"]
    fp_bd_path = os.path.join(output_dir, "fingerprints_backdoor.npy")
    fp_b_path = os.path.join(output_dir, "fingerprints_benign.npy")

    if args.encoder:
        if not os.path.exists(fp_bd_path) or not os.path.exists(fp_b_path):
            raise FileNotFoundError("--encoder requires fingerprints_backdoor.npy and fingerprints_benign.npy in the output directory.")
        return np.load(fp_bd_path), np.load(fp_b_path), fp_bd_path, fp_b_path

    fp_backdoor = extract_response_offset_features(
        prompts=prompts_backdoor,
        config=config,
        scaling_factors=SCALING_FACTORS,
        seed=args.seed,
        base_model_path=args.base_model_path or BASE_SD_PATH,
        output_dir=output_dir,
        device=DEVICE,
        batch_size=args.batch_size,
    )
    np.save(fp_bd_path, fp_backdoor)

    fp_benign = extract_response_offset_features(
        prompts=prompts_benign,
        config=config,
        scaling_factors=SCALING_FACTORS,
        seed=args.seed,
        base_model_path=args.base_model_path or BASE_SD_PATH,
        output_dir=output_dir,
        device=DEVICE,
        batch_size=args.batch_size,
    )
    np.save(fp_b_path, fp_benign)
    return fp_backdoor, fp_benign, fp_bd_path, fp_b_path


def score_detection_features(detector, fp_backdoor, fp_benign):
    scores_benign = detector.score_samples(fp_benign)
    scores_backdoor = detector.score_samples(fp_backdoor)
    y_true = np.concatenate([np.ones(len(scores_backdoor)), np.zeros(len(scores_benign))])
    y_scores = np.concatenate([scores_backdoor, scores_benign])
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    threshold = detector.radius ** 2
    y_pred = (y_scores > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    return {
        "scores_benign": scores_benign,
        "scores_backdoor": scores_backdoor,
        "y_true": y_true,
        "y_scores": y_scores,
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": roc_auc,
        "threshold": threshold,
        "accuracy": accuracy,
        "confusion": (tn, fp, fn, tp),
    }


def save_detection_outputs(args, config, meta, prompts_backdoor, prompts_benign, fp_backdoor, fp_benign, metrics, start_time):
    output_dir = config["output_dir"]
    result_path = os.path.join(output_dir, "result.txt")
    end_time = datetime.now()

    scores_benign = metrics["scores_benign"]
    scores_backdoor = metrics["scores_backdoor"]
    tn, fp, fn, tp = metrics["confusion"]

    with open(result_path, "w", encoding="utf-8") as f:
        f.write(f"Training Set: {config['train_prompt_file']}\n")
        f.write(f"Test Set: {config['prompt_file']}\n")
        f.write(f"Lambda Values: {SCALING_FACTORS}\n")
        f.write(f"Num Scales: {NUM_SCALES}\n")
        f.write(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Duration: {end_time - start_time}\n\n")
        f.write(f"AUROC: {metrics['roc_auc']:.6f}\n")
        f.write(f"Threshold R²: {metrics['threshold']:.4e}\n")
        f.write(f"Accuracy: {metrics['accuracy']:.6f}\n")
        f.write(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}\n")
        f.write("\nBackdoor Samples\n")
        for i, (prompt, score) in enumerate(zip(prompts_backdoor, scores_backdoor), start=1):
            f.write(f"{i}. {prompt[:80]} | Distance: {score:.4e}\n")
        f.write("\nBenign Samples\n")
        for i, (prompt, score) in enumerate(zip(prompts_benign, scores_benign), start=1):
            f.write(f"{i}. {prompt[:80]} | Distance: {score:.4e}\n")

    num_layers = meta["hparams"]["num_layers"]
    num_steps = meta["hparams"].get("num_steps", NUM_PROBE_STEPS)
    fp_b_4d = fp_benign.reshape(-1, len(SCALING_FACTORS), num_steps, num_layers)
    fp_bd_4d = fp_backdoor.reshape(-1, len(SCALING_FACTORS), num_steps, num_layers)
    mean_b_per_layer = np.mean(fp_b_4d, axis=(0, 1, 2))
    mean_bd_per_layer = np.mean(fp_bd_4d, axis=(0, 1, 2))

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    fig.suptitle(f"Encoder Detection Report - {args.attack_method}", fontweight="bold")

    axes[0].hist(scores_benign, bins=50, alpha=0.7, label="Benign", color="blue", density=True)
    axes[0].hist(scores_backdoor, bins=50, alpha=0.7, label="Backdoor", color="red", density=True)
    axes[0].axvline(metrics["threshold"], color="orange", linestyle="--", linewidth=2, label=f"R²={metrics['threshold']:.2e}")
    axes[0].set_title("Distance Distribution")
    axes[0].set_xlabel("Distance to Center")
    axes[0].set_ylabel("Density")
    axes[0].legend(fontsize=8)

    axes[1].plot(metrics["fpr"], metrics["tpr"], "b-", label=f"AUC={metrics['roc_auc']:.3f}")
    axes[1].plot([0, 1], [0, 1], "k--")
    axes[1].set_title("ROC Curve")
    axes[1].legend()

    cm = np.array([[tn, fp], [fn, tp]])
    axes[2].imshow(cm, cmap="Blues")
    for i, j in np.ndindex(cm.shape):
        axes[2].text(j, i, str(cm[i, j]), ha="center", va="center")
    axes[2].set_xticks([0, 1])
    axes[2].set_yticks([0, 1])
    axes[2].set_xticklabels(["Pred Benign", "Pred Backdoor"])
    axes[2].set_yticklabels(["True Benign", "True Backdoor"])
    axes[2].set_title("Confusion Matrix")

    x = np.arange(num_layers)
    axes[3].plot(x, mean_b_per_layer, "o-", color="blue", label="Benign", alpha=0.7, markersize=2)
    axes[3].plot(x, mean_bd_per_layer, "o-", color="red", label="Backdoor", alpha=0.7, markersize=2)
    axes[3].set_title("Per-Layer Mean MSE")
    axes[3].set_xlabel("Cross-Attention Layer")
    axes[3].set_ylabel("Mean MSE")
    axes[3].legend()
    axes[3].grid(alpha=0.4)

    plt.tight_layout()
    report_path = os.path.join(output_dir, "report_overall.png")
    plt.savefig(report_path, dpi=300)
    plt.close()

    print(f"AUROC: {metrics['roc_auc']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Results saved to: {output_dir}")


def run_detection(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    start_time = datetime.now()
    config = ATTACK_CONFIGS[args.attack_method]
    model_path = args.detector_path if args.detector_path else config["output_model"]
    prompts_backdoor, prompts_benign = load_detection_prompts(config["prompt_file"])
    detector, meta = load_encoder_detector(model_path)
    fp_backdoor, fp_benign, _, _ = extract_detection_features(args, config, prompts_backdoor, prompts_benign)
    metrics = score_detection_features(detector, fp_backdoor, fp_benign)
    save_detection_outputs(args, config, meta, prompts_backdoor, prompts_benign, fp_backdoor, fp_benign, metrics, start_time)


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.batch_size is None:
        args.batch_size = DEFAULT_BATCH_SIZE
    apply_path_overrides(args)
    run_detection(args)


if __name__ == "__main__":
    main()
