# SET: Input-Level Backdoor Detection in Text-to-Image Diffusion Models

Official implementation for **Scaling Exposes the Trigger: Input-Level Backdoor Detection in Text-to-Image Diffusion Models via Cross-Attention Scaling**.

SET detects backdoored text prompts for text-to-image diffusion models from an active probing perspective. It applies controlled cross-attention scaling perturbations, observes **Cross-Attention Scaling Response Divergence (CSRD)** across denoising steps, extracts response-offset features, and learns a compact benign response space for input-level backdoor detection.

## News

- 2026-04-14: Paper released on arXiv: [arXiv:2604.12446](https://arxiv.org/abs/2604.12446).

## Method Overview

1. **Cross-attention scaling probe**: apply controlled multi-scale perturbations to cross-attention during T2I denoising.
2. **Response-offset feature extraction**: capture response evolution across probe steps and construct CSRD-style response-offset features.
3. **Benign response space fitting**: train an encoder detector on clean prompts to model compact benign response behavior.
4. **Deviation-based detection**: score evaluation prompts by distance from the learned benign response space.

## Repository Structure

```text
SET/
├── train.py          # training pipeline
├── detect.py         # detection pipeline
├── ptp_utils.py      # Attention scaling, feature extraction, model loading, detector, and P2P utilities
├── requirements.txt  # Python dependencies
├── README.md
├── LICENSE
└── CITATION.cff
```

## Environment Setup

```bash
conda create -n set python=3.10
conda activate set
pip install -r requirements.txt
```

The code expects a CUDA-enabled PyTorch environment for practical experiments.

## Data and Model Preparation

Prepare the following resources locally:

- A Stable Diffusion v1-4 base model.
- Prompt files in plain text format, one prompt per line.
- Backdoored model checkpoints for the attack setting being evaluated.

Suggested layout:

```text
SET/
├── data/
│   ├── train_prompts.txt
│   └── eval_prompts/
│       ├── rickrolling.txt
│       ├── twt.txt
│       └── clean.txt
├── checkpoints/
│   └── <attack-checkpoints>
├── models/
└── outputs/
```

The built-in attack names include `rickrolling`, `twt`, `villan_mignneko`, `villan_github`, `villan_anonymous`, `eviledit`, `pixel`, and `clean`. Important paths can be overridden from the command line.

## Train SET Detector

`train.py` implements the training-side pipeline: load clean prompts, run cross-attention scaling probes, extract response-offset features, fit the encoder detector, and save the benign response space model.

Example:

```bash
python train.py \
  --attack_method rickrolling \
  --base_model_path /path/to/stable-diffusion-v1-4 \
  --prompt_file data/train_prompts.txt \
  --backdoored_model_path checkpoints/rickrolling \
  --detector_save_path models/set_detector_rickrolling.safetensors \
  --output_dir outputs/train/rickrolling_seed42 \
  --seed 42 \
  --epochs 100 \
  --batch_size 20 \
  --gpu 0
```

Useful options:

- `--attack_method`: attack/backdoor setting.
- `--base_model_path`: clean T2I base model path.
- `--prompt_file`: clean prompts used to train the benign response space.
- `--backdoored_model_path`: backdoored model/checkpoint path.
- `--detector_save_path`: output path for the trained SET detector.
- `--output_dir`: output directory for features, logs, and figures.
- `--encoder`: reuse existing response-offset features in `--output_dir` and train only the encoder detector.

## Run Detection

`detect.py` implements the detection-side pipeline: load evaluation prompts, extract response-offset features for backdoor and benign inputs, score distances from the benign response space, and save the detection report.

The evaluation prompt file is expected to contain at least 1000 prompts: the first 500 are treated as backdoor prompts and the next 500 as benign prompts.

Example:

```bash
python detect.py \
  --attack_method rickrolling \
  --base_model_path /path/to/stable-diffusion-v1-4 \
  --prompt_file data/eval_prompts/rickrolling.txt \
  --backdoored_model_path checkpoints/rickrolling \
  --detector_path models/set_detector_rickrolling.safetensors \
  --output_dir outputs/detect/rickrolling_seed42 \
  --seed 42 \
  --batch_size 20 \
  --gpu 0
```

Useful options:

- `--detector_path`: trained detector path. If omitted, the attack config default is used.
- `--encoder`: reuse `fingerprints_backdoor.npy` and `fingerprints_benign.npy` in `--output_dir` and only run scoring/reporting.

## Outputs

Training produces files such as:

- `result.txt`: training summary and learned threshold information.
- `<attack>_fingerprints.npy`: clean response-offset features.
- `detector.safetensors` and `models/set_detector_<attack>.safetensors`: trained benign response space detector.
- `training_encoder_histogram_overall.png` and `training_loss_curves.png`: diagnostic figures.
- `layer_stages.json`: selected cross-attention layer stage metadata.

Detection produces files such as:

- `result.txt`: AUROC, threshold, accuracy, and per-sample distance summary.
- `fingerprints_backdoor.npy` / `fingerprints_benign.npy`: extracted response-offset features.
- `report_overall.png`: distance distribution, ROC curve, confusion matrix, and per-layer MSE summary.

## Citation

If you find this repository useful, please cite:

```bibtex
@article{li2026scaling,
  title={Scaling Exposes the Trigger: Input-Level Backdoor Detection in Text-to-Image Diffusion Models via Cross-Attention Scaling},
  author={Li, Zida and Li, Jun and Sha, Yuzhe and Li, Ziqiang and Xiong, Lizhi and Fu, Zhangjie},
  journal={arXiv preprint arXiv:2604.12446},
  year={2026},
  doi={10.48550/arXiv.2604.12446}
}
```

## License

This project is released under the Apache License 2.0.

## Acknowledgements

This code builds on the PyTorch, Hugging Face Diffusers, Transformers, and Stable Diffusion ecosystems. Parts of the attention-control utilities follow the Prompt-to-Prompt style implementation.
