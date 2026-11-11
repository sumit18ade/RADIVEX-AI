# RADIVEX AI
### Radiology Vision Intelligence

**Advancing Medical Image Analysis Through AI**

RADIVEX AI is an enhanced and independently developed version built upon the existing ViT-Med research codebase. The project extends the inherited chest-radiograph AI workflow with additional dataset-preparation utilities, validation scripts, configuration work, and planned model experiments.

This repository acknowledges the original codebase and research contributions while documenting the additional work carried out for RADIVEX AI.

---

## Project Overview

RADIVEX AI explores deep-learning approaches for chest X-ray image analysis, with an emphasis on Vision Transformer architectures and pretrained visual representations.

The current development work includes:
- Dataset inspection and preparation utilities.
- Chest X-ray label processing and CSV metadata generation.
- Image preprocessing and dataset validation.
- PyTorch DataLoader testing.
- Configuration updates for local dataset use.
- A foundation for further model training, evaluation, and inference experiments.

**Development status:** Dataset preparation and DataLoader validation have been tested on a small sample dataset. Full-scale training, comprehensive evaluation, and clinical validation are not yet complete.

## What Has Been Extended

The repository retains the underlying research implementation and adds RADIVEX AI development work, including:
- Additional scripts for dataset inspection and preparation.
- Label-building and label-distribution utilities.
- Dataset validation checks.
- DataLoader smoke testing.
- Project-specific configuration and documentation.

The project will continue to evolve as further experiments and evaluation are completed.

## Technology

- Python
- PyTorch and Torchvision
- NumPy and Pandas
- OpenCV
- scikit-learn
- Hugging Face tools
- timm
- Matplotlib and TensorBoard

## Repository Structure

```text
RADIVEX-AI/
├── config/
├── data/
├── models/
├── build_labels.py
├── build_labels_fixed.py
├── check_dataset.py
├── extract_sample.py
├── inspect_dataset.py
├── inspect_labels.py
├── label_distribution.py
├── prepare_sample.py
├── sample_inspect.py
├── test_dataloader.py
├── validate_dataset.py
├── feature_extractor.py
├── main_vitmed.py
├── Train_Valid_vitmed.py
├── Prediction_vitmed.py
├── README.md
└── LICENSE
```

## Dataset

The data pipeline is designed around the NIH ChestX-ray14 dataset format. Dataset files are not included in this repository. Obtain the dataset from its authorized source and comply with its terms of use.

Expected local layout:

```text
sample_data/
└── NIH_ChestX-ray14/
    ├── final_cxr14_master_list.csv
    └── CXR14/
        └── preprocessed224/
```

Set the appropriate dataset root in `config/config.yaml`. The current local development configuration uses:

```yaml
file_path: sample_data/
```

The sample dataset used for initial validation is not a substitute for the complete dataset and should not be used to draw model-performance conclusions.

## Environment Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install compatible dependencies for your Python and hardware environment:

```bash
python -m pip install torch torchvision
python -m pip install numpy pandas scipy scikit-learn matplotlib opencv-python tqdm tensorboardX timm transformers datasets accelerate huggingface-hub tokenizers safetensors
```

The original research implementation used specific library versions. For reproducible experiments, check package compatibility and use a suitable GPU-enabled PyTorch build when training on NVIDIA hardware.

## Current Validation Commands

Run from the repository root after preparing the dataset:

```bash
python validate_dataset.py
python check_dataset.py
python test_dataloader.py
```

The main training and inference scripts are inherited from the underlying research implementation. Review `config/config.yaml` and the relevant script arguments before running experiments.

## Research Background

This project builds upon the following research publications associated with the underlying ViT-Med research codebase:

1. S. Tayebi Arasteh, L. Misera, J. N. Kather, D. Truhn, and S. Nebelung, “Enhancing diagnostic deep learning via self-supervised pretraining on large-scale, unlabeled non-medical images,” *European Radiology Experimental*, 2024. https://doi.org/10.1186/s41747-023-00411-3

2. S. Tayebi Arasteh et al., “Resolution-dependent self-supervised transfer in chest radiograph classification,” *Communications Medicine*, 2026. https://doi.org/10.1038/s43856-026-01897-9

Original research repository: https://github.com/tayebiarasteh/vit-med

## Roadmap

- [x] Repository setup for RADIVEX AI
- [x] Dataset inspection and sample preparation
- [x] Label-processing utilities
- [x] Dataset validation
- [x] DataLoader smoke test
- [ ] Complete dataset integration
- [ ] Full model training experiments
- [ ] Comprehensive performance evaluation
- [ ] Inference interface and documentation

## Important Disclaimer

RADIVEX AI is an academic research and experimental project. It is not a clinically validated medical device or diagnostic system. Its outputs must not be used to make medical decisions or replace advice from qualified healthcare professionals.

## Authors and Contributions

**Original research codebase:** Soroosh Tayebi Arasteh and the contributors credited in the original repository.

**RADIVEX AI extension and project development:** Sumit Ganesh Ade.

RADIVEX AI builds upon the existing codebase and adds further project-specific development. Original notices and attribution are retained in accordance with the applicable license.

- GitHub: https://github.com/sumit18ade
- Portfolio: https://sumit18x-portfolio.vercel.app/

---

<p align="center">
  <strong>RADIVEX AI</strong><br>
  Radiology Vision Intelligence<br>
  Advancing Medical Image Analysis Through AI
</p>

