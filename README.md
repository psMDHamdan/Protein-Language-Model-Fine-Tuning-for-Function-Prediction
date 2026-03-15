# 🧬 Protein Language Model Fine-Tuning for Function Prediction

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow)](https://huggingface.co)
[![Model](https://img.shields.io/badge/Model-ESM--2-green)](https://huggingface.co/facebook/esm2_t6_8M_UR50D)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)
[![Status](https://img.shields.io/badge/Status-In%20Development-orange)]()

Fine-tune pre-trained **Protein Language Models (PLMs)** — specifically **ESM-2** — on **Gene Ontology (GO) term prediction** from amino acid sequences alone.

> ⚠️ **This project is a work in progress.** The current implementation demonstrates the full pipeline end-to-end, but further development is needed before production use. See the [Future Work](#-future-work--further-development-needed) section.

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Pipeline](#-pipeline)
- [Quickstart](#-quickstart)
- [Project Structure](#-project-structure)
- [Results](#-results)
- [Future Work](#-future-work--further-development-needed)
- [Tech Stack](#-tech-stack)

---

## 🔍 Overview

Predicting protein function from sequence alone is one of the central challenges in computational biology. This project leverages large-scale unsupervised pre-training on millions of protein sequences (ESM-2) and adapts it to multi-label GO term prediction using **LoRA** (Low-Rank Adaptation) — making fine-tuning efficient even on small GPUs.

**Key facts:**
- **Data:** UniProtKB/Swiss-Prot via REST API (5,000 reviewed proteins, top-200 GO terms)
- **Model:** `facebook/esm2_t6_8M_UR50D` (8M params) with LoRA adapters (~14% trainable)
- **Task:** Multi-label classification (predicting multiple GO terms per protein)
- **Colab-ready:** Full pipeline runs on a free T4 GPU

---

## 🔄 Pipeline

```
UniProt REST API
      │
      ▼
Data Collection & Preprocessing
(5k proteins, top-200 GO terms, multi-label binarization)
      │
      ▼
ESM-2 + LoRA Fine-Tuning
(HuggingFace Trainer, fp16, cosine LR, early stopping)
      │
      ▼
Evaluation
(F1, Precision, Recall, AUC-ROC, per-label F1)
      │
      ▼
Interpretation
(Attention maps, Precision@k on known proteins: TP53, Insulin, Beta-galactosidase)
      │
      ▼
Inference
(predict_go_terms() — any sequence → ranked GO predictions)
```

---

## 🚀 Quickstart

### Option 1 — Google Colab (Recommended)
1. Open [`protein_function_prediction.ipynb`](protein_function_prediction.ipynb) in Colab
2. Set Runtime → **GPU (T4)**
3. Run **Cell 1** (install + upgrade packages) — the runtime will auto-restart
4. After restart, **skip Cell 1** and run all remaining cells

### Option 2 — Local
```bash
git clone https://github.com/psMDHamdan/Protein-Language-Model-Fine-Tuning-for-Function-Prediction.git
cd Protein-Language-Model-Fine-Tuning-for-Function-Prediction

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run data collection
python data_collection.py

# Run training
python trainer.py
```

---

## 📁 Project Structure

```
.
├── protein_function_prediction.ipynb  # Main notebook (Colab-ready, 11 sections)
├── data_collection.py                 # UniProt REST API downloader
├── dataset_loader.py                  # PyTorch Dataset + GO label binarizer
├── model_factory.py                   # ESM-2 + LoRA model builder
├── trainer.py                         # HuggingFace Trainer wrapper
├── requirements.txt                   # Python dependencies
├── data/                              # Downloaded protein data (git-ignored)
└── outputs/                           # Saved models + plots (git-ignored)
```

---

## 📊 Results

> Results are based on a small development run (5,000 Swiss-Prot proteins, top-200 GO terms, 3 epochs, ESM-2 8M, Colab T4).

| Metric | Value |
|--------|-------|
| Weighted F1 | _reported after training_ |
| Weighted Precision | _reported after training_ |
| Weighted Recall | _reported after training_ |
| AUC-ROC (macro) | _reported after training_ |

Interpretation was validated on 3 well-known proteins:
- **TP53** (DNA binding, apoptosis)
- **Human Insulin** (hormone activity, glucose homeostasis)
- **E. coli Beta-galactosidase** (hydrolase activity, carbohydrate metabolism)

---

## 🔭 Future Work — Further Development Needed

This project demonstrates the methodology but requires the following improvements before research use:

| Area | Current State | What's Needed |
|------|--------------|---------------|
| **Model scale** | ESM-2 8M params | Upgrade to ESM-2 650M or 3B for competitive accuracy |
| **Data scale** | 5,000 proteins, top-200 GO | Scale to full Swiss-Prot (~570k proteins, all GO terms) |
| **Label coverage** | Top-200 GO terms | Hierarchical GO propagation (parent terms) |
| **Baseline comparison** | None yet | BLASTp baseline on CAFA3/4 benchmarks |
| **Evaluation benchmark** | Held-out split | Validate on CAFA4 official test set |
| **Multi-task** | GO only | Add EC number + subcellular localization heads |
| **Calibration** | Uncalibrated sigmoid | Temperature scaling or Platt scaling |
| **Interpretability** | Attention maps | Probing classifiers, saliency maps |

---

## 🛠 Tech Stack

| Tool | Purpose |
|------|---------|
| `transformers` | ESM-2 model + Trainer |
| `peft` | LoRA parameter-efficient fine-tuning |
| `torch` | Deep learning backend |
| `biopython` | Protein sequence utilities |
| `scikit-learn` | Multi-label binarization, metrics |
| `pandas` | Data handling |
| `matplotlib` | Visualizations |

---

## 📚 References

- **ESM-2:** Lin et al., *Evolutionary-scale prediction of atomic-level protein structure with a language model*, Science 2023
- **LoRA:** Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022
- **UniProtKB:** The UniProt Consortium, *UniProt: the Universal Protein Knowledgebase*, NAR 2021
- **Gene Ontology:** The Gene Ontology Consortium, *The Gene Ontology resource*, NAR 2021
- **CAFA:** Zhou et al., *The CAFA challenge reports improved protein function prediction and new functional annotations*, Genome Biology 2019

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built by [Hamdan](https://github.com/psMDHamdan) · Contributions welcome!*
