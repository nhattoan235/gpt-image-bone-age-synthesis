# GPT-Image Bone Age Synthesis

> Investigating GPT-Image generative techniques for synthetic hand X-ray generation, evaluated through the bone age estimation task — with the goal of proposing improvements to generative methods for hand radiographs.

**Status:** 🚧 In progress (undergraduate thesis project)

---

## 📋 Overview

This project investigates the use of OpenAI's **GPT-Image (gpt-image-1)** generative model for synthesizing / editing pediatric hand X-ray images, and evaluates the clinical reliability of the generated images through a downstream deep learning task: **bone age estimation** from hand radiographs.

The work is grounded in recent findings showing that visually realistic generative inpainting can significantly degrade the performance of medical AI models — even when edits are confined to non-diagnostic regions of the image. Building on this baseline, this project aims to:

1. Survey the bone age estimation problem and relevant deep learning approaches
2. Analyze the **Synthetic Hand X-Ray** dataset (RSNA-based, GPT-Image-inpainted)
3. Reproduce and extend the experimental pipeline from the reference paper
4. Evaluate deep learning models trained/tested on synthetic data
5. Analyze the root causes of performance degradation
6. Propose directions to improve the generative approach

## 🎯 Objectives

- [ ] Survey the bone age prediction problem from X-ray images
- [ ] Analyze the Synthetic Hand X-Ray dataset
- [ ] Study the GPT-Image-1 generation method described in the reference paper
- [ ] Run generation experiments (prompt/parameter variations)
- [ ] Evaluate results using deep learning models on the generated dataset
- [ ] Analyze the causes of performance degradation
- [ ] Propose and evaluate improvements to the generative approach

## 🗂️ Datasets

| Dataset | Link | Notes |
|---|---|---|
| RSNA Bone Age Challenge (Kaggle mirror) | https://www.kaggle.com/datasets/kmader/rsna-bone-age | Original dataset: 14,236 pediatric hand radiographs (12,611 train / 1,425 val / 200 test), labeled with bone age (months) and sex |
| RSNA Pediatric Bone Age Challenge — official page | https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-pediatric-bone-age-challenge-2017 | Official description, usage & attribution terms |
| Synthetic Hand X-Ray Dataset for Bone Age | https://www.kaggle.com/datasets/felipematsuoka/synthetic-hand-x-ray-dataset-for-bone-age/ | 200 original + 600 GPT-Image-1-inpainted radiographs (CC BY 4.0) |

## 📚 References

**Core reference paper**
- Matsuoka, F. A. et al. (2025). *Evaluating the Clinical Impact of Generative Inpainting on Bone Age Estimation.* arXiv:2511.23066.
  - Abstract page: https://arxiv.org/abs/2511.23066
  - Full PDF: https://arxiv.org/pdf/2511.23066
  - Reference code & pipeline (GitHub): https://github.com/felipe-matsuoka123/EVALUATING-THE-CLINICAL-IMPACT-OF-GENERATIVE-INPAINTING-ON-BONE-AGE-ESTIMATION

**Dataset origin paper**
- Halabi, S. S., Prevedello, L. M., Kalpathy-Cramer, J., et al. (2018). *The RSNA Pediatric Bone Age Machine Learning Challenge.* Radiology, 290(2), 498–503. https://pubs.rsna.org/doi/10.1148/radiol.2018180736

**GPT-Image / OpenAI API documentation**
- Image generation guide: https://developers.openai.com/api/docs/guides/image-generation
- API reference (Create image / Edit image): https://developers.openai.com/api/reference/resources/images/methods/generate
- GPT Image 1 model page: https://developers.openai.com/api/docs/models/gpt-image-1

## 🧠 Key Baseline Findings (from the reference paper)

These numbers serve as the baseline this project compares against:

| Metric | Original images | Inpainted (GPT-Image-1) images |
|---|---|---|
| Bone age MAE (months) | 6.26 | 30.11 |
| Bone age MAE after linear calibration | — | 19.55 |
| Gender classification AUC | 0.955 | 0.704 |
| Pixel intensity SD | 23.05 | 31.91 |

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:** NumPy, OpenCV, TensorFlow / PyTorch
- **Environment:** Google Colab / Visual Studio Code
- **Generative model:** OpenAI GPT-Image-1 API

## 📁 Project Structure

```
gpt-image-bone-age-synthesis/
├── docs/                  # Survey & analysis documents (literature review, dataset analysis)
├── data/                  # Scripts to download/prepare RSNA + Synthetic Hand X-Ray datasets
├── generation/             # GPT-Image generation pipeline (prompts, masks, parameters)
├── models/                # Downstream models (bone age regression, gender classification)
├── evaluation/            # MAE/RMSE/AUC computation, statistical analysis
├── notebooks/              # Experiment notebooks (Colab-ready)
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- An OpenAI API key with access to `gpt-image-1` (requires organization verification)

### Installation
```bash
git clone https://github.com/nhattoan235/gpt-image-bone-age-synthesis.git
cd gpt-image-bone-age-synthesis
pip install -r requirements.txt
```

### Environment variables
Create a `.env` file:
```
OPENAI_API_KEY=your_api_key_here
```

## 📊 Evaluation Metrics

- **MAE / RMSE** (months) — bone age regression error
- **AUC** — gender classification performance
- **Pixel intensity distribution** — to detect structural alterations introduced by generation
- **Inter-generation consistency** — standard deviation of predictions across multiple synthetic versions of the same patient

## ⚠️ Known Limitations (inherited from the reference study)

- Generated images are PNG without DICOM metadata (no spatial calibration or modality parameters)
- Patient demographic information beyond sex is not available, limiting subgroup bias analysis
- Findings are based on a single generative model configuration (gpt-image-1, "high" quality, single prompt template)

## 📝 License

This project is for academic research purposes (undergraduate thesis). Dataset usage follows the original licenses:
- RSNA Bone Age Challenge Dataset — academic research and education use only, per RSNA attribution requirements
- Synthetic Hand X-Ray Dataset for Bone Age — CC BY 4.0

## 👤 Author

*[Nguyen Nhat Toan - Dinh Tan Phuong - Le Tran Phu]* — Undergraduate thesis, [HUIT], [2026]
