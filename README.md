# 🩺 Medical Image Captioning

A Final Year B.Tech Project submitted to the **Department of Computer Science and Engineering**, Indian Institute of Information Technology Kottayam (April 2025). This project focuses on generating clinically relevant captions for chest X-ray images using state-of-the-art transformer-based vision-language models.

---

## 📌 Table of Contents

- [Abstract](#abstract)
- [Contributors](#contributors)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Models Used](#models-used)
  - [Two-Stage Fine-Tuning](#two-stage-fine-tuning)
- [Training Configuration](#training-configuration)
- [Evaluation & Results](#evaluation--results)
- [Conclusion](#conclusion)
- [Future Work](#future-work)
- [References](#references)
- [Contact](#contact)

---

## 📚 Abstract

The objective of this project is to develop an AI-driven system for automated medical image captioning, designed to generate clinically accurate, contextually relevant captions for chest X-rays. Unlike manual radiological reporting, our system provides a scalable and efficient solution by leveraging transformer-based models: **ViT+GPT2**, **BLIP-2**, and **GIT**. We also introduce a **two-stage fine-tuning pipeline** (MeSH-based and caption-based) and an enhanced model that integrates **DINOv2** to capture fine-grained visual features. This project aims to support radiologists, improve diagnostic efficiency, and reduce cognitive load.

---

## 👨‍💻 Contributors

- **Gandhi Deep Bankim** – 2021BCS0040  
- **Rohit Raj** – 2021BCS0065  
- **Abhishek Raj** – 2021BCS0141  
- **Gaurav Yadav** – 2021BCS0023  
- **Supervisor**: Dr. Priyadharshini S

---

## 🗃️ Dataset

- **Name**: Indiana University Chest X-ray Collection (IU-CXR)
- **Samples**: 3,330 frontal chest X-ray + report pairs
- **Format**: PNG images + free-text reports
- **Text Sections Used**: Findings & Impressions
- **Annotations**: MeSH terms extracted for auxiliary supervision
- **Preprocessing**:
  - Resized to 224×224 or 512×512
  - Normalized pixel values
  - Converted grayscale to RGB
  - Filtered to retain only frontal-view X-rays

---

## 🧠 Methodology

### 🔹 Models Used

#### ✅ ViT + GPT2
- **Encoder**: Vision Transformer (ViT-base-patch16-224-in21k)
- **Decoder**: GPT-2 (openai-community/gpt2)
- Combined via `VisionEncoderDecoderModel`

#### ✅ BLIP-2
- **Vision Encoder**: ViT-Base
- **Q-Former**: 6-layer transformer with learnable queries
- **Language Model**: OPT-2.7B
- Q-Former is the only trainable module; others frozen

#### ✅ GIT
- **Vision Encoder**: CLIP ViT
- **Decoder**: Transformer-based Causal Language Model
- Unified for image-to-text generation

---

### 🔁 Two-Stage Fine-Tuning

#### Stage 1: MeSH-Based Adaptation
- Fine-tune Q-Former using MeSH annotations
- Enhances medical domain understanding

#### Stage 2: Caption-Based Adaptation
- Fine-tune on full radiology reports
- Uses cross-entropy loss for generation accuracy

#### ✅ DINOv2 Integration (Final Architecture)
- Dual Vision Encoders: BLIP-2 + DINOv2
- Dual Q-Formers for feature extraction
- Features fused and decoded into text
- Fine-tuned using **LoRA** for efficiency

---

## ⚙️ Training Configuration

| Model                     | Epochs | Batch Size | LR      | Modules Trained                         |
|--------------------------|--------|------------|---------|------------------------------------------|
| ViT+GPT2                 | 7      | 8          | 5e-4    | VisionEncoderDecoderModel                |
| BLIP-2                   | 25     | 2          | 5e-4    | Q-Former only                            |
| GIT                      | 15     | 8          | 4e-5    | Full model                               |
| BLIP-2 Two-Stage         | 5+15   | 2          | 5e-4/1e-4 | Stage 1: ViT+QF, Stage 2: QF only       |
| BLIP-2 + DINOv2 Two-Stage| 5+15   | 2          | 5e-4/5e-5 | Dual Q-Formers (LoRA), decoder frozen   |

All models were trained with **mixed precision (FP16)** using `torch.amp` and `GradScaler`.

---

## 📈 Evaluation & Results

| Model                     | BLEU-1 | BLEU-4 | ROUGE-L |
|--------------------------|--------|--------|---------|
| ViT+GPT2                 | 0.4955 | 0.2254 | 0.1424  |
| BLIP-2 (Single-stage)    | 0.6579 | 0.3941 | 0.2408  |
| GIT                      | 0.7455 | 0.4627 | 0.2695  |
| BLIP-2 (Two-stage)       | 0.6738 | 0.4087 | 0.2260  |

> 🏆 GIT achieved the best performance across all metrics. Two-stage fine-tuned BLIP-2 showed notable improvements over its single-stage variant.

---

## ✅ Conclusion

This project demonstrates the effectiveness of transformer-based models for generating clinically accurate and context-aware captions for chest X-rays. The use of two-stage fine-tuning significantly improves medical domain alignment. GIT outperformed other models in evaluation metrics, while BLIP-2 with DINOv2 presents a scalable solution for domain adaptation with limited data.

---

## 🔮 Future Work

- Add **interpretability** using Grad-CAM and CRP for clinical trust
- Expand to other modalities: CT, MRI, Ultrasound
- Incorporate **domain-specific LLMs** (e.g., BioGPT, Med-PaLM)
- Deploy as a **web-based decision support tool** for radiologists
- Conduct **clinical validation** with expert feedback

---

## 📚 References

1. Hou et al., *RATCHET: Transformer for Chest X-ray Reporting*, 2021  
2. Indiana University Chest X-ray Collection (IU-CXR)  
3. Chen et al., *VisualGPT*, 2022  
4. Salesforce, *BLIP-2*, 2023  
5. Wang et al., *GIT: Generative Image-to-Text*, 2022  
6. Zhang et al., *SAM-Guided Medical Captioning*, 2024  
7. Wang et al., *METransformer*, 2023  
8. Beddiar et al., *Explainability for Medical Image Captioning*, 2023

---

## 📬 Contact

For queries or collaboration, contact:

**Deep Gandhi**  
📧 deepgandhi017@gmail.com  
🎓 Indian Institute of Information Technology, Kottayam  
