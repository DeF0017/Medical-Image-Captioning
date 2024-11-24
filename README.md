# Medical Image Captioning

This project, "Medical Image Captioning," aims to develop an advanced AI-based system capable of generating accurate and contextually relevant captions for complex medical images, such as chest X-rays. It integrates state-of-the-art computer vision and natural language processing techniques to enhance diagnostic workflows.

## Authors
- Gandhi Deep Bankim (2021BCS0040)
- Rohit Raj (2021BCS0065)
- Abhishek Raj (2021BCS0141)
- Gaurav Yadav (2021BCS0023)

### Supervisor
- Dr. Priyadharshini S  
  Department of Computer Science and Engineering,  
  Indian Institute of Information Technology, Kottayam, India.

---

## Abstract
This project leverages transformer models like ViT+GPT2, BLIP-2, and GIT to generate clinically accurate captions for medical images. It bridges the gap between vision and language tasks, enhancing radiological workflows and improving diagnostic precision. The approach addresses challenges like limited labeled datasets and the need for interpretability in healthcare applications.

---

## Project Structure
### 1. Data Processing
- **Dataset**: Indiana University Chest X-ray Collection (IU Chest X-ray).
- **Image Preprocessing**:
  - Resizing images to a uniform resolution.
  - Normalization based on model requirements.
  - Retaining only frontal-view X-rays for consistency.
- **Text Preprocessing**:
  - Extracting "Findings" and "Impressions" sections from reports.
  - Tokenizing text using relevant model tokenizers.

### 2. Models
#### **ViT+GPT2**
- Combines Vision Transformer (ViT) and GPT-2.
- Used for multimodal tasks like image captioning.

#### **BLIP-2**
- Integrates a Vision Transformer, Q-Former, and the OPT-2.7B language model.
- Focuses on fine-tuning the Q-Former for better visual-text alignment.

#### **GIT (Generative Image-to-Text Transformer)**
- Employs CLIP for vision encoding and GPT-style transformers for text generation.
- Optimized for efficient image-to-text generation.

### 3. Training
- Mixed precision training with AdamW optimizer.
- Regular checkpoints for saving and evaluating model performance.

---

## Results
The project evaluated models using BLEU and ROUGE metrics:

| Model         | BLEU-1 | BLEU-4 | ROUGE-1 | ROUGE-L |
|---------------|--------|--------|---------|---------|
| **ViT+GPT2**  | 0.4955 | 0.2254 | 0.2634  | 0.1424  |
| **BLIP-2**    | 0.6579 | 0.3941 | 0.3534  | 0.2408  |
| **GIT**       | 0.7455 | 0.4627 | 0.3809  | 0.2695  |

GIT achieved the highest overall performance, demonstrating its capability in generating precise and clinically relevant captions.

---

## Conclusion
This work illustrates the potential of AI-driven systems to revolutionize medical diagnostics by automating image interpretation. Future plans include:
1. Integrating more advanced models for better feature extraction.
2. Experimenting with new training strategies and loss functions.
3. Enhancing model explainability to build trust and transparency in medical AI solutions.

---

## References
Key references include:
1. **RATCHET**: Medical Transformer for Chest X-ray Diagnosis and Reporting.
2. **VisualGPT**: Data-efficient Adaptation of Pretrained Language Models for Image Captioning.
3. **BLIP-2**: Bootstrapped Language-Image Pre-training.

See the full bibliography in the report for additional sources.

---

## How to Run
1. Clone this repository.
2. Install dependencies from `requirements.txt`.
3. Prepare the dataset and place it in the `data/` directory.
4. Run the training script for the desired model using:
   ```bash
   python train.py --model <model_name>

