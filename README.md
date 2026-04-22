🩻 Chest X-Ray Classification using Vision Transformers (ViT & Swin)

This project implements a deep learning pipeline for chest X-ray image classification using two state-of-the-art Vision Transformer architectures:

Vision Transformer (ViT)
Swin Transformer

It includes:

Training pipeline with class balancing
Evaluation with Accuracy and F1-score
Demo script with heatmap visualization
Side-by-side comparison of models
📌 Features
✅ Training ViT and Swin Transformer models
✅ Weighted sampling for class imbalance handling
✅ Early stopping based on validation F1-score
✅ Model checkpoint saving
✅ Simple heatmap visualization overlay
✅ Side-by-side comparison (Original vs ViT vs Swin)
📁 Project Structure
.
├── train_and_eval.py          # Training pipeline (ViT & Swin)
├── demo.py                    # Inference + visualization
├── vit.pth                    # Saved ViT model weights
├── swin.pth                   # Saved Swin model weights
└── result.png                 # Output visualization image

🧠 Models

🔹 Vision Transformer (ViT)
Pretrained vit_b_16 from torchvision, fine-tuned for 4-class classification.
🔹 Swin Transformer

Pretrained swin_t model adapted for medical image classification.

Both models:
Use frozen backbone (only classifier head is trained)
Output: 4 classes

📊 Dataset
Expected dataset structure:
dataset/
├── class_0/
├── class_1/
├── class_2/
└── class_3/
Each folder contains chest X-ray images (.png, .jpg, .jpeg).


🚀 Training

Run training for both models:

python train_and_eval.py  

Models will be saved as:

vit.pth
swin.pth

🔍 Inference

Run prediction on a single image:

python demo.py --image path/to/xray.jpg

Output:
Class prediction (ViT & Swin)
Confidence score
Heatmap visualization
Saved result image:
result.png
Heatmap is simplified (not Grad-CAM) for lightweight visualization.
Backbone layers are frozen to speed up training.

MIT License
