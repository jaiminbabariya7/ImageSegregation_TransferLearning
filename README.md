# Image Classification via Transfer Learning

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?logo=keras)
![Accuracy](https://img.shields.io/badge/Accuracy-95%25+-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

> Image classification and segregation using transfer learning (ResNet50). Fine-tunes pretrained ImageNet weights on a custom dataset, achieving 95%+ accuracy with data augmentation and progressive unfreezing.

## Architecture
```
Raw Images (multi-class dataset)
        ↓
Data Augmentation Pipeline
  ├── RandomFlip, RandomRotation, RandomZoom
  └── Normalisation (ImageNet mean/std)
        ↓
Pretrained Base Model (ResNet50)
  ├── Phase 1: Train only top layers (frozen base)
  └── Phase 2: Fine-tune last N blocks (unfrozen)
        ↓
Custom Classification Head
  ├── GlobalAveragePooling2D
  ├── Dense(256, relu) + Dropout(0.5)
  └── Dense(num_classes, softmax)
        ↓
Model Evaluation + Image Segregation by Predicted Class
```

## Model Performance

| Model | Test accuracy | Best validation accuracy |
|---|---|---|
| ResNet50 (fine-tuned) | 95.4% | 96.3% |

## Key Techniques
- **Transfer learning** — leverage ImageNet pretrained weights
- **Progressive unfreezing** — fine-tune from top layers down
- **Data augmentation** — expand dataset diversity to prevent overfitting
- **Learning rate scheduling** — cosine decay with warm restarts
- **Image segregation** — auto-sort predicted images into class folders

## Setup
```bash
git clone https://github.com/jaiminbabariya7/ImageSegregation_TransferLearning
cd ImageSegregation_TransferLearning
pip install tensorflow keras pillow numpy matplotlib
# Open and run the notebooks in order:
# 1. notebooks/01_data_prep.ipynb
# 2. notebooks/02_transfer_learning.ipynb
```

## Skills Demonstrated
`TensorFlow` · `Keras` · `Transfer Learning` · `CNN` · `Data Augmentation` · `Image Classification` · `Fine-Tuning` · `Python`
