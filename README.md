# Image Classification via Transfer Learning (Core vs Well Logs)

![Python](https://img.shields.io/badge/Python-3-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?logo=keras)
![Test accuracy](https://img.shields.io/badge/Test%20accuracy-95.4%25-brightgreen)

> *Postgraduate project, St. Clair College.*

Binary image classifier that separates **core sample images** from **well log images** using **transfer learning with ResNet50** (ImageNet weights) in TensorFlow/Keras. The raw dataset was small (tens of images per class), so the project focuses on preprocessing and augmentation to build a usable training set, then trains a classification head on a frozen ResNet50 base.

## Pipeline
```
Raw images (Core, Well Log)
        ↓
data_preprocessing.ipynb
  ├── Review image characteristics (sizes, channels)
  ├── Convert to grayscale and resize to 256×256 (OpenCV, Pillow)
  ├── Augment with imgaug: horizontal flip, additive Gaussian noise,
  │   Gaussian blur on low-entropy / low-contrast images
  └── Save 400 augmented images per class → Augmented Dataset/ (800 total)
        ↓
cnn_tl_res.ipynb
  ├── Shuffle and split per class: 35% train / 35% validation / 30% test
  │   → 568 train, 571 validation, 522 test images (Split Data/)
  ├── Training-time augmentation (ImageDataGenerator): rescale, rotation,
  │   width/height shift, shear, zoom, horizontal flip
  ├── Model: ResNet50 (ImageNet, include_top=False, base layers frozen)
  │     → GlobalAveragePooling2D
  │     → Dense(1024, ReLU, L2=0.1) → Dropout(0.8)
  │     → Dense(2, softmax)
  ├── Train: Adam, categorical cross-entropy, 10 epochs, batch size 32,
  │   TensorBoard logging (logs/)
  └── Evaluate on the held-out test set; confusion matrix and
      accuracy / loss curves
```

## Results

| Model | Test accuracy (`model.evaluate`) | Best validation accuracy | Final validation accuracy |
|---|---|---|---|
| ResNet50, frozen base + trained head | **95.4%** | 96.3% (epoch 9) | 89.7% (epoch 10) |

## Known issues and next steps
- **Classification report cell:** the per-class report in `cnn_tl_res.ipynb` shows ~50% because the test generator shuffles images (`shuffle=True` by default), so predictions and `test_generator.classes` are not in the same order. The fix is `shuffle=False` on the test generator; the 95.4% figure comes from `model.evaluate`, which is not affected.
- **Possible leakage:** augmented images are created before the train/validation/test split, so variants of the same original image can land in different splits, which can make scores optimistic. Splitting the raw images first, then augmenting only the training set, would give a cleaner estimate.
- **Fine-tuning:** the ResNet50 base is kept frozen; unfreezing the top blocks with a low learning rate is a natural next experiment.
- Histogram equalization is computed in `preprocess()` but the resized (not equalized) image is what gets saved.

## Repository structure
```
├── data_preprocessing.ipynb   # image review, grayscale + resize, augmentation
├── cnn_tl_res.ipynb           # split, ResNet50 transfer learning, evaluation
├── Dataset/                   # raw images (Core, Well Log)
├── Augmented Dataset/         # 400 augmented images per class
├── Split Data/                # Train / Valid / Test folders
└── logs/                      # TensorBoard logs
```

## Setup
```bash
git clone https://github.com/jaiminbabariya7/ImageSegregation_TransferLearning
cd ImageSegregation_TransferLearning
pip install tensorflow opencv-python pillow scikit-image imgaug scikit-learn matplotlib numpy
```
Open the notebooks in order (`data_preprocessing.ipynb`, then `cnn_tl_res.ipynb`) and update the hard-coded dataset paths at the top of each notebook to your local folders.

## Skills demonstrated
`TensorFlow` · `Keras` · `Transfer Learning` · `ResNet50` · `OpenCV` · `imgaug` · `Data Augmentation` · `Image Classification` · `TensorBoard` · `scikit-learn`
