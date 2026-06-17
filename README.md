# OptiView DX: Retinal Vessel Segmentation & Diabetic Retinopathy Detection

OptiView DX is a deep learning-based system for automated retinal vessel segmentation and early Diabetic Retinopathy (DR) analysis from fundus images. The project uses an enhanced **Attention ResUNet** architecture to accurately identify retinal blood vessels, including fine capillaries, supporting efficient and reliable clinical screening.

## 🚀 Features

* Automated retinal vessel segmentation
* Early DR-related retinal abnormality analysis
* Attention-based feature learning for improved vessel detection
* Image preprocessing using CLAHE, gamma correction, and normalization
* Visualization of segmented vessels on fundus images

## 🏗️ Model Architecture

### Attention ResUNet

* **Residual Blocks:** Improve feature learning and gradient flow.
* **Attention Gates:** Focus on vessel regions while suppressing background noise.
* **U-Net Structure:** Combines encoder-decoder architecture with skip connections for precise segmentation.

## 📊 Dataset

* **FIVES** (Fundus Image Vessel Segmentation Dataset)

## 🛠️ Tech Stack

* Python 3.12
* TensorFlow / Keras
* OpenCV
* Scikit-Image
* NumPy
* Matplotlib
* Scikit-Learn

## 📈 Evaluation Metrics

* Dice Coefficient
* IoU (Jaccard Index)
* Accuracy
* Precision
* Recall
* Specificity
* F1-Score
* AUC-ROC




