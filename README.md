# Image Classification - Model Training

This repository contains the training pipeline for an **Intel Image classification** system using deep learning. It includes implementations in **both TensorFlow and PyTorch**, focusing on classifying images into categories such as buildings, forest, mountains etc.

---

## Objective

To build and compare robust CNN models using TensorFlow and PyTorch for classification using image data. This forms part of a portfolio-ready end-to-end ML project including model training, evaluation, and web deployment.

---

## Components

### TensorFlow Training

* Data Augmentation: `RandomFlip`, `RandomZoom`, `RandomRotation`, etc.
* CNN Layers + BatchNorm + Dropout
* `GlobalAveragePooling2D` + Dense layers
* Regularization with `L1L2`
* `EarlyStopping` and `ModelCheckpoint` callbacks

### PyTorch Training

* CNN architecture with similar depth and regularization
* `AdaptiveAvgPool2d` instead of GAP
* `Dropout` and `BatchNorm` used
* Best model saved using `torch.save`

---

## Repository Structure

```
.
├── TFTrain.py               # TensorFlow training script
├── TFTest.py                # TensorFlow testing script
├── TorchTrain.py            # PyTorch training script
├── TorchTest.py             # PyTorch testing script
├── reports/                 # Confusion matrices and evaluation visuals
├── Data/
│   ├── train/
│   ├── val/
│   └── test/
├── requirements.txt         # Dependencies
├── README.md                
└── LICENSE                  # MIT License

```

---

## Evaluation Results

* **TensorFlow**

  * Best Training Accuracy: **84.25%**, Training Loss: **0.5474**, Validation Accuracy: **83.0%**, Validation Loss: **0.5870**

* **PyTorch**

  * Best Training Accuracy: **87.01%**, Training Loss: **0.3631**, Validation Accuracy: **87.1%**, Validation Loss: **0.3473**

### Evaluation Metrics (Seaborn + Matplotlib):

* Classification Reports
* Confusion Matrices

#### TensorFlow Model

<div align="center">
  <img src="reports/TF.png" alt="TensorFlow Confusion Matrix"/>
</div>

#### PyTorch Model

<div align="center">
  <img src="reports/Torch.png" alt="PyTorch Confusion Matrix"/>
</div>

---

## How to Train

### TensorFlow

```bash
python TFTrain.py
python TFTest.py
```

### PyTorch

```bash
python TorchTrain.py
python TorchTest.py
```

Make sure your dataset is placed in the `Data/` directory with subfolders:

```
Data/
├── train/
├── val/
└── test/
```
* Note: If the Data doesn't have Validation or Test Data Sets, you can divide training Data Set using **Split.py** uploaded within **Data** folder. Rename the corresponding datasets in the code. 

#### Example:
```bash
cd Data
python Split.py ./path # Replace the ./path with your training dataset. Note that you shouldn't name the path as train, val or test to run the code.
cd ..
```
---

## Dependencies

```bash
pip install -r requirements.txt
```

* TensorFlow
* PyTorch
* matplotlib, seaborn, scikit-learn

GPU acceleration is recommended if available.

---

## Highlights

* Dual framework training (TF + Torch)
* Data augmentation and regularization
* Clean modular design for reproducibility
* Evaluation visualizations for clarity
* Ready for deployment in Flask app

---

## License

MIT License – see `LICENSE` for full details.

---

## Author

Created by **JAGAN-JGUN** for professional portfolio showcasing.

---

## Contact

For queries or collaboration:

* GitHub: [JAGAN-JGUN](https://github.com/JAGAN-JGUN)
* Email: [jaganjgun008@gmail.com](mailto:jaganjgun008@gmail.com)
