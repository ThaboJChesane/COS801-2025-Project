# COS801 Project 2025 - BioNeorons
# Medical Image Classification Using Chest X-rays  
### CNN, CNN-Logits, and LiteDenseNet-BC

## [Chest X-Ray COVID-19 & Pneumonia Dataset](https://www.kaggle.com/datasets/prashant268/chest-xray-covid19-pneumonia/data)


This repository contains the full implementation, training pipeline, evaluation scripts, and analysis for a medical image classification project focused on detecting **COVID-19**, **Normal**, and **Pneumonia** cases from chest X-ray images.

The project implements three deep learning models of increasing complexity:

1. **Standard CNN (Softmax output)**
2. **CNN with Logits Output**
3. **LiteDenseNet-BC** – a custom lightweight DenseNet variant that achieved the best performance.

The work is based on the dataset from Kaggle:  
**Chest X-ray (COVID-19 & Pneumonia)**  
and includes comprehensive analysis, Grad-CAM explainability, ROC curves, confidence calibration, and performance comparisons against related published work.

---

## 🚀 Project Overview

The goal of this project is to evaluate how different CNN architectures perform on chest X-ray classification and determine whether architectural innovations (logits-based learning or dense connectivity) improve accuracy, generalisation, and calibration in a medical imaging context.

The implemented models are:

- **Standard CNN** — a simple baseline architecture.
- **CNN_Logits** — identical to the Standard CNN except the final softmax layer is replaced with raw logits for improved numerical stability.
- **LiteDenseNet-BC** — a custom compact DenseNet variant inspired by DenseNet-BC, designed for efficiency and strong feature reuse.

LiteDenseNet-BC achieved **state-of-the-art performance** on the dataset, surpassing both the baseline models in this repository and models reported in related literature.


## 📊 Results Summary

### **Model Performance**

| Model             | Accuracy | Macro F1 | Weighted F1 |
|------------------|----------|----------|-------------|
| Standard CNN      | 0.8020   | 0.7849   | 0.8098      |
| CNN_Logits        | 0.9224   | 0.9219   | 0.9245      |
| LiteDenseNet-BC   | **0.9612** | **0.9590** | **0.9613** |

---

## 📈 Visual Outputs

This project includes extensive visualization tools:

- Confusion matrices  
- ROC curves (one-vs-rest)  
- Training/validation loss curves  
- Confidence distribution histograms  
- Grad-CAM heatmaps & overlays  
- Misclassification visual inspections  

All visuals are generated automatically in the notebooks.

---

## 🧠 Explainability (Grad-CAM)

Grad-CAM heatmaps highlight the lung regions most relevant to the model’s prediction.  
LiteDenseNet-BC consistently attends to clinically meaningful structures, improving trust and interpretability.

## 🔬 Methods & Models

### **1. Standard CNN**
- 4 convolutional blocks: 32 → 64 → 128 → 256 filters  
- BatchNorm + ReLU + MaxPool  
- Dropout for regularisation  
- Dense classifier with softmax output  

### **2. CNN_Logits**
- Same architecture as Standard CNN  
- Final activation removed → outputs raw logits  
- Trained with `SparseCategoricalCrossentropy(from_logits=True)`  
- Improved calibration & stability  

### **3. LiteDenseNet-BC**
- Dense connectivity  
- Bottleneck layers (1×1 conv)  
- Compression layers  
- Lightweight growth rate for efficiency  
- Strong performance across all metrics  


