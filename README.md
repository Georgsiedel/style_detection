# Style Transfer Data Augmentation for Robust Object Detection

This repository accompanies the paper **“TBA”**. It allows to validate the positive effect of two style transfer methods for object detection training on benchmark datasets.

---

## 🧭 Overview

- Training of corruption-robust object detectors using the Ultralytics package
- Support and results for COCO and PascalVOC benchmarks as well as Yolo26s and RTDetr models
- Support for **stylization-based augmentation** with
  - The [original AdaIN](https://arxiv.org/abs/1703.06868) as implemented [here](https://arxiv.org/abs/2512.15675), extended with a blending method for arbitrary-resolution images
  - The lightweight, arbitrary-resolution [MicroAST method](https://arxiv.org/abs/2211.15313)

---

## 📂 Repository Structure

- `train.py` – training script  
- `evaluate.py` – evaluation script  
- `runs/detect/` – results and checkpoint folder, automatic naming convention of subfolders
---

## 🎨 Stylization Features

Stylization utilities are included in the `nst` and MicroAST subdirectories.

Stylization with AdaIN ('nst') requires encoded image features from **Painter-by-Numbers**.

Required file in `data/`:
`style_feats_adain_1000.npy`


For exact reproduction, download the 1000 features used here:

- https://zenodo.org/records/16279015

Stylization with MicroAST ('microast') draws style statistics randomly from a precomputed distribution, hence requiring a file "style_distribution.npz".
Use the script precompute_style_distribution.py to obtain the file, here using the train-1 split from [Painter-by-Numbers](https://www.kaggle.com/c/painter-by-numbers).

---

## 🧭 Model Architectures and Data configurations

All training setup is based on the ultralytics package. Refer to their documentation for information on how to adapt models and data.

---

## 📚 Evaluation

Includes robustness on real-world c-corruptions (computed on the fly once and cached in the data repo).
Results saved in a separate val_{} subfolder of runs/detect/

## ✅ Capabilities Summary

- corruption-robust training of Yolo and RTDetr object detectors
- integration of style transfer augmentation
- Ultralytics-based training
- real-world corruption robustness evaluation

---


