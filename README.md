# Emotion Detection (Experimental)

> 🚧 This project is still under development and experimental. Model performance is not final, and the dataset structure is still being tuned.

## 📌 Overview

This project aims to perform **emotion detection** from facial images using deep learning techniques. It leverages **transfer learning** with `ResNet50` (and optionally VGG) as the backbone model. The input images are resized to 48x48 pixels and passed through an augmented image pipeline before training.

## 💡 Features

- ✅ Image classification into 7 emotion classes: `angry`, `fear`, `happy`, `surprise`, `disgust`, `neutral`, `sad`
- ✅ Transfer learning using `ResNet50`
- ✅ Augmented training pipeline for better generalization
- ✅ Visual inference output with predicted class and probability
- ⚠️ Still under testing; dataset size, balance, and labeling might affect performance

## 🛠 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

🚀 Training the Model
Run main.py to start training using ResNet50:

The script will:
- Load images from train/ and test/
- Apply image augmentation
- Train the model and save it as emotion_detection_resnet50.h5

📌 Notes
Make sure your dataset is arranged in subfolders under train/ and test/, each representing one class (e.g., train/happy, train/sad, etc.)
The model might be unstable or inaccurate due to limited data or training duration.



