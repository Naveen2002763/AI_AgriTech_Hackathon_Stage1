# AI_AgriTech_Hackathon_Stage1
# AI for AgriTech Hackathon – Stage 1 🌾

## 🔍 Problem Statement
Build a CNN-based plant segmentation/classification model to identify **crop vs weed** from agricultural field images.
## Dataset

Due to size constraints, only a sample of the dataset is included in this repo.  
The full dataset (1,300 images and labels) can be downloaded from https://www.kaggle.com/datasets/ravirajsinh45/crop-and-weed-detection-data-with-bounding-boxes.

📁 Note: Due to GitHub's file size constraints, only 3 sample images and corresponding labels have been included in the `data/` folder for structure reference.  
📦 The full dataset (1,300+ images) can be downloaded from [Kaggle here](https://www.kaggle.com/datasets/ravirajsinh45/crop-and-weed-detection-data-with-bounding-boxes).

📂 Project Structure:
- `data/`: Sample structure with a few training/validation images and labels
- `src/yolov5/`: Custom training, validation, and detection scripts
- `src/yolov5/runs/`: Output visualizations (e.g., best.png, confusion matrix)


## 🧠 Model Used
YOLOv5 (custom-trained on labeled crop/weed dataset)

## 📁 Project Structure
- `data/` – Contains dataset (`images`, `labels`, `data.yaml`, `classes.txt`)
- `src/yolov5/` – Includes training, validation, detection, evaluation scripts
- `runs/` – YOLOv5 output: weights, predictions, confusion matrix

## 🧪 Evaluation Results
- **Accuracy**: 93%
- **Precision**: Crop (0.97), Weed (0.89)
- **Recall**: Crop (0.91), Weed (0.96)
- **F1 Score**: 0.93

## 📷 Output
![Confusion Matrix](src/yolov5/runs/val/exp3/confusion_matrix.png)

## 🚀 How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model (Optional)
python train.py --img 640 --batch 16 --epochs 50 --data ../../data/data.yaml --weights yolov5s.pt

# Validate on val set
python val.py --weights runs/train/yolo_crop_weed5/weights/best.pt --data ../../data/data.yaml --task val --save-txt

# Run custom evaluation
python evaluate_custom_metrics.py --pred runs/val/exp3/labels --gt ../../data/labels/val --names ../../data/classes.txt
