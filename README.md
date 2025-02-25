# CoughAnalyzer

## Usage
Download data with
`git clone https://github.com/roneelsharan/CoughSegmentation`

Check `notebooks/EDA.ipynb` for EDA

To add background noise to the audio, run `add_noise.py`, which will populate a new directory called "data_w_noise" that adds background noise to the original wav files.

Run `python src/data/gen_dataset.py` to generate the dataset with format as
![](figures/data_format.jpg)

Requirements: `pip install -r requirements.txt`


## Yolo Cough Detection Results
Validation Set Labels:
![](src/yolo/runs/detect/val/val_batch0_labels.jpg)

Validation Set Predictions (conf=0.3, iou=0.4):
![](src/yolo/runs/detect/val/val_batch0_pred.jpg)