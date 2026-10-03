# CNN Challenge - ITCS 6169 Computer Vision Assignment 1

This repository contains the implementation and experiments for Assignment 1 - The CNN Challenge. The task is to classify scene images into 16 different classes using different CNN-based models.

The experiments are included in `CNN_Challenge_Experiments.ipynb`. I also provided `train.py` and `evaluate.py` so that the selected final model can be trained and evaluated separately.

## Final Model

The final selected model is ConvNeXt-Small initialized with ImageNet-22K pretrained weights and fine-tuned on the provided scene recognition dataset.

Final configuration:

- Model: ConvNeXt-Small
- Pretraining: ImageNet-22K
- Input size: 224 x 224 RGB
- Number of classes: 16
- Total training images provided: 2400
- Training split: 1920 images
- Validation split: 480 images
- Batch size: 32
- Epochs: 20
- Optimizer: Adam
- Learning rate: 0.0001
- Split seed: 0
- Training seed: 42
- MixUp alpha: 0.2
- CutMix alpha: 1.0
- Random Erasing probability: 0.25

The training transformations contains RandomResizedCrop, RandomHorizontalFlip, ColorJitter, ImageNet normalization and Random Erasing. MixUp or CutMix is also applied during training.

The validation and test images are resized to 224 x 224 and normalized using the ImageNet mean and standard deviation.

## Secret Recipe

The main improvements came from combining ImageNet-22K pretrained ConvNeXt-Small with full fine-tuning and stronger data augmentation. RandomResizedCrop, RandomHorizontalFlip, ColorJitter, Random Erasing, MixUp, and CutMix were used during training. The final model was selected based on validation accuracy rather than test accuracy.

## Results

The original Experiment 6 run achieved 97.92% validation accuracy and 97.25% test accuracy.

The same Experiment 6 configuration was rerun for reproducibility and achieved:

- Validation Accuracy: 97.50%
- Test Accuracy: 96.25%
- Test Loss: 0.1543

The submitted `experiment6_best_model.pth` checkpoint is from the reproducibility run.

## Repository Files

- `CNN_Challenge_Experiments.ipynb` - has the experiments performed for the assignment
- `train.py` - standalone training code for the selected final model
- `evaluate.py` - standalone evaluation code for a trained checkpoint
- `requirements.txt` - required Python packages
- `AI_USAGE.md` - AI usage documentation required for the assignment

The final model checkpoint has been provided through Google Drive because the checkpoint file is larger than GitHub's normal file size limit.

## Running the Notebook in Google Colab

The experiments in `CNN_Challenge_Experiments.ipynb` were performed in Google Colab using a GPU.

To run the notebook:

1. Open `CNN_Challenge_Experiments.ipynb` in Google Colab.

2. Enable a GPU from:

   `Runtime -> Change runtime type -> GPU`

3. Download the `train.zip` and `test.zip` files provided with the assignment.

4. Upload both `train.zip` and `test.zip` directly to the Colab `/content/` directory using the Files panel.

5. Run the notebook cells in order from beginning.

The notebook contains the code for extracting both ZIP files. After extraction, the notebook uses the training data from:

`/content/train_data/train`

and the test data from:

`/content/test_data/test`

The notebook should be run in order because later experiment cells use datasets, variables, and settings created in earlier cells.

A GPU is recommended for running the ConvNeXt experiments.

## Running the Standalone Code Locally

The selected model can also be trained and evaluated using `train.py` and `evaluate.py` without running the complete notebook.

Python and Git are required. An internet connection is also required the first time the training script downloads the official ConvNeXt repository and ImageNet-22K pretrained weights.

The local environment used to verify the standalone scripts was:

- Python 3.14.5
- PyTorch 2.11.0
- torchvision 0.26.0
- timm 1.0.30

The required Python packages are listed in `requirements.txt`.

### Local Dataset Preparation

Download the `train.zip` and `test.zip` files provided with the assignment and extract both files.

For the commands below, place the extracted `train` folder and extracted `test` folder in the same main project folder containing `train.py` and `evaluate.py`.

The extracted `train` folder should contain the 16 class folders and a total of 2400 images.

The extracted `test` folder should contain the same 16 class folders and a total of 400 images.

Run the following commands from this main project folder.

### Local Environment Setup

On Windows PowerShell, create a virtual environment:

```powershell
python -m venv .venv
```

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Training with train.py

Before running `train.py`, make sure that:

- the extracted `train` folder is in the same main project folder
- the `train` folder contains all 16 class folders
- the virtual environment is activated
- the packages in `requirements.txt` are installed
- Git is installed
- an internet connection is available if the ConvNeXt repository and pretrained weights have not already been downloaded

Run:

```powershell
python train.py --train-dir train --output reproduced_experiment6.pth
```

The script creates the 80/20 training-validation split used for the selected model:

- Training images: 1920
- Validation images: 480
- Split seed: 0

Training with a GPU is advisable. Training on CPU is possible but it takes a lot of time.

## Evaluating a Newly Trained Model

After `train.py` finishes, keep the extracted `test` folder in the same main project folder.

The newly trained checkpoint can be evaluated using:

```powershell
python evaluate.py --test-dir test --checkpoint reproduced_experiment6.pth
```

The evaluation script checks that the test dataset contains 400 images and 16 classes before evaluating the model.

## Evaluating the Submitted Checkpoint

The submitted final checkpoint is:

`experiment6_best_model.pth`

It is available at the following Google Drive link:

https://drive.google.com/file/d/1l6wonqsqfDbkpnlHCxxpWGUceFlmMQNm/view?usp=sharing

Download `experiment6_best_model.pth` and place it in the same main project folder containing `evaluate.py` and the extracted `test` folder.

Please Make sure the virtual environment is activated and the packages in `requirements.txt` are installed.

Then run:

```powershell
python evaluate.py --test-dir test --checkpoint experiment6_best_model.pth
```

The submitted checkpoint was tested using the standalone evaluation script and produced:

```text
Test Loss: 0.1543
Test Accuracy: 96.25%
```

The validation accuracy stored with this checkpoint is 97.50%.

## Pretrained Model and Weights

The selected model uses the official Meta ConvNeXt implementation and ConvNeXt-Small ImageNet-22K pretrained weights.

Official ConvNeXt repository:

https://github.com/facebookresearch/ConvNeXt

The ImageNet-22K pretrained weights are loaded before the original classification head is replaced with a 16-class classification head. The complete model is then fine-tuned on the assignment training dataset.

The standalone training script automatically downloads the official ConvNeXt repository and pretrained checkpoint when they are not already available locally.

## Reproducing the Final Model

To reproduce the selected Experiment 6 setup, first we have to complete the local environment and dataset preparation described above.

Train the model using:

```powershell
python train.py --train-dir train --output reproduced_experiment6.pth
```

After training is complete, evaluate the reproduced checkpoint using:

```powershell
python evaluate.py --test-dir test --checkpoint reproduced_experiment6.pth
```

The training/validation split uses split seed 0 and contains 1920 training images and 480 validation images. The Experiment 6 training seed is 42.

Model selection is based on validation accuracy, and the model state with the best validation accuracy is restored at the end of training.

To reproduce only the evaluation of the submitted checkpoint, download `experiment6_best_model.pth` from the Google Drive link above, place it in the main project folder, and run:

```powershell
python evaluate.py --test-dir test --checkpoint experiment6_best_model.pth
```
