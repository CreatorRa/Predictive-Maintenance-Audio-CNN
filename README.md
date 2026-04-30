# Predictive Maintenance (PdM) with Deep Learning

This repository provides a robust, end-to-end Deep Learning pipeline for Acoustic Predictive Maintenance (PdM). By framing mechanical degradation as a spatial pattern recognition problem, this project utilizes a Convolutional Neural Network (CNN) to detect impending pump failures from Log-Mel-Spectrograms. The architecture is explicitly engineered to overcome the two primary challenges of industrial acoustic data: severe minority class imbalance (~11% anomaly rate) and extreme background interference (-6dB Signal-to-Noise Ratio).

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

###  Project Highlights
* **The Challenge:** Detecting mechanical faults in a worst-case scenario: heavy factory background noise (-6dB SNR) and a severe ~11% class imbalance (faults are rare).
* **The Approach:** Raw audio is converted into 2D Log-Mel-Spectrograms. To prevent overfitting, a massive 2.5-million-parameter dense head was replaced with a **Global Average Pooling (GAP)** layer, shrinking the model to ~97k parameters and acting as a structural regularizer.
* **The Results:** Through a robust 12-point Cartesian Grid Search over Learning Rate, Batch Size, and Base Filters, the champion model (`LR=5e-4, BS=64, BF=32`) achieved a **held-out Test F1-Score of 0.866** and an **ROC-AUC of 0.9703**.

---

##  Repository Structure
```text
Predictive-Maintenance-Audio-CNN/
├── src/
│   ├── preprocess.py    # Audio → Log-Mel-Spectrogram conversion
│   ├── dataset.py       # SpectrogramDataset (lazy loading), class weights, splits
│   ├── model.py         # AudioClassifier (4-block CNN with GAP head)
│   ├── train.py         # Adam + weighted CE + ReduceLROnPlateau + early stopping
│   ├── evaluate.py      # F1 / ROC-AUC / confusion matrix, CSV result tracking
│   └── explain.py       # Grad-CAM visualization
├── tests/
│   ├── conftest.py      # sys.path injection for src/ imports
│   ├── test_preprocess.py
│   ├── test_dataset.py
│   └── test_model.py
├── tune.py              # Orchestrator: Cartesian grid search via subprocess
├── requirements.txt     # Python dependencies
└── docs/
    └── tuning_visualizations/   # Output folder for ROC, Confusion Matrices, and Grad-CAM
```

---

## Acquiring the Data (MIMII Dataset)

This project uses the **MIMII (Malfunctioning Industrial Machines Investigation and Inspection)** dataset. Due to file size limits, the data is not included in this repository.

1. Go to the official Zenodo repository: [https://zenodo.org/records/3384388](https://zenodo.org/records/3384388)
2. Download the audio files for the **Pump** machine type at **-6 dB** SNR (`-6_dB_pump.zip`).
3. Extract the downloaded `.zip` file into a directory named `data/raw/` in the root of this project.

*(Note: If you are running this in Colab, see the Colab instructions below for a fast way to download the data directly to your cloud instance).*

---

## How to Run the Code

You can run this project locally in your terminal or entirely in the cloud using Google Colab. 

### Option A: Run directly in Google Colab (Recommended)
Google Colab provides free GPU access, which is highly recommended for running the `tune.py` grid search.

1. **Fork this repository** to your own GitHub account using the "Fork" button at the top right of this page.
2. Open [Google Colab](https://colab.research.google.com/).
3. Open a new Notebook and set your runtime to use a GPU:
   * `Runtime` -> `Change runtime type` -> Hardware accelerator: `T4 GPU`.
4. In the first cell of your notebook, clone your forked repository and move into the directory:
   ```bash
   !git clone [https://github.com/YOUR_GITHUB_USERNAME/Predictive-Maintenance-Audio-CNN.git](https://github.com/YOUR_GITHUB_USERNAME/Predictive-Maintenance-Audio-CNN.git)
   %cd Predictive-Maintenance-Audio-CNN
   ```
5. Install the requirements:
   ```bash
   !pip install -r requirements.txt
   ```
6. Download and unzip the data directly from Zenodo into the Colab environment (this is much faster than uploading it from your computer):
   ```bash
   !mkdir -p data/raw
   !wget [https://zenodo.org/record/3384388/files/-6_dB_pump.zip](https://zenodo.org/record/3384388/files/-6_dB_pump.zip) -P data/raw/
   !unzip -q data/raw/-6_dB_pump.zip -d data/raw/
   ```
7. Fire the engines! Run the full hyperparameter grid search:
   ```bash
   !python tune.py
   ```

### Option B: Run locally in your Terminal
If you have a local GPU or just want to run inference on your CPU, you can run the pipeline directly from your terminal.

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_GITHUB_USERNAME/Predictive-Maintenance-Audio-CNN.git](https://github.com/YOUR_GITHUB_USERNAME/Predictive-Maintenance-Audio-CNN.git)
   cd Predictive-Maintenance-Audio-CNN
   ```
2. **Create and activate a virtual environment:**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Ensure your data is in place:** Verify you have downloaded the MIMII pump dataset from the Zenodo link above and extracted it into the `data/raw/` folder.
5. **Run the grid search / training loop:**
   ```bash
   python tune.py
   ```

---

## Running the Test Suite
This project includes a suite of unit tests to prevent silent regressions (e.g., verifying that audio padding happens at the *end* of a clip, not the beginning). To run the tests, simply execute:
```bash
pytest tests/
```

### One Extra Thing You Need for Your Repository:

For people to clone your repository and run it, you **must** include a `requirements.txt` file in your root folder. Create a new file named `requirements.txt` and paste this into it (these are the standard libraries you've been using based on our conversation):

```text
torch>=2.0.0
torchaudio>=2.0.0
librosa>=0.10.0
numpy>=1.22.0
pandas>=1.5.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
seaborn>=0.12.0
pytest>=7.0.0
```
## Disclaimer 
This pipeline was developed and rigorously evaluated as an academic, scientific proof-of-concept using the MIMII dataset under simulated factory noise conditions (-6dB SNR). It is not intended for immediate plug-and-play deployment in a live, mission-critical industrial environment. Every factory has a unique acoustic profile; deploying this model in a new facility will require fine-tuning to account for Domain Shift (different room reverberations, distinct background machinery, etc.). This software should be used for research and educational purposes, and should never replace multi-sensor safety systems or professional engineering judgment.