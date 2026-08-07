# UrbanSound8K Environmental Sound Classification

This project trains and evaluates models that classify short urban audio clips into 10 environmental sound categories (such as car horn, dog bark, siren, drilling, and street music) using the UrbanSound8K dataset. It solves the problem of automatically recognizing what kind of sound is present in a short audio recording, without a human having to listen to and label it manually. The notebook covers the full pipeline: downloading the dataset, exploring it, extracting audio features, training a convolutional neural network, evaluating it against a classic machine learning baseline, and running 10-fold cross-validation. It is intended for people learning or prototyping audio classification with PyTorch, students working with the UrbanSound8K dataset, or anyone who needs a reference pipeline for spectrogram-based sound classification.

## Features

- Automatic download of the UrbanSound8K dataset via `kagglehub`, with automatic detection of the folder layout (`audio/fold1...` or `fold1...` directly under the dataset root).
- Exploratory data analysis: class distribution, samples per fold, clip duration statistics, and mel-spectrogram visualization for one example per class.
- Two parallel feature representations extracted from each audio clip:
  - A normalized, fixed-size log-mel spectrogram (2D) used as input to a CNN.
  - A 40-dimensional MFCC vector averaged over time, used as input to a Random Forest baseline.
- Feature caching to disk (`.npy` files) so that feature extraction only has to run once.
- A strict fold-based train/validation/test split (folds 1-8 for training, fold 9 for validation, fold 10 for testing) to avoid data leakage between clips that come from the same original recording.
- A custom PyTorch `Dataset` (`UrbanSoundDataset`) and a small CNN (`AudioCNN`) with three convolutional blocks, batch normalization, max pooling, and dropout.
- A training loop with learning-rate scheduling (`ReduceLROnPlateau`), best-model checkpointing, and loss/accuracy curves.
- Test-set evaluation with a classification report and confusion matrix.
- An optional full 10-fold cross-validation loop that reproduces the standard UrbanSound8K evaluation protocol.
- A Random Forest baseline trained on the averaged MFCC features, for comparison against the CNN.
- A ready-to-use (commented-out) inference block for running the trained model on a new, unseen audio file.

## Project Structure

The project is a single Jupyter notebook (`UrbanSound8K.ipynb`) organized into numbered sections:

1. **Imports**: loads all required libraries and sets the random seed (`SEED = 42`) and compute device (`DEVICE`).
2. **Dataset Download**: retrieves the UrbanSound8K dataset with `kagglehub.dataset_download` and resolves the audio folder location.
3. **Exploratory Data Analysis (EDA)**: inspects class counts, fold counts, missing values, duplicates, clip durations, and plots example mel-spectrograms.
4. **Feature Engineering**: defines `load_audio`, `extract_mel_spectrogram`, and `extract_mfcc_mean`, then runs extraction over the whole dataset and caches the results.
5. **Train/Validation/Test Split**: builds boolean masks over folds and wraps the arrays in `UrbanSoundDataset` and `DataLoader` objects.
6. **Model Definition**: defines the `AudioCNN` class.
7. **Training**: defines `train_one_epoch` and `evaluate`, then runs the main training loop for `EPOCHS = 30` and saves the best checkpoint to `best_audio_cnn.pt`.
8. **Test-Set Evaluation**: loads the best checkpoint and reports accuracy, a classification report, and a confusion matrix on fold 10.
9. **Full 10-Fold Cross-Validation (optional)**: retrains a fresh `AudioCNN` for each of the 10 folds and reports mean accuracy.
10. **Baseline: Random Forest on MFCC Features**: trains a `RandomForestClassifier` on the MFCC vectors and compares its accuracy to the CNN.
11. **Inference on a New Audio File**: a template block (commented out) showing how to run a trained model on a single new `.wav` file.

At runtime, the notebook also creates a `feature_cache/` folder containing `mel_specs.npy`, `mfcc_features.npy`, `labels.npy`, and `folds.npy`, and saves the best CNN weights to `best_audio_cnn.pt` in the working directory.

## Requirements

- Python 3.
- **A GPU is required to train the CNN in a reasonable amount of time.** The notebook automatically selects `cuda` if a compatible GPU is available and falls back to `cpu` otherwise:

```python
  DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

  Training on CPU is technically possible but will be substantially slower for 30 epochs over roughly 8,700 audio clips, and is not the intended way to run this project. Make sure you have a CUDA-capable GPU with an up-to-date NVIDIA driver, and a PyTorch build that includes CUDA support, before running the training and cross-validation sections. If you are running this notebook on a hosted platform (for example a cloud notebook service), select a GPU-backed runtime before executing the cells.

## Installation

1. Clone or download the project so that `UrbanSound8K.ipynb` is available locally.
2. Install the required Python packages:

```bash
   pip install kagglehub librosa torch torchvision tqdm soundfile numpy pandas matplotlib seaborn scikit-learn
```

   This matches the install line already present in the first cell of the notebook:

```python
   #!pip install kagglehub librosa torch torchvision tqdm soundfile
```

3. Install a CUDA-enabled build of PyTorch that matches your GPU driver. Check the official PyTorch installation instructions for the correct command for your CUDA version, since the plain `pip install torch` command does not always include GPU support.
4. Make sure you have access to Kaggle (the dataset is downloaded automatically through `kagglehub`, which may prompt for Kaggle credentials the first time it runs).

## Usage

Open `UrbanSound8K.ipynb` in Jupyter (or a compatible notebook environment) and run the cells in order:

```bash
jupyter notebook UrbanSound8K.ipynb
```

Running the notebook top to bottom will:

1. Download the UrbanSound8K dataset and load `UrbanSound8K.csv`.
2. Produce EDA plots (class distribution, fold distribution, duration histogram, example spectrograms).
3. Extract mel-spectrogram and MFCC features for every clip, caching them under `feature_cache/`.
4. Build the train (folds 1-8), validation (fold 9), and test (fold 10) splits.
5. Train the `AudioCNN` model for 30 epochs, saving the best checkpoint to `best_audio_cnn.pt`:

```python
   EPOCHS = 30
   LR = 1e-3
   MODEL_PATH = "best_audio_cnn.pt"
```

6. Evaluate the best checkpoint on the test set (fold 10) and display a classification report and confusion matrix.
7. Optionally run full 10-fold cross-validation to get the mean accuracy across all folds (this repeats training 10 times and costs roughly 10x a single run).
8. Train and evaluate a `RandomForestClassifier` baseline on the averaged MFCC features, and compare its accuracy to the CNN.

To classify a new, unseen `.wav` file after training, uncomment and adapt the inference block at the end of the notebook:

```python
NEW_FILE_PATH = "./some_new_sound.wav"
y = load_audio(NEW_FILE_PATH)
mel = extract_mel_spectrogram(y)
x = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

model.eval()
with torch.no_grad():
    logits = model(x)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))

print(f"Predicted class: {class_names[pred_idx]} (probability: {probs[pred_idx]:.3f})")
```

## Configuration

The notebook uses in-code constants instead of a separate configuration file or environment variables. The main ones are set near the top of each relevant section:

| Constant | Value | Purpose |
|---|---|---|
| `SEED` | `42` | Random seed for NumPy and PyTorch, for reproducibility. |
| `SAMPLE_RATE` | `22050` | Sample rate (Hz) used to load and resample audio. |
| `DURATION` | `4.0` | Fixed clip length (seconds) that every audio sample is padded or truncated to. |
| `N_MELS` | `64` | Number of mel bands used for the spectrogram. |
| `N_FFT` | `1024` | FFT window size used to compute the spectrogram. |
| `HOP_LENGTH` | `512` | Hop length between successive spectrogram frames. |
| `BATCH_SIZE` | `32` | Batch size used by all `DataLoader` objects. |
| `EPOCHS` | `30` | Number of training epochs for the main CNN run. |
| `LR` | `1e-3` | Learning rate for the Adam optimizer. |
| `MODEL_PATH` | `"best_audio_cnn.pt"` | File path where the best-performing model checkpoint is saved. |
| `CACHE_DIR` | `"./feature_cache"` | Directory where extracted features are cached as `.npy` files. |

To change training behavior (for example, the number of epochs or the learning rate), edit these constants directly in the corresponding notebook cell before running it.

## Technologies Used

- **Language**: Python 3.
- **Deep learning**: PyTorch (`torch`, `torch.nn`, `torch.optim`, `torch.utils.data`).
- **Classic machine learning**: scikit-learn (`RandomForestClassifier`, `classification_report`, `confusion_matrix`, `accuracy_score`).
- **Audio processing**: librosa (loading audio, computing mel-spectrograms and MFCCs), soundfile.
- **Data handling**: NumPy, pandas.
- **Visualization**: Matplotlib, Seaborn, `librosa.display`.
- **Dataset access**: kagglehub (for downloading UrbanSound8K).
- **Utilities**: tqdm (progress bars during feature extraction).

## Author

*M. E. Petra Bošković*
