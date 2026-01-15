# Speech Emotion Recognition

**Author:** Christina Martinez

A deep learning project to classify emotions from speech audio using CNN and mel-spectrograms.

## Project Overview

* **Goal:** Classify speaker emotions (anger, disgust, fear, happiness, sadness, neutral) from audio files
* **Approach:** CNN trained on mel-spectrogram representations of speech audio

## Setup Instructions

### Prerequisites

* Python 3.8 or higher
* Virtual environment (venv)
* pip (Python package manager)

### Installation

1. **Create and activate virtual environment:**
   ```
   python -m venv venv
   venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```
   
   Or manually install:
   ```
   pip install numpy pandas scipy librosa soundfile tensorflow scikit-learn matplotlib seaborn jupyter
   ```

3. **Verify setup:**
   ```
   python -c "import tensorflow; print(f'TensorFlow version: {tensorflow.__version__}')"
   ```

## Getting Started

### Step 1: Download the Dataset

1. Go to: https://www.kaggle.com/datasets/ejlok1/cremad/
2. Download the CREMA-D dataset (7,442 audio files, 6 emotions)
3. Extract to: `data/raw/CREMA-D/`

Your folder structure should look like:
```
data/raw/CREMA-D/
├── 1001_DFA_ANG_XX.wav
├── 1001_DFA_DIS_XX.wav
├── 1001_DFA_FEA_XX.wav
... (more audio files)
```

### Step 2: Explore the Data

1. Start Jupyter:
   ```
   jupyter notebook
   ```

2. Open and run: `01_data_exploration.ipynb`
   * Loads audio files
   * Visualizes waveforms and mel-spectrograms
   * Shows emotion distribution

### Step 3: Train the Model

1. Open and run: `02_model_training.ipynb`
   * Preprocesses all audio files
   * Builds the CNN model
   * Trains for ~31 epochs
   * Evaluates on test set
   * Saves model as: `models/best_model.h5`

**Training Time:**
* CPU: ~2-7 hours
* GPU: ~30 minutes to 1 hour

## Model Details

### Dataset
* **Name:** CREMA-D
* **Total Files:** 7,442 audio files
* **Emotions:** Anger, Disgust, Fear, Happiness, Sadness, Neutral
* **Split:** 70% training, 15% validation, 15% test

### Architecture
* **Input:** Mel-spectrograms (128 × 87 pixels)
* **Three convolutional blocks:**
  * Block 1: 32 filters → MaxPool → Dropout(0.3)
  * Block 2: 64 filters → MaxPool → Dropout(0.3)
  * Block 3: 128 filters → MaxPool → Dropout(0.3)
* **Fully Connected:** Dense(256) → Dense(128) → Dense(6 emotions)
* **Optimizer:** Adam (lr=0.001)
* **Early Stopping:** Yes (prevents overfitting)

### Results
* **Test Accuracy:** 36%
* **Best Emotion:** Anger (50% precision)
* **Worst Emotion:** Neutral (20% precision)

## Project Structure

```
speech-emotion-recognition/
├── data/
│   ├── raw/                    # Original CREMA-D dataset
│   └── processed/              # Processed mel-spectrograms (generated)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb      # Data analysis
│   └── 02_model_training.ipynb        # Model training
│
├── models/
│   └── best_model.h5           # Trained model (generated)
│
├── results/
│   ├── training_history.png    # Loss/accuracy curves
│   ├── confusion_matrix.png    # Per-class confusion
│   └── per_class_metrics.csv   # Detailed metrics
│
├── Speech_Emotion_Recognition_Report.pdf  # Full report
├── README.md                   # This file
└── requirements.txt            # Dependencies
```

## Model - Important Note

The trained model (`best_model.h5`) is not included in the repository due to file size limitations.

To get the trained model:

1. Follow steps in "Getting Started" above
2. Run `02_model_training.ipynb`
3. Model automatically saves as: `models/best_model.h5`

## Using the Trained Model

Once you have `models/best_model.h5`, you can use it:

```python
from tensorflow.keras.models import load_model
import librosa
import numpy as np

# Load the model
model = load_model('models/best_model.h5')

# Load an audio file
audio_path = 'path/to/audio.wav'
y, sr = librosa.load(audio_path, sr=22050)

# Convert to mel-spectrogram
mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

# Normalize
mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
mel_spec_db = np.expand_dims(mel_spec_db, axis=(0, -1))

# Predict emotion
predictions = model.predict(mel_spec_db)
emotions = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Neutral']
emotion = emotions[np.argmax(predictions)]
print(f"Predicted emotion: {emotion}")
```

## Troubleshooting

### Problem: "FileNotFoundError: CREMA-D not found"
**Solution:** Make sure you downloaded the dataset and extracted it to `data/raw/CREMA-D/`

### Problem: "ModuleNotFoundError: No module named 'tensorflow'"
**Solution:** Run `pip install -r requirements.txt` again

### Problem: "Out of memory" during training
**Solution:** Open `02_model_training.ipynb` and reduce batch size from 32 to 16

### Problem: Training is very slow
**Solution:** Make sure you're using GPU if available

## Key Files

| File | Purpose |
|------|---------|
| `01_data_exploration.ipynb` | Load, explore, visualize CREMA-D dataset |
| `02_model_training.ipynb` | Build, train, evaluate CNN model |
| `Speech_Emotion_Recognition_Report.pdf` | Full technical report |
| `requirements.txt` | Python package dependencies |

## References

* Cao, H., et al. (2014). "CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset"
* Kingma, D.P., & Ba, J. (2015). "Adam: A Method for Stochastic Optimization"
* Ioffe, S., & Szegedy, C. (2015). "Batch Normalization: Accelerating Deep Network Training"
* McFee, B., et al. (2015). "librosa: Audio and Music Signal Analysis in Python"


