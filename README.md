# Speech Emotion Recognition
## Author
Christina Martinez - CS 4337 Computer Vision

A deep learning project to classify emotions from speech audio using CNN and mel-spectrograms.

## Project Overview
- **Goal:** Classify speaker emotions (happy, sad, angry, fearful, neutral, etc.) from audio files
- **Approach:** CNN trained on mel-spectrogram representations of speech audio
- **Course:** CS 4337 001 - Computer Vision

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- Virtual environment (venv)

### Installation
1. Create virtual environment:
```bash
   python -m venv venv
   venv\Scripts\activate
```

2. Install dependencies:
```bash
   pip install numpy pandas scipy librosa soundfile tensorflow scikit-learn matplotlib seaborn jupyter
```

3. Verify setup:
```bash
   python test_setup.py
```

## Project Structure
speech_emotion_recognition/
├── data/
│   ├── raw/                 # Original datasets
│   └── processed/           # Processed files
├── notebooks/               # Jupyter notebooks
├── src/                     # Source code
├── results/                 # Results and visualizations
├── models/                  # Trained models
├── README.md
└── requirements.txt