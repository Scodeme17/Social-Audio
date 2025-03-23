# Quantum-Enhanced Social Audio Sentiment Analysis

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-orange)
![Quantum](https://img.shields.io/badge/Quantum-Pennylane-purple)

A real-time voice sentiment detection system that combines **quantum machine learning** with **Hugging Face NLP** to analyze emotions in social audio conversations (e.g., live chats, podcasts).

## Features
- **Real-Time Voice Processing**: Capture and analyze audio input from a microphone.
- **Hybrid Quantum-Classical Model**: BERT embeddings enhanced with PennyLane quantum circuits.
- **Kaggle Dataset Integration**: Pre-trained on emotion-labeled audio (CREMA-D).
- **Live Sentiment Feedback**: Instantly classify sentiments as **positive**, **negative**, or **neutral**.

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/quantum-social-audio.git
cd quantum-social-audio
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

#### `requirements.txt`
```
transformers==4.30.0
torch==2.0.1
pennylane==0.32.0
sounddevice==0.4.6
scipy==1.10.1
librosa==0.9.2
kaggle==1.5.16
python-dotenv==1.0.0
huggingface_hub==0.15.1
```

### 3. Kaggle API Setup
1. Get your Kaggle API token (`kaggle.json`) from [Kaggle Account Settings](https://www.kaggle.com/settings).
2. Place `kaggle.json` in `~/.kaggle/`:
```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

## Project Directory Structure
```
quantum_social_audio/
├── data/                   # Kaggle dataset
├── models/                # Saved models
├── utils/
│   ├── data_loader.py     # Dataset preprocessing
│   ├── quantum_layer.py   # Quantum circuit
│   └── audio_handler.py   # Real-time voice processing
├── requirements.txt
├── train.py               # Train quantum model
├── live_demo.py           # Real-time voice sentiment detection
└── README.md
```

## Dataset Setup

### 1. Download CREMA-D Dataset
```bash
python utils/data_loader.py
```
- This downloads and extracts the dataset into `data/raw_audio/`.

### 2. Directory Structure
```
data/
├── raw_audio/          # Audio files (e.g., 1001_DFA_ANG_XX.wav)
├── labels.csv          # Emotion-to-sentiment mappings
└── transcripts/        # Auto-generated text transcripts
```

## Model Architecture
- **Hybrid Quantum-BERT**: Combines BERT embeddings with a PennyLane quantum circuit.
  ```python
  class QuantumSentimentModel(nn.Module):
      def __init__(self):
          super().__init__()
          self.bert = BertModel.from_pretrained('bert-base-uncased')
          self.quantum = QuantumLayer()  # PennyLane quantum layer
          self.classifier = nn.Linear(4, 3)  # 3 sentiment classes
  ```

## Training
Train the quantum-enhanced model:
```bash
python train.py
```
- Model weights are saved to `models/quantum_bert.pth`.

## Live Demo
Run real-time sentiment detection from your microphone:
```bash
python live_demo.py
```
**Output**:
```
Listening... Press Ctrl+C to stop.
Transcribed: "This is fantastic!"
Sentiment: positive 🟢
```

## Contributing
1. Fork the repository.
2. Create a branch: `git checkout -b feature/new-feature`.
3. Commit changes: `git commit -m 'Add new feature'`.
4. Push to the branch: `git push origin feature/new-feature`.
5. Submit a pull request.

## License
MIT License. See [LICENSE](LICENSE).

## Acknowledgements
- **Kaggle** for the [CREMA-D Dataset](https://www.kaggle.com/datasets/ejlok1/cremad).
- **Hugging Face** for Whisper ASR and BERT models.
- **PennyLane** for quantum machine learning tools.
