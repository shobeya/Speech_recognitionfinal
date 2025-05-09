Accent-Aware Speech Recognition System
This project implements a machine learning-based Accent Recognition System designed to classify the accent or language of a given audio sample using MFCC (Mel-Frequency Cepstral Coefficient) features. It helps enhance speech recognition systems by making them more robust to speaker accents, an essential aspect of building inclusive and diverse voice-driven applications.

📌 Project Description
Accent and language variability can significantly affect the performance of speech recognition systems. This project addresses that by training a classifier on audio features to predict the accent or language spoken. It leverages preprocessed MFCC features extracted from a speech dataset and trains models such as Random Forest to classify speech samples based on their acoustic properties.

🧠 Key Features
Extracts MFCC features from audio data

Trains a supervised machine learning model to classify accents

Provides an interactive UI using Tkinter for real-time predictions

Achieves up to 78.79% accuracy on test data

🗃️ Dataset
We used the Accent MFCC Data CSV file, which contains:

Columns: X1, X2, ..., Xn — MFCC features

language: The label denoting the accent or language class

Note: Features were pre-extracted using standard MFCC processing pipelines in tools like librosa.

🧰 Tech Stack
Python 3.12

Libraries: pandas, numpy, scikit-learn, tkinter, librosa

Model: Random Forest Classifier

UI: Tkinter-based GUI for user interaction and prediction

📊 Model Performance
Metric	Value
Accuracy	78.79%
Classifier	Random Forest

🚀 How to Run
Clone or download this repository.

Make sure your working directory includes:

accent-mfcc-data-1.csv inside a data/ folder

train_model.py – to train the classifier

app.py – to run the GUI

Run training:

bash
Copy code
python train_model.py
Launch the GUI:

bash
Copy code
python app.py
📁 Project Structure
graphql
Copy code
Accent-Aware-System/
│
├── data/
│   └── accent-mfcc-data-1.csv
│
├── train_model.py       # Training script
├── app.py               # Tkinter GUI for predictions
├── model.pkl            # Trained model file (generated after training)
└── README.md
📌 Use Case
This system can be integrated into larger speech recognition applications to preprocess and classify speaker accents, enhancing the accuracy of downstream ASR (Automatic Speech Recognition) systems by adapting to speaker variations.

📚 References
Jurafsky, D., & Martin, J. H. (2023). Speech and Language Processing. https://web.stanford.edu/~jurafsky/slp3/

Librosa Documentation – https://librosa.org/

Speech-to-Text Transcription App (Tkinter-Based)
This project is a simple and efficient speech-to-text transcription system developed using Python and Tkinter for the user interface. It leverages speech_recognition, pydub, and jiwer to convert audio to text and assess accuracy.

✅ Features
🎤 Upload .wav audio files for transcription

✍️ Real-time transcription using Google Speech Recognition API

📊 Transcription accuracy comparison using ground truth (optional)

🖥️ Lightweight desktop GUI using Tkinter

📂 File Structure
bash
Copy code
speech-to-text-app/
├── app.py                 # Main Tkinter GUI
├── transcribe.py          # Core speech-to-text logic
├── utils.py               # Helper functions
├── samples/               # Sample audio files
├── ground_truth/          # Optional ground truth transcriptions
├── README.md
├── requirements.txt
🧠 How It Works
Tkinter GUI lets the user select an audio file

speech_recognition recognizes spoken words via Google's API

jiwer compares transcribed output with a reference (if available)

pydub helps in preprocessing audio formats

🧪 Accuracy Evaluation
If a reference .txt file is provided, the app computes Word Error Rate (WER) using the jiwer package.

🛠️ Installation
bash
Copy code
git clone https://github.com/your-username/speech-to-text-app.git
cd speech-to-text-app
pip install -r requirements.txt
▶️ Usage
bash
Copy code
python app.py
Use the GUI to browse and select a .wav file

Click “Transcribe”

(Optional) Compare transcription against a ground truth .txt file

📦 Requirements
nginx
Copy code
speechrecognition
pydub
jiwer
tkinter
🏁 Sample Output
vbnet
Copy code
Transcription: "Hello everyone welcome to the project demo."
Accuracy (WER): 8.3%
📌 Notes
Make sure ffmpeg is installed and added to your system PATH for audio format handling.

Internet connection is required for Google’s speech recognition API.

📚 References
SpeechRecognition Docs

PyDub Docs

Jiwer Docs

Emotion Classification from Speech using MFCC Features
This project focuses on classifying human emotions from speech audio using Mel-Frequency Cepstral Coefficients (MFCC) as features and machine learning models for prediction.

✅ Features
🎙️ Extracts MFCC features from speech audio (.wav) files

🤖 Trains a classifier (e.g., SVM, Random Forest, or CNN) to detect emotions

📊 Evaluates model accuracy, confusion matrix, and F1-score

💾 Supports batch training with labeled datasets

🖥️ Optional GUI using Tkinter for ease of use

🧠 How It Works
Preprocessing: Load .wav files and extract MFCCs

Feature Extraction: Use librosa to compute MFCCs

Model Training: Train a machine learning classifier on labeled MFCC data

Prediction: Predict the emotion category of new audio samples

🧪 Supported Emotions
Depends on dataset used (e.g., RAVDESS, SAVEE, etc.) — typical labels include:

😡 Angry

😢 Sad

😀 Happy

😐 Neutral

😲 Fearful

😍 Surprised

🛠️ Installation
bash
Copy code
git clone https://github.com/your-username/speech-emotion-classifier.git
cd speech-emotion-classifier
pip install -r requirements.txt
▶️ Usage
Train the Model
bash
Copy code
python train_model.py
Predict with New Audio
bash
Copy code
python predict.py --file path/to/audio.wav
(Optional) Run the Tkinter App
bash
Copy code
python app.py
📂 File Structure
bash
Copy code
emotion-classifier/
├── app.py                 # Tkinter GUI
├── train_model.py         # Model training script
├── predict.py             # Emotion prediction script
├── utils/                 # Feature extraction and preprocessing functions
├── dataset/               # Audio files and labels
├── models/                # Saved models
├── requirements.txt
└── README.md
📦 Requirements
nginx
Copy code
librosa
scikit-learn
numpy
pandas
tkinter
matplotlib
joblib
📊 Example Output
yaml
Copy code
Predicted Emotion: Happy  
Model Accuracy: 85.7%
📚 References
Librosa Documentation

RAVDESS Dataset

scikit-learn Emotion Classification Guide
