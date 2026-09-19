# Sign Language Detector

An AI-based computer vision application that detects and recognizes predefined sign language hand gestures in real time using a webcam.

The project uses **Python, OpenCV, CVZone, MediaPipe, and a trained Keras/TensorFlow model** to detect a user's hand and classify the performed gesture.

---

## 📌 Project Overview

Communication can be difficult for people who use sign language when interacting with people who do not understand it.

The **Sign Language Detector** is designed to recognize predefined hand gestures through a webcam and display the corresponding meaning in real time.

The system captures video from the webcam, detects the hand, processes the captured image, and uses a trained machine learning model to predict the gesture.

### Currently Supported Gestures

- 👋 **Hello**
- ❤️ **I Love You**
- 🙏 **Thank You**
- ✅ **Yes**

---

## ✨ Features

- Real-time hand gesture detection
- Webcam-based input
- Hand tracking using MediaPipe through CVZone
- Gesture classification using a trained Keras/TensorFlow model
- Live prediction displayed on the screen
- Predefined sign language gesture recognition
- Python-based desktop application

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| OpenCV | Webcam access and image processing |
| CVZone | Computer vision utilities and hand tracking |
| MediaPipe | Hand detection and landmark tracking |
| TensorFlow / Keras | Machine learning model and gesture classification |
| NumPy | Numerical and image data processing |

---

## 📂 Project Structure

```text
Sign-Language-Detector/
│
├── Data/
│   └── Dataset and captured gesture data
│
├── Model/
│   ├── keras_model.h5
│   └── labels.txt
│
├── datacollection.py
├── test.py
├── requirements.txt
├── .gitignore
└── README.md
```

> **Note:** File names inside the `Model/` folder may vary depending on the trained model used.

---

## ⚙️ How It Works

The application follows this pipeline:

```text
Webcam
   ↓
Video Frame Capture
   ↓
Hand Detection
   ↓
Hand Image Processing
   ↓
Trained Keras Model
   ↓
Gesture Classification
   ↓
Predicted Sign Displayed
```

### Processing Pipeline

1. The webcam captures a live video stream.
2. CVZone/MediaPipe detects the user's hand.
3. The detected hand region is extracted and processed.
4. The processed image is passed to the trained Keras model.
5. The model predicts the corresponding gesture.
6. The predicted gesture is displayed in the application window.

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Yxminn/Sign-Language-Detector.git
cd Sign-Language-Detector
```

### 2. Create a Virtual Environment

It is recommended to use a virtual environment.

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

Make sure your webcam is connected and available.

Run the main detection script:

```bash
python test.py
```

The application will open the webcam and begin detecting the supported hand gestures.

---

## 📸 Data Collection

The project includes `datacollection.py`, which can be used to capture gesture images for creating or expanding the dataset.

Run:

```bash
python datacollection.py
```

The collected images can then be used as training data for expanding the gesture recognition model.

---

## 🧠 Machine Learning Model

The project uses a trained **Keras/TensorFlow classification model** to identify predefined hand gestures.

The model uses:

- `keras_model.h5` — trained classification model
- `labels.txt` — gesture labels used by the model

The processed hand image is passed to the trained model, which predicts the corresponding gesture class.

---

## 🎯 Project Goals

The main objectives of this project are:

- To detect hand gestures using computer vision.
- To recognize predefined sign language gestures in real time.
- To demonstrate the use of machine learning for gesture classification.
- To build a simple and accessible sign language recognition application.

---

## 🔮 Future Improvements

Possible future enhancements include:

- Increasing the number of supported signs
- Supporting a larger sign language vocabulary
- Improving model accuracy
- Adding sentence formation
- Adding text-to-speech output
- Supporting multiple hand gestures
- Developing a more advanced graphical user interface
- Creating a web or mobile version
- Improving recognition under different lighting conditions

---

## 📊 Current Supported Signs

| Gesture | Meaning |
|---|---|
| 👋 | Hello |
| ❤️ | I Love You |
| 🙏 | Thank You |
| ✅ | Yes |

---

## 💻 Requirements

- Python 3.x
- Working webcam
- Windows, Linux, or macOS
- Internet connection for initial dependency installation

---

## ⚠️ Notes

- The application requires access to a working webcam.
- Recognition performance can depend on lighting, camera quality, hand position, and background conditions.
- The current model is designed for the predefined gestures included in the project.

---

## 👨‍💻 Author

**Yxminn**

GitHub: [https://github.com/Yxminn](https://github.com/Yxminn)

---

## 📜 License

This project was created for educational and academic purposes.
