# 🤟 ASL Sign Language Detection System

A deep learning-based computer vision project that detects and classifies American Sign Language (ASL) hand gestures into alphabets and control signs. This project demonstrates practical implementation of Convolutional Neural Networks (CNNs) for image classification tasks.

---

## 🚀 Project Overview

This project trains a CNN model to recognize ASL hand signs from images and predict the corresponding alphabet or command (e.g., SPACE, DELETE).

It covers the complete deep learning workflow:

* Data loading and preprocessing
* Model building using CNN
* Training and validation
* Model saving and inference

---

## 🧠 Tech Stack

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Matplotlib
* Scikit-learn

---

## 📂 Project Structure

```
asl-sign-language-detection/
│
├── dataset/                 # (Not included - large Kaggle dataset)
├── saved_model/
│   └── asl_model.h5
├── src/
│   ├── data_loader.py
│   ├── model.py
│   ├── train.py
│   └── predict.py
├── main.py
├── requirements.txt
└── README.md
```

---

## ⚙️ How It Works

### 1. Data Loading

* Uses `ImageDataGenerator`
* Automatically splits data into training (80%) and validation (20%)
* Applies normalization (rescaling pixel values)

### 2. Model Architecture

* Convolutional Neural Network (CNN)
* Multiple Conv2D + MaxPooling layers
* Dense layers with Dropout for regularization
* Softmax output layer for multi-class classification

### 3. Model Training

* Optimizer: Adam
* Loss: Categorical Crossentropy
* Trained for 10 epochs
* Evaluated using validation accuracy

### 4. Prediction System

* Loads trained model
* Preprocesses input image
* Outputs predicted ASL character

---

## ▶️ How to Run

### 1. Clone the Repository

```
git clone https://github.com/your-username/asl-sign-language-detection.git
cd asl-sign-language-detection
```

### 2. Install Dependencies

```
pip install -r requirements.txt
```

### 3. Add Dataset

Download the ASL Alphabet Dataset from Kaggle and place it in:

```
dataset/asl_alphabet_train/
```

### 4. Train the Model

```
python main.py
```

### 5. Run Prediction

```
from src.predict import predict_image
predict_image("path_to_image.jpg")
```

---

## 📊 Classes Supported

* Alphabets: A–Z
* Special Signs:

  * SPACE
  * DELETE
  * NOTHING

---

## 💡 Key Highlights

* End-to-end deep learning pipeline
* Clean modular code structure
* Image preprocessing and augmentation support
* Multi-class classification with CNN
* Ready-to-use prediction module

---

## 📌 Future Improvements

* Add real-time detection using webcam
* Improve model accuracy with transfer learning (ResNet, MobileNet)
* Deploy as a web application (Streamlit/Flask)
* Add UI for live predictions

---

## 🙌 Conclusion

This project demonstrates strong fundamentals in deep learning and computer vision, including CNN design, image preprocessing, and model deployment for real-world applications like sign language recognition.

---

## 📬 Contact

Feel free to connect for collaboration or feedback!

---

⭐ If you found this project useful, consider giving it a star!
