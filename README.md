# Face Detection 🚀

A simple and effective **Face Detection and Recognition** project implemented using **Python** and machine learning techniques. This repository detects and recognizes faces in real-time or from stored images using a **K-Nearest Neighbors (KNN)** classifier.

---

## 🔍 Project Overview

This project demonstrates how to detect and recognize faces using a dataset of face embeddings and a trained KNN model. It supports **real-time face recognition** through a webcam as well as inference on stored images.

✔️ Detects faces in images and webcam video  
✔️ Uses a trained KNN classifier  
✔️ Easy to run with Python and OpenCV  

---

## 📂 Repository Structure

Face-Detection/
├── Face_recuesion.ipynb # Notebook showing training and demo
├── realtime_knn.py # Script for real-time face detection
├── face_knn.pkl # Trained KNN model
├── face_knn2.pkl # Alternative trained model
└── README.md # Project documentation

yaml
Copy code

---

## 🛠️ Features

- 🖼️ Face detection and recognition using a KNN classifier  
- 🎥 Real-time face detection via webcam  
- 📊 Bounding boxes around detected faces  
- 🔄 Works with both stored images and live camera feed  

---

## 📥 Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/iitdamansharma/Face-Detection.git
cd Face-Detection
2️⃣ Create a Virtual Environment (Optional but Recommended)
bash
Copy code
python3 -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows
3️⃣ Install Dependencies
bash
Copy code
pip install opencv-python numpy scikit-learn
▶️ How to Run
🔴 Real-Time Face Detection
bash
Copy code
python realtime_knn.py
This will open your webcam and start detecting faces in real time.

📘 Jupyter Notebook Demo
Open and run:

bash
Copy code
Face_recuesion.ipynb
This notebook demonstrates:

Face embedding generation

Model training

Face recognition results

🧠 How It Works
Face images are converted into numerical feature vectors (embeddings).

A KNN classifier is trained on labeled face data.

During inference:

Faces are detected using OpenCV.

Features are extracted and passed to the trained KNN model.

The closest match determines the predicted identity.

📌 Example Output
Real-time bounding boxes around detected faces with predicted labels.

(Add a screenshot or GIF here to improve visual appeal.)

👨‍💻 Technologies Used
Python

OpenCV — image processing and webcam handling

scikit-learn (KNN) — classification

NumPy — numerical operations

📝 Future Improvements
🔹 Replace KNN with deep learning models (CNN, YOLO, FaceNet)

🔹 Add emotion or age prediction

🔹 Improve accuracy with a larger dataset

🔹 Build a GUI or web app for easier interaction

📜 License
This project is open-source and intended for educational and learning purposes.

🙏 Acknowledgements
Thanks to the open-source community and tutorials that helped in building this project.

👨‍💻 Author
Aman Sharma
IIT (ISM) Dhanbad
GitHub: https://github.com/iitdamansharma
