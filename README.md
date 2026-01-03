# Face Detection 🚀

A simple and effective **Face Detection** project implemented using Python and machine learning techniques. This repository detects and recognizes faces in real-time or from stored images using K-Nearest Neighbors (KNN) classification.

---

## 🔍 Project Overview

This project demonstrates how to detect and recognize faces using a dataset of face embeddings and a KNN model. It includes tools to train a model and perform **real-time face recognition** from a webcam feed.

✔️ Detects faces in images and webcam video  
✔️ Uses a trained KNN classifier  
✔️ Easy to run with Python and OpenCV

---

## 📂 Repository Structure
Face-Detection/
├── Face_recuesion.ipynb # Notebook showing training + demo
├── realtime_knn.py # Script for real-time face detection
├── face_knn.pkl # Trained KNN model
├── face_knn2.pkl # Alternative trained model
└── README.md # This file


---

## 🛠️ Features

- 🖼️ Face detection and recognition using KNN classifier  
- 🎥 Real-time webcam detection  
- 📊 Visual bounding boxes around detected faces  
- 🔄 Works with stored images or live camera feed  

---

## 📥 Installation & Setup

1. **Clone the repo**
bash
   git clone https://github.com/iitdamansharma/Face-Detection.git
   
2.Create a virtual environment
bash
python3 -m venv venv
source venv/bin/activate     # macOS / Linux
venv\Scripts\activate        # Windows

3.Install dependencies
bash
pip install -r requirements.txt

▶️ How to Run
Real-Time Detection
python realtime_knn.py
This will open your webcam and start detecting faces live.
This will open your webcam and start detecting faces live.

Jupyter Notebook Demo

Open and run:

Face_recuesion.ipynb


This notebook shows face embedding training and detection outputs.
Jupyter Notebook Demo
📌 Example Output

Replace above with an actual screenshot from the notebook or real-time detection.

🧠 How It Works

This project:

Uses images to generate face embeddings.

Trains a KNN classifier on them.

Detects faces in new frames using the trained model.

The KNN model predicts the closest match based on stored face embeddings.

👨‍💻 Technologies Used

Python

OpenCV for image & webcam processing

scikit-learn (KNN) for classification

NumPy — data handling

📝 Future Improvements

Here are a few ideas you can note for future work:

🔹 Replace KNN with deep learning (e.g., CNN, YOLO)

🔹 Add emotion/age prediction

🔹 Build a GUI for easier use 🎨

📜 License

This project is open source and available for personal and educational use.

🙏 Acknowledgements

Thanks to the open-source community and tutorials that helped build this project.
📜 License

This project is open-source and intended for educational and learning purposes.

👨‍💻 Author

Aman Sharma
IIT (ISM) Dhanbad
GitHub: https://github.com/iitdamansharma
