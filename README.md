🌾 AgroVet – Multilingual AI-Integrated Web Application for Animal Husbandry and Livestock Management

AgroVet is a smart AI-powered web application developed using Python Flask that assists farmers and livestock owners in managing animal health and improving productivity. The system integrates machine learning and deep learning models to predict livestock health conditions and provides access to veterinary services and government schemes through a simple multilingual interface.

🚀 Features

🧠 AI-based livestock disease/health prediction

🌐 Web application using Flask

🗣 Multilingual-friendly user interface

📍 Veterinary service locator (map integration)

📜 Government livestock schemes information

📊 Livestock data processing using CSV dataset

🛠 Technologies Used

Backend

Python

Flask

Machine Learning / Deep Learning (Keras)

Frontend

HTML

CSS

Data

CSV Dataset

📂 Project Structure
finalyear/
│
├── app.py                  # Main Flask application
├── s1.py                   # Livestock data handling
├── s2.py                   # Processing logic
├── s3.py                   # AI model prediction
│
├── s1/data/
│   └── livestock_data.csv  # Dataset
│
├── s3/
│   ├── keras_model.h5      # Trained AI model
│   └── labels.txt          # Prediction labels
│
├── templates/
│   ├── home.html
│   ├── login.html
│   ├── s1.html
│   ├── s2.html
│   ├── s3.html
│   ├── veterinary_map.html
│   └── govt-schemes.html
│
├── static/
│   ├── style.css
│   └── images/
│
└── Procfile.txt

⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/yourusername/AgroVet.git
cd AgroVet

2️⃣ Install Required Packages
pip install flask tensorflow pandas numpy

3️⃣ Run the Application
python app.py

4️⃣ Open in Browser
http://127.0.0.1:5000/

🔄 Workflow

User enters livestock details

Flask server processes input

AI model predicts health condition

Result displayed on web interface

🎯 Project Objective

To provide farmers with an easy-to-use AI-based system for livestock health monitoring, veterinary assistance, and agricultural information in multiple languages to improve animal care and reduce losses.

📈 Future Enhancements

Real-time language translation

Mobile app version

Cloud database integration

More disease prediction models

Farmer notification system

🤝 Contribution

Contributions are welcome!
Feel free to fork this project and submit pull requests.

📜 License

This project is for educational purposes.
