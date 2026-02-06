# AgroVet – Multilingual AI-Integrated Web Application for Animal Husbandry and Livestock Management

AgroVet is an AI-powered web application built using Python Flask to assist farmers and livestock owners in monitoring animal health and improving productivity. The system integrates machine learning and deep learning models to predict livestock diseases and provides veterinary services and government scheme information through a multilingual interface.

---

## Features

- AI-based livestock disease prediction  
- Web application using Flask  
- Multilingual user interface  
- Veterinary service locator  
- Government schemes information  
- CSV-based livestock dataset  

---

## Technologies Used

### Backend
- Python  
- Flask  
- TensorFlow / Keras  

### Frontend
- HTML  
- CSS  

### Data
- CSV files  


## Project Structure

finalyear/
├── app.py
├── s1.py
├── s2.py
├── s3.py
├── s1/data/livestock_data.csv
├── s3/keras_model.h5
├── s3/labels.txt
├── templates/
│ ├── home.html
│ ├── login.html
│ ├── s1.html
│ ├── s2.html
│ ├── s3.html
│ ├── veterinary_map.html
│ └── govt-schemes.html
├── static/
│ ├── style.css
│ └── images/
└── Procfile.txt




## Installation and Setup

### Clone the repository

```bash
git clone https://github.com/yourusername/AgroVet.git
cd AgroVet
Install required packages
bash
Copy code
pip install flask tensorflow pandas numpy
Run the application
bash
Copy code
python app.py
Open in browser
cpp
Copy code
http://127.0.0.1:5000/
Workflow
User enters livestock details

Flask server processes the request

AI model predicts the health condition

Result is displayed on the web interface

Project Objective
To provide an intelligent and easy-to-use platform for farmers to manage livestock health, access veterinary services, and receive government-related information using artificial intelligence and multilingual support.

Future Enhancements
Mobile application development

Cloud database integration

Real-time language translation

Advanced disease prediction models

Notification and alert system

License
This project is developed for academic and educational purposes.

vbnet
Copy code

If you’d like, I can also:

Write a short GitHub description  
Add contribution guidelines  
Add requirements.txt  
Create a professional project banner text  


