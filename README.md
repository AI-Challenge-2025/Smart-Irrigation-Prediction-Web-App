# 💧 Smart Irrigation Prediction Web App

Predict whether you should water your crops using a trained AI model based on sensor data. Upload a CSV file, and get real-time predictions!

## 🚀 Features

* Upload `.csv` files with sensor data
* Predict whether irrigation is needed ("💧 รดน้ำ" or "☀️ ไม่รดน้ำ")
* View predictions in a clean table
* Download the results as `.csv`

## 🧠 Model

* Algorithm: `RandomForestClassifier`
* Trained with cleaned data (`TARP.csv`) from agricultural sensors
* Preprocessing includes removing columns with >70% NaN and filling remaining missing values with mean
* Saved model: `Smart_irrigation_model.pkl`

## 📂 Project Structure

```
.
├── app.py                      # Streamlit web app
├── Smart_irrigation_model.pkl # Trained ML model
├── requirements.txt           # Required packages
├── Clean-TARP-Final.csv       # Cleaned dataset used for training
├── X_train.csv / y_train.csv  # Split training data
├── X_test.csv / y_test.csv    # Split testing data
└── README.md                  # This file
```

## ▶️ Getting Started

1. **Install dependencies:**

```bash
pip install -r requirements.txt
```

2. **Run the app:**

```bash
streamlit run app.py
```

3. **Upload your sensor `.csv` file** and view predictions

## 🧪 Sample Input Format

Example columns (no `Status` column):

```
Soil Moisture,Temperature, Soil Humidity,Time
32.5,27.0,43.1,2025-05-28 06:00
...
```

## 🛠 requirements.txt

```
streamlit
pandas
scikit-learn
joblib
```

## 📤 Deploy (Optional)

Use [Streamlit Cloud](https://share.streamlit.io/) to deploy your app for free:

1. Push your repo to GitHub
2. Login to Streamlit Cloud → Deploy from GitHub
3. Done!

## 📚 License

MIT License © 2025

---

Built with ❤️ by \[เคน x แชท AI]