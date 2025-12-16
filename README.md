# Flood and Landslide Prediction using Machine Learning

## Overview

This project is a web-based application designed to predict the risk of floods and landslides using Machine Learning. It processes environmental data such as **Water Level**, **Rainfall**, and **Soil Moisture** to classify the risk level. The system integrates with **ThingSpeak** for fetching sensor data and sends real-time alerts via **Telegram**.

## Features

- **Machine Learning**: detailed analysis and prediction using trained models (Random Forest, Gradient Boosting, etc.).
- **Real-time Data Integration**: Fetches live sensor data from ThingSpeak channels.
- **Web Interface**: User-friendly dashboard built with Flask.
- **User Authentication**: Secure Login and Registration system.
- **Alert System**:
  - Visual alerts on the dashboard.
  - **Telegram Bot** integration for pushing instant notifications to users.
- **Dashboard**: View history of predictions and risk assessments.
- **Exploratory Data Analysis (EDA)**: Automated generation of histograms, correlations, and confusion matrices during training.

## Tech Stack

- **Backend**: Python, Flask
- **Database**: SQLite
- **Machine Learning**: Scikit-Learn, Pandas, NumPy, Joblib
- **Visualization**: Matplotlib, Seaborn
- **APIs**: ThingSpeak (IoT Data), Telepot (Telegram Bot)

## Project Structure

```
├── app.py                 # Main Flask application
├── Train.py               # ML Model training and evaluation script
├── Test.py                # Testing script
├── database.db            # SQLite database (Users & Predictions)
├── data.csv               # Dataset for training
├── flood_dataset.csv      # Alternative dataset
├── outputs/               # Directory for saved models and plots
│   ├── best_model.joblib  # The best performing trained model
│   ├── scaler.joblib      # Data scaler
│   └── ...
├── templates/             # HTML templates for the web app
├── static/                # Static assets (CSS/JS)
└── README.md              # Project documentation
```

## Installation

1.  **Clone the Repository** (if applicable) or navigate to the project directory.

2.  **Install Dependencies**:
    Ensure you have Python installed. Install the required libraries using pip:
    ```bash
    pip install flask pandas numpy scikit-learn matplotlib seaborn joblib telepot requests
    ```

## Usage

### 1. Train the Model (Optional)
If you want to retrain the machine learning models with new data (`data.csv`):
```bash
python Train.py --data data.csv
```
This will:
- Perform data analysis and save plots to `outputs/`.
- Train multiple models and save the best one to `outputs/best_model.joblib`.

### 2. Run the Web Application
Start the Flask server:
```bash
python app.py
```

### 3. Access the Application
Open your web browser and go to:
`http://127.0.0.1:5000`

- **Register/Login**: Create an account to access features.
- **Input / Predict**:
  - The system can auto-fetch values from ThingSpeak.
  - Or manually enter Water Level, Rain, and Soil Moisture values.
  - Click **Predict** to see the risk status.
- **Dashboard**: View your past predictions.

## Configuration

- **Telegram Bot**: The bot token is configured in `app.py`.
- **ThingSpeak**: API keys for fetching data are configured in `app.py`.

## License

[Your License Here]
