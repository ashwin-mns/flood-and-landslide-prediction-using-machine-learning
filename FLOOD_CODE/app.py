from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import sqlite3
import joblib
import numpy as np
from datetime import datetime
import os
import telepot
bot=telepot.Bot("8274686037:AAH67rN9oiXYV00eGCsPtNjQzbcjwvjteBo")
ch_id="893804937"


app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Database initialization
def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    
    # Create users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create predictions table
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            water_level REAL NOT NULL,
            rain REAL NOT NULL,
            soil_moisture REAL NOT NULL,
            prediction TEXT NOT NULL,
            probability REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()


MODEL_PATH = "outputs/best_model.joblib"
SCALER_PATH = "outputs/scaler.joblib"
ENCODER_PATH = "outputs/label_encoder.joblib"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(ENCODER_PATH)
    



# Database helper functions
def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    return conn

def get_user_by_username(username):
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
    conn.close()
    return user

def create_user(username, email, password):
    conn = get_db_connection()
    try:
        conn.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
                     (username, email, password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def save_prediction(user_id, water_level, rain, soil_moisture, prediction, probability):
    conn = get_db_connection()
    conn.execute('''
        INSERT INTO predictions (user_id, water_level, rain, soil_moisture, prediction, probability)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (user_id, water_level, rain, soil_moisture, prediction, probability))
    conn.commit()
    conn.close()

def get_user_predictions(user_id):
    conn = get_db_connection()
    predictions = conn.execute(
        'SELECT * FROM predictions WHERE user_id = ? ORDER BY created_at DESC',
        (user_id,)
    ).fetchall()
    conn.close()
    return predictions

def delete_prediction(prediction_id, user_id):
    conn = get_db_connection()
    conn.execute('DELETE FROM predictions WHERE id = ? AND user_id = ?', (prediction_id, user_id))
    conn.commit()
    conn.close()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        if get_user_by_username(username):
            flash('Username already exists!', 'error')
            return render_template('register.html')
        
        if create_user(username, email, password):
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
        else:
            flash('Registration failed!', 'error')
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = get_user_by_username(username)
        if user and user['password'] == password:  # In production, use proper password hashing
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid credentials!', 'error')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        flash('Please log in to access the dashboard.', 'warning')
        return redirect(url_for('login'))
    
    predictions = get_user_predictions(session['user_id'])
    return render_template('dashboard.html', predictions=predictions)

@app.route('/delete_prediction/<int:prediction_id>')
def delete_prediction_route(prediction_id):
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Not authorized'})
    
    delete_prediction(prediction_id, session['user_id'])
    return jsonify({'success': True})

@app.route('/input')
def input_page():
    if 'user_id' not in session:
        flash('Please log in to make predictions.', 'warning')
        return redirect(url_for('login'))
    import requests
    data=requests.get("https://api.thingspeak.com/channels/3160477/feeds.json?api_key=25MC3O5UM2XY2YXW&results=2")
    water=data.json()['feeds'][-1]['field1']
    rain=data.json()['feeds'][-1]['field2']
    soil=data.json()['feeds'][-1]['field3']

    return render_template('input.html', water=water, rain=rain, soil=soil)





from flask import render_template, request, session, flash, redirect, url_for

@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:
        flash('Please log in to make predictions.', 'warning')
        return redirect(url_for('login'))
    
    try:
        water_level = float(request.form['water_level'])
        rain = float(request.form['rain'])
        soil_moisture = float(request.form['soil_moisture'])
        
        print(f"Received prediction request - Water: {water_level}, Rain: {rain}, Soil: {soil_moisture}")
        
        # Validate inputs
        if water_level < 0 or water_level > 20:
            flash('Water level must be between 0 and 20.', 'danger')
            return redirect(url_for('input_page'))
        
        if rain < 0 or rain > 4095:
            flash('Rain sensor value must be between 0 and 4095.', 'danger')
            return redirect(url_for('input_page'))
        
        if soil_moisture < 0 or soil_moisture > 4095:
            flash('Soil moisture must be between 0 and 4095.', 'danger')
            return redirect(url_for('input_page'))
        
        # Prepare input for model
        X = np.array([[water_level, rain, soil_moisture]])
        X_scaled = scaler.transform(X)
        
        # Make prediction
        y_pred = model.predict(X_scaled)
        pred_label = label_encoder.inverse_transform(y_pred)[0]

        print(f"Prediction result: {pred_label}")

        if soil_moisture > 3000:
            landslide_alert="High risk of landslide due to saturated soil. Take necessary precautions."
            flash(landslide_alert, 'warning')
        else:
            landslide_alert="Soil moisture levels are normal. No immediate landslide risk detected."
            flash(landslide_alert, 'info')
        
        # Get probability if available
        probability = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_scaled)[0]
            probability = float(max(proba))
            print(f"Probability: {probability}")
        
        # Save prediction to database
        save_prediction(session['user_id'], water_level, rain, soil_moisture, pred_label, probability)
        bot.sendMessage(ch_id,f"New Prediction:\nUser: {session['username']}\nWater Level: {water_level}\nRain: {rain}\nSoil Moisture: {soil_moisture}\nPrediction: {pred_label}\nProbability: {probability}\nLandslide Alert: {landslide_alert}")
        # Render result template with prediction data
        return render_template('result.html',
                             prediction=pred_label,
                             probability=probability,
                             water_level=water_level,
                             rain=rain,
                             soil_moisture=soil_moisture,landslide_alert=landslide_alert)
    
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        flash(f'Prediction error: {str(e)}', 'danger')
        return redirect(url_for('input_page'))

@app.route('/result')
def result():
    # This route is only for direct access protection
    flash('Please make a prediction first to see results.', 'warning')
    return redirect(url_for('input_page'))



if __name__ == '__main__':
    init_db()
    app.run(debug=True)