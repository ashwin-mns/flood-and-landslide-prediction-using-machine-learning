from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import sqlite3
import joblib
import numpy as np
from datetime import datetime
import os
import telepot
bot=telepot.Bot("8274686037:AAH67rN9oiXYV00eGCsPtNjQzbcjwvjteBo")
ch_id="893804937"




# Get the directory of the current file (absolute path)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'database.db')

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Global variables for lazy loading
bot = None
model = None
scaler = None
label_encoder = None

def get_bot():
    global bot
    if bot is None:
        try:
            bot = telepot.Bot("8274686037:AAH67rN9oiXYV00eGCsPtNjQzbcjwvjteBo")
        except Exception as e:
            print(f"Bot initialization failed: {e}")
    return bot

def load_models():
    global model, scaler, label_encoder
    if model is None:
        try:
            MODEL_PATH = os.path.join(BASE_DIR, "outputs", "best_model.joblib")
            SCALER_PATH = os.path.join(BASE_DIR, "outputs", "scaler.joblib")
            ENCODER_PATH = os.path.join(BASE_DIR, "outputs", "label_encoder.joblib")
            
            if os.path.exists(MODEL_PATH):
                model = joblib.load(MODEL_PATH)
                scaler = joblib.load(SCALER_PATH)
                label_encoder = joblib.load(ENCODER_PATH)
                print("Models loaded successfully.")
            else:
                print(f"Model files not found at {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading models: {e}")

# Load models on startup (non-blocking)
load_models()

# Database helper functions
def get_db_connection():
    if not os.path.exists(DB_PATH):
        # Fallback for Vercel if DB missing (to prevent crash, though logic will fail)
        print("Database file not found!")
        return None
        
    try:
        # Attempt to connect in read-only mode if possible, or standard
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.OperationalError:
         # Fallback to standard connect if uri mode fails or other issue
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            return conn
        except Exception as e:
            print(f"DB Connection failed: {e}")
            return None

def get_user_by_username(username):
    conn = get_db_connection()
    if not conn: return None
    try:
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        return user
    finally:
        conn.close()

def create_user(username, email, password):
    conn = get_db_connection()
    if not conn: return False
    try:
        # Check if we can write (might fail on Vercel)
        conn.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
                     (username, email, password))
        conn.commit()
        return True
    except sqlite3.OperationalError as e:
        print(f"DB Write failed (likely Read-Only filesystem): {e}")
        return False
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def save_prediction(user_id, water_level, rain, soil_moisture, prediction, probability):
    conn = get_db_connection()
    if not conn: return
    try:
        conn.execute('''
            INSERT INTO predictions (user_id, water_level, rain, soil_moisture, prediction, probability)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, water_level, rain, soil_moisture, prediction, probability))
        conn.commit()
    except Exception as e:
        print(f"Save prediction failed: {e}")
    finally:
        conn.close()

def get_user_predictions(user_id):
    conn = get_db_connection()
    if not conn: return []
    try:
        predictions = conn.execute(
            'SELECT * FROM predictions WHERE user_id = ? ORDER BY created_at DESC',
            (user_id,)
        ).fetchall()
        return predictions
    finally:
        conn.close()

def delete_prediction(prediction_id, user_id):
    conn = get_db_connection()
    if not conn: return
    try:
        conn.execute('DELETE FROM predictions WHERE id = ? AND user_id = ?', (prediction_id, user_id))
        conn.commit()
    except Exception as e:
        print(f"Delete failed: {e}")
    finally:
        conn.close()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/debug')
def debug():
    # Debug route to check status
    status = {
        "base_dir": BASE_DIR,
        "db_exists": os.path.exists(DB_PATH),
        "model_loaded": model is not None,
        "files_in_outputs": os.listdir(os.path.join(BASE_DIR, 'outputs')) if os.path.exists(os.path.join(BASE_DIR, 'outputs')) else "outputs dir missing"
    }
    return jsonify(status)

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
            flash('Registration failed! (Note: Database might be read-only on Vercel)', 'error')
    
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
    
    water, rain, soil = 0, 0, 0
    try:
        import requests
        # Add timeout to prevent hanging
        data=requests.get("https://api.thingspeak.com/channels/3160477/feeds.json?api_key=25MC3O5UM2XY2YXW&results=2", timeout=5)
        if data.status_code == 200:
            feeds = data.json().get('feeds', [])
            if feeds:
                water=feeds[-1].get('field1', 0)
                rain=feeds[-1].get('field2', 0)
                soil=feeds[-1].get('field3', 0)
    except Exception as e:
        print(f"ThingSpeak error: {e}")

    return render_template('input.html', water=water, rain=rain, soil=soil)


from flask import render_template, request, session, flash, redirect, url_for

@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:
        flash('Please log in to make predictions.', 'warning')
        return redirect(url_for('login'))
    
    # Reload model if missing (lazy load attempt)
    if model is None:
        load_models()
        if model is None:
             flash('Prediction model could not be loaded on server.', 'danger')
             return redirect(url_for('input_page'))

    try:
        water_level = float(request.form['water_level'])
        rain = float(request.form['rain'])
        soil_moisture = float(request.form['soil_moisture'])
        
        # Validate inputs
        if water_level < 0 or water_level > 20:
            flash('Water level must be between 0 and 20.', 'danger')
            return redirect(url_for('input_page'))
        
        # Prepare input for model
        X = np.array([[water_level, rain, soil_moisture]])
        X_scaled = scaler.transform(X)
        
        # Make prediction
        y_pred = model.predict(X_scaled)
        pred_label = label_encoder.inverse_transform(y_pred)[0]

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
        
        # Save prediction to database
        save_prediction(session['user_id'], water_level, rain, soil_moisture, pred_label, probability)
        
        # Send telegram (lazy load bot)
        try:
           tb = get_bot()
           if tb:
               tb.sendMessage(ch_id,f"New Prediction:\nUser: {session['username']}\nWater Level: {water_level}\nRain: {rain}\nSoil Moisture: {soil_moisture}\nPrediction: {pred_label}\nProbability: {probability}\nLandslide Alert: {landslide_alert}")
        except Exception as e:
            print(f"Telegram failed: {e}")

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
    # init_db() # Disabled for Vercel/Production to prevent overwrite/lock issues if not needed
    app.run(debug=True)
