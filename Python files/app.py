from flask import Flask, request, jsonify, render_template
import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import joblib
import plotly.express as px
import warnings as wr
from datetime import datetime
import sqlite3
import os

wr.filterwarnings("ignore")

app = Flask(__name__)

# Configuration
app.config['DATABASE'] = 'predictions.db'
app.config['REQUIRED_FIELDS'] = ['total_feed_intake', 'feeding_intensity']  # Minimum required fields

# Load model
model = joblib.load('model.pkl')

# Define expected features and their validation rules
FEATURE_VALIDATION = {
    'Feeding Intensity': {'min': 5, 'max': 100, 'default': 50, 'unit': 'units'},
    'Total feed intake (g)': {'min': 100, 'max': 5000, 'default': None, 'unit': 'grams'},
    'No.  of feeding- bout/h': {'min': 0, 'max': 50, 'default': 10, 'unit': 'times/hour'},
    'No  of Head flicks/h': {'min': 0, 'max': 100, 'default': 20, 'unit': 'times/hour'},
    'No.  of drinking/h': {'min': 0, 'max': 50, 'default': 15, 'unit': 'times/hour'},
    'No. of preening/h': {'min': 0, 'max': 60, 'default': 20, 'unit': 'times/hour'},
    'No.  of feeder pecking/h': {'min': 0, 'max': 200, 'default': 50, 'unit': 'times/hour'},
    'No.  of Walking/h': {'min': 0, 'max': 100, 'default': 30, 'unit': 'times/hour'},
    'GE  (kcal/kg)': {'min': 0, 'max': 5000, 'default': 3000, 'unit': 'kcal/kg'},
    'N%': {'min': 0, 'max': 100, 'default': 5, 'unit': '%'}
}

# Feature mapping from form fields to model features
FEATURE_MAPPING = {
    'feeding_intensity': 'Feeding Intensity',
    'total_feed_intake': 'Total feed intake (g)',
    'feeding_bout': 'No.  of feeding- bout/h',
    'head_flicks': 'No  of Head flicks/h',
    'drinking': 'No.  of drinking/h',
    'preening': 'No. of preening/h',
    'feeder_pecking': 'No.  of feeder pecking/h',
    'walking': 'No.  of Walking/h',
    'ge_kcal': 'GE  (kcal/kg)',
    'n_percent': 'N%'
}

# Initialize database
def init_db():
    with sqlite3.connect(app.config['DATABASE']) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                feeding_intensity REAL,
                total_feed_intake REAL,
                feeding_bout REAL,
                head_flicks REAL,
                drinking REAL,
                preening REAL,
                feeder_pecking REAL,
                walking REAL,
                ge_kcal REAL,
                n_percent REAL,
                predicted_fcr REAL,
                alert TEXT,
                target_egg_weight REAL,
                total_feed_required REAL,
                daily_feed_per_bird REAL,
                feeding_schedule TEXT
            )
        ''')
        conn.commit()

# Database helper function
def get_db():
    db = sqlite3.connect(app.config['DATABASE'])
    db.row_factory = sqlite3.Row
    return db

# Function to calculate feed requirements
def calculate_feed_requirements(target_egg_weight_g, predicted_fcr):
    """Calculate total feed needed based on predicted FCR."""
    total_feed_required = target_egg_weight_g * predicted_fcr
    daily_feed = total_feed_required / 30  # Assume a 30-day cycle
    return total_feed_required, daily_feed

# Function to generate feeding schedule
def generate_feeding_schedule(daily_feed):
    """Create a feeding schedule based on daily feed requirements."""
    morning_feed = daily_feed * 0.6  # 60% in the morning
    afternoon_feed = daily_feed * 0.4  # 40% in the afternoon
    return {
        "Morning (6 AM - 8 AM)": f"{morning_feed:.2f} g",
        "Afternoon (3 PM - 5 PM)": f"{afternoon_feed:.2f} g"
    }

# Input validation function
def validate_input(feature_name, value):
    """Validate a single feature value."""
    rules = FEATURE_VALIDATION.get(feature_name, {})
    
    if value is None or value == '':
        if rules.get('default') is not None:
            return rules['default'], f"Used default value {rules['default']} for {feature_name}"
        return None, f"{feature_name} is required"
    
    try:
        num_value = float(value)
    except ValueError:
        return None, f"Invalid number format for {feature_name}"
    
    if 'min' in rules and num_value < rules['min']:
        return None, f"{feature_name} must be ≥ {rules['min']} {rules.get('unit', '')}"
    
    if 'max' in rules and num_value > rules['max']:
        return None, f"{feature_name} must be ≤ {rules['max']} {rules.get('unit', '')}"
    
    return num_value, None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict')
def predict_page():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Initialize database if not exists
        if not os.path.exists(app.config['DATABASE']):
            init_db()
        
        # Get and validate form data
        form_data = request.json
        if not form_data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required fields
        missing_fields = [f for f in app.config['REQUIRED_FIELDS'] if f not in form_data or not form_data[f]]
        if missing_fields:
            return jsonify({
                'error': 'Missing required fields',
                'missing': missing_fields
            }), 400
        
        # Process all features
        features = {}
        validation_notes = []
        
        for form_field, model_feature in FEATURE_MAPPING.items():
            value = form_data.get(form_field)
            validated_value, note = validate_input(model_feature, value)
            
            if validated_value is None and note:
                if FEATURE_VALIDATION[model_feature].get('default') is None:
                    return jsonify({'error': note}), 400
                validation_notes.append(note)
            
            features[model_feature] = validated_value if validated_value is not None else FEATURE_VALIDATION[model_feature]['default']
        
        # Prepare features for prediction
        features_df = pd.DataFrame([features.values()], columns=features.keys())
        
        # Make prediction
        predicted_fcr = model.predict(features_df)[0]
        
        # Generate alert message
        alert = ""
        if predicted_fcr > 2.0:
            alert = "Warning: High FCR detected! Consider reviewing feeding conditions and bird health."
        else:
            alert = "Great! Low FCR indicates excellent feed efficiency."
        
        # Store prediction in database
        with get_db() as db:
            db.execute('''
                INSERT INTO predictions (
                    feeding_intensity, total_feed_intake, feeding_bout, head_flicks,
                    drinking, preening, feeder_pecking, walking, ge_kcal, n_percent,
                    predicted_fcr, alert
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                features['Feeding Intensity'], features['Total feed intake (g)'],
                features['No.  of feeding- bout/h'], features['No  of Head flicks/h'],
                features['No.  of drinking/h'], features['No. of preening/h'],
                features['No.  of feeder pecking/h'], features['No.  of Walking/h'],
                features['GE  (kcal/kg)'], features['N%'],
                predicted_fcr, alert
            ))
            db.commit()
        
        return jsonify({
            'predicted_FCR': round(predicted_fcr, 2),
            'alert': alert,
            'validation_notes': validation_notes if validation_notes else None
        })
    
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred during prediction'}), 500

@app.route('/optimize', methods=['POST'])
def optimize():
    try:
        input_data = request.json
        if not input_data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate target egg weight
        target_egg_weight = input_data.get('target_egg_weight_g')
        if not target_egg_weight:
            return jsonify({'error': 'Target egg weight is required'}), 400
        
        try:
            target_egg_weight = float(target_egg_weight)
            if target_egg_weight <= 200 or target_egg_weight > 700:
                return jsonify({'error': 'Target egg weight must be between 200 and 700 grams'}), 400
        except ValueError:
            return jsonify({'error': 'Invalid target egg weight format'}), 400
        
        # Get or calculate predicted FCR
        predicted_fcr = input_data.get('predicted_FCR')
        if not predicted_fcr:
            # If no FCR provided, we need to calculate it from features
            features = {}
            for form_field, model_feature in FEATURE_MAPPING.items():
                value = input_data.get(form_field)
                validated_value, note = validate_input(model_feature, value)
                
                if validated_value is None and note:
                    if FEATURE_VALIDATION[model_feature].get('default') is None:
                        return jsonify({'error': note}), 400
                
                features[model_feature] = validated_value if validated_value is not None else FEATURE_VALIDATION[model_feature]['default']
            
            features_df = pd.DataFrame([features.values()], columns=features.keys())
            predicted_fcr = model.predict(features_df)[0]
        else:
            try:
                predicted_fcr = float(predicted_fcr)
            except ValueError:
                return jsonify({'error': 'Invalid FCR format'}), 400
        
        # Calculate feed requirements
        total_feed_required, daily_feed = calculate_feed_requirements(target_egg_weight, predicted_fcr)
        feeding_schedule = generate_feeding_schedule(daily_feed)
        
        # Update database with optimization results
        with get_db() as db:
            db.execute('''
                UPDATE predictions 
                SET target_egg_weight = ?,
                    total_feed_required = ?,
                    daily_feed_per_bird = ?,
                    feeding_schedule = ?
                WHERE id = (SELECT MAX(id) FROM predictions)
            ''', (
                target_egg_weight,
                total_feed_required,
                daily_feed,
                str(feeding_schedule)
            ))
            db.commit()
        
        return jsonify({
            'predicted_FCR': round(predicted_fcr, 2),
            'total_feed_required': f"{total_feed_required:.2f} g",
            'daily_feed_per_bird': f"{daily_feed:.2f} g/day",
            'feeding_schedule': feeding_schedule
        })
    
    except Exception as e:
        app.logger.error(f"Optimization error: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred during optimization'}), 500

@app.route('/history', methods=['GET'])
def get_history():
    try:
        with get_db() as db:
            cursor = db.execute('''
                SELECT timestamp, predicted_fcr, alert 
                FROM predictions 
                ORDER BY timestamp DESC 
                LIMIT 50
            ''')
            history = [dict(row) for row in cursor.fetchall()]
        
        return jsonify({'history': history})
    
    except Exception as e:
        app.logger.error(f"History error: {str(e)}")
        return jsonify({'error': 'Could not retrieve prediction history'}), 500

# Initialize database on startup
if not os.path.exists(app.config['DATABASE']):
    init_db()

if __name__ == '__main__':
    app.run(debug=True)