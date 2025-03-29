from flask import Flask, request, jsonify, render_template
import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import joblib
import plotly.express as px

app = Flask(__name__)

# Load model
model = joblib.load('model.pkl')

# Load dataset
df = pd.read_csv('C:/PROJECT4.2/Updated_dataset.csv')

# Define expected features
EXPECTED_FEATURES = ['Feeding Intensity', 'Total feed intake (g)', 'No.  of feeding- bout/h', 
                     'No  of Head flicks/h', 'No.  of drinking/h', 'No. of preening/h',
                     'No.  of feeder pecking/h', 'No. of cage pecking', 'No.  of Walking/h',
                     'GE  (kcal/kg)', 'N%']

# Function to calculate feed requirements based on target egg weight and predicted FCR
def calculate_feed_requirements(target_egg_weight_g, predicted_fcr):
    """Calculate total feed needed based on predicted FCR."""
    total_feed_required = target_egg_weight_g * predicted_fcr
    daily_feed = total_feed_required / 30  # Assume a 30-day cycle
    return total_feed_required, daily_feed

# Function to generate an optimal feeding schedule
def generate_feeding_schedule(daily_feed):
    """Create a feeding schedule based on daily feed requirements."""
    morning_feed = daily_feed * 0.6  # 60% in the morning
    afternoon_feed = daily_feed * 0.4  # 40% in the afternoon
    return {
        "Morning (6 AM - 8 AM)": f"{morning_feed:.2f} g",
        "Afternoon (3 PM - 5 PM)": f"{afternoon_feed:.2f} g"
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_bird_ids', methods=['GET'])
def get_bird_ids():
   try:  
       bird_ids = df['Bird ID.'].unique().tolist()
       return jsonify({'bird_ids': bird_ids})
   except Exception as e:
       return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.json
        if not input_data or 'bird_id' not in input_data:
            return jsonify({'error': 'Bird ID is missing'}), 400
        
        bird_id = input_data['bird_id']
        bird_data = df[df['Bird ID.'] == bird_id].iloc[0]

        # Ensure the input data has the correct feature names
        features = bird_data[EXPECTED_FEATURES].values.reshape(1, -1)
        features_df = pd.DataFrame(features, columns=EXPECTED_FEATURES)  # Convert to DataFrame with feature names

        predicted_fcr = model.predict(features_df)[0]
        
        alert = ""
        if predicted_fcr > 2.0:
            alert = "Warning: High FCR detected! Consider reviewing feeding conditions and bird health."
        else:
            alert = "Great! Low FCR indicates excellent feed efficiency."

        return jsonify({
            'bird_id': bird_id,
            'predicted_FCR': round(predicted_fcr, 2),
            'alert': alert,
            'redirect_url': f'/dashboard?bird_id={bird_id}&predicted_FCR={round(predicted_fcr, 2)}&alert={alert}'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/optimize', methods=['POST'])
def optimize():
    try:
        input_data = request.json
        bird_id = input_data.get('bird_id')
        target_egg_weight = float(input_data.get('target_egg_weight_g'))
        
        bird_data = df[df['Bird ID.'] == bird_id].iloc[0]
        predicted_fcr = model.predict(bird_data[EXPECTED_FEATURES].values.reshape(1, -1))[0]

        # Calculate feed requirements and schedule
        total_feed_required, daily_feed = calculate_feed_requirements(target_egg_weight, predicted_fcr)
        feeding_schedule = generate_feeding_schedule(daily_feed)

        # Convert feeding schedule dictionary to a formatted string
        feeding_schedule_str = "\n".join([f"{time}: {amount}" for time, amount in feeding_schedule.items()])

        return jsonify({
            'bird_id': bird_id,
            'predicted_FCR': round(predicted_fcr, 2),
            'total_feed_required': f"{total_feed_required:.2f} g",
            'daily_feed_per_bird': f"{daily_feed:.2f} g/day",
            'feeding_schedule': feeding_schedule_str  # Return as a formatted string
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/historical_fcr/<bird_id>', methods=['GET'])
def get_historical_fcr(bird_id):
    try:
        bird_history = df[df['Bird ID.'] == bird_id][['date', 'FCR']]
        if bird_history.empty:
            return jsonify({'error': 'No historical data available for this bird'}), 404

        return bird_history.to_json(orient='records')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Dash app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

dash_app = dash.Dash(__name__, server=app, url_base_pathname='/dashboard/', external_stylesheets=external_stylesheets)

dash_app.layout = html.Div([
    html.H1('Feed Efficiency Dashboard'),
    html.Div(id='bird-info', style={'margin-bottom': '20px'}),
    html.Div(id='alert-message', style={'margin-bottom': '20px'}),  # Div for alert message
    html.Label('Enter Target Egg Weight (g):'),
    dcc.Input(id='target-egg-weight', type='number', placeholder='e.g., 450', style={'margin-right': '10px'}),
    html.Button('Optimize Feed', id='optimize-button', n_clicks=0),
    html.Div(id='optimization-output', style={'margin-top': '20px'}),  # Div for optimization results
    dcc.Graph(id='historical-fcr-graph', style={'margin-top': '30px'})  # Graph for historical FCR
])

@dash_app.callback(
    [Output('bird-info', 'children'), Output('alert-message', 'children'), Output('optimization-output', 'children'), Output('historical-fcr-graph', 'figure')],
    [Input('url', 'search'), Input('optimize-button', 'n_clicks')],
    [State('target-egg-weight', 'value')]
)
def display_bird_info(search, n_clicks, target_egg_weight):
    import urllib.parse as urlparse
    params = urlparse.parse_qs(search[1:])
    bird_id = params.get('bird_id', [''])[0]
    predicted_fcr = params.get('predicted_FCR', [''])[0]
    alert = params.get('alert', [''])[0]
    
    bird_info_display = "No data available"
    alert_message = ""
    optimization_output = ""

    if bird_id and predicted_fcr:
        bird_info_display = [
            html.H4(f"Bird ID: {bird_id} | Predicted FCR: {predicted_fcr}")
        ]
        alert_message = html.P(alert, style={'color': 'red' if 'Warning' in alert else 'green'})

        # Handle optimization when the button is clicked
        if n_clicks > 0 and target_egg_weight:
            try:
                target_egg_weight = float(target_egg_weight)
                total_feed_required, daily_feed = calculate_feed_requirements(target_egg_weight, float(predicted_fcr))
                feeding_schedule = generate_feeding_schedule(daily_feed)

                optimization_output = [
                    html.H5("Optimal Feed Plan"),
                    html.P(f"Total Feed Required: {total_feed_required:.2f} g"),
                    html.P(f"Daily Feed Per Bird: {daily_feed:.2f} g/day"),
                    html.P("Feeding Schedule:"),
                    html.Ul([html.Li(f"{time}: {amount}") for time, amount in feeding_schedule.items()])
                ]
            except Exception as e:
                optimization_output = html.P(f"Error during optimization: {str(e)}", style={'color': 'red'})

    # Fetch historical FCR data
    try:
        bird_history = df[df['Bird ID.'] == bird_id][['date', 'FCR']]
        fig = px.line(bird_history, x='date', y='FCR', title=f'FCR History for Bird {bird_id}')
    except Exception as e:
        fig = px.line(title='No historical data available')

    return bird_info_display, alert_message, optimization_output, fig

if __name__ == '__main__':
    app.run(debug=True)