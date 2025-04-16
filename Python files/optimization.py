import pandas as pd
import numpy as np
import joblib

# Load dataset
df = pd.read_csv("C:/PROJECT4.2/Updated_dataset.csv")  # Replace with your actual file path

# Load trained model
model = joblib.load("model.pkl")

# Define feature columns (ensure they match training features)
feature_columns = ['Feeding Intensity', 'Total feed intake (g)', 'No.  of feeding- bout/h', 'No  of Head flicks/h',
                   'No.  of drinking/h', 'No. of preening/h', 'No.  of feeder pecking/h',
                   'No.  of Walking/h', 'GE  (kcal/kg)', 'N%']
X = df[feature_columns]

# Function to calculate feed requirements based on target egg weight and predicted FCR
def calculate_feed_requirements(target_egg_weight_kg):
    """Calculate total feed needed based on predicted FCR."""
    # Predict FCR
    X_temp = X.copy()
    X_temp = X_temp[model.feature_names_in_]
    predicted_fcr = np.mean(model.predict(X_temp))

    # Calculate total feed required
    total_feed_required = target_egg_weight_kg * predicted_fcr
    daily_feed = total_feed_required / 30  # Assume a 30-day cycle
    
    return predicted_fcr, total_feed_required, daily_feed

# Function to generate an optimal feeding schedule
def generate_feeding_schedule(daily_feed):
    """Create a feeding schedule based on daily feed requirements."""
    morning_feed = daily_feed * 0.6  # 60% in the morning
    afternoon_feed = daily_feed * 0.4  # 40% in the afternoon

    return {
        "Morning (6 AM - 8 AM)": f"{morning_feed:.2f} g",
        "Afternoon (3 PM - 5 PM)": f"{afternoon_feed:.2f} g"
    }

# Function to get complete optimization results
def get_optimization_results(target_egg_weight_kg):
    """Return predicted FCR, feed amount, and recommended schedule."""
    predicted_fcr, total_feed_required, daily_feed = calculate_feed_requirements(target_egg_weight_kg)
    feeding_schedule = generate_feeding_schedule(daily_feed)
    
    return {
        "Predicted FCR": f"{predicted_fcr:.2f}",
        "Total Feed Required": f"{total_feed_required:.2f} g",
        "Daily Feed Per Bird": f"{daily_feed:.2f} g/day",
        "Feeding Schedule": feeding_schedule
    }

# Example usage
results = get_optimization_results(target_egg_weight_kg=450)
print(results)
