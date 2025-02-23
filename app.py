from flask import Flask, request, jsonify
import pickle
import json
import numpy as np
from flask_cors import CORS
import logging

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load the pre-trained model
with open('random_forest_V4.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

def calculate_features(data):
    """
    Calculate features from the game session data.
    """
    # Initialize counters using a dictionary
    counters = {
        "total_steps": 0,
        "move_hand_count": 0,
        "collect_coin_count": 0,
        "next_level_count": 0,
        "configuration_count": 0,
        "use_hint_count": 0,
        "change_timer_count": 0,
        "change_cell_count": 0,
        "rate_level_count": 0,
        "collect_use_powerup_count": 0,
        "open_inventory_count": 0,
        "change_lives_count": 0,
        "visit_leaderboard_count": 0,
        "perform_undo_count": 0,
        "accept_gift_count": 0,
        "change_interface_count": 0,
        "reject_gift_count": 0,
        "open_rules_count": 0,
        "perform_reset_count": 0,
    }

    # Parse JSON data
    events = json.loads(data)

    # Count events
    for event in events:
        counters["total_steps"] += 1

        event_type = event.get("type")
        action = event.get("action")

        if event_type == "gameplay":
            if action == "move hand":
                counters["move_hand_count"] += 1
            elif action == "collect coin":
                counters["collect_coin_count"] += 1
            elif action == "next level":
                counters["next_level_count"] += 1
            elif action == "open rules":
                counters["open_rules_count"] += 1
            elif action == "perform reset":
                counters["perform_reset_count"] += 1
            elif action == "perform undo":
                counters["perform_undo_count"] += 1
            elif action == "accept gift":
                counters["accept_gift_count"] += 1
            elif action == "reject gift":
                counters["reject_gift_count"] += 1

        elif event_type == "configuration":
            counters["configuration_count"] += 1
            if action == "use hint":
                counters["use_hint_count"] += 1
            elif "timer" in event:
                counters["change_timer_count"] += 1
            elif action == "change cell count":
                counters["change_cell_count"] += 1
            elif action == "change color scheme":
                counters["change_interface_count"] += 1
            elif action == "change lives count":
                counters["change_lives_count"] += 1

        elif action == "rate level":
            counters["rate_level_count"] += 1
        elif action in ["sell powerup", "use powerup", "collect powerup"]:
            counters["collect_use_powerup_count"] += 1
        elif action == "open inventory":
            counters["open_inventory_count"] += 1
        elif action in ["leaderboard", "visit leaderboard"]:
            counters["visit_leaderboard_count"] += 1

    # Calculate average steps per level
    average_steps = counters["total_steps"] / (counters["next_level_count"] + 1) if counters["next_level_count"] > 0 else counters["total_steps"]

    # Create a dictionary of features
    features = {
        "Average step": average_steps,
        "Total Steps": counters["total_steps"],
        "Move hand": counters["move_hand_count"],
        "Collect coin": counters["collect_coin_count"],
        "Next level": counters["next_level_count"],
        "Configuration": counters["configuration_count"],
        "Hint": counters["use_hint_count"],
        "Change timer": counters["change_timer_count"],
        "Change cell count": counters["change_cell_count"],
        "Rate level": counters["rate_level_count"],
        "change levels COUNT": 0,  # Placeholder
        "Collect & use powerup": counters["collect_use_powerup_count"],
        "Open rules": counters["open_rules_count"],
        "Open inventory": counters["open_inventory_count"],
        "Change interface look like (color, cell, clock shape)": counters["change_interface_count"],
        "Change lives count": counters["change_lives_count"],
        "Visit leaderboard": counters["visit_leaderboard_count"],
        "Perform reset": counters["perform_reset_count"],
        "Perform undo": counters["perform_undo_count"],
        "Accept gift": counters["accept_gift_count"],
        "Reject gift": counters["reject_gift_count"],
    }

    return features

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict personality based on game session data.
    """
    app.logger.info("Received prediction request")

    # Validate request
    if not request.json or 'session' not in request.json:
        app.logger.error("Missing session data in request")
        return jsonify({'error': 'Missing session data'}), 400

    data = request.json['session']
    
    try:
        # Calculate features
        features = calculate_features(data)

        # Convert features to a NumPy array for prediction
        X_new = np.array([float(value) for value in features.values()]).reshape(1, -1)

        # Make prediction
        prediction = loaded_model.predict(X_new)
        prediction_value = int(prediction[0]) if isinstance(prediction[0], np.int64) else prediction[0]

        app.logger.info(f"Prediction result: {prediction_value}")
        return jsonify({'prediction': prediction_value})
    
    except Exception as e:
        app.logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint.
    """
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    # Run the app in development mode
    app.run(debug=True)