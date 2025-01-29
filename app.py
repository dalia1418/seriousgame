from flask import Flask, request, jsonify
import pickle
import json
import pandas as pd
import numpy as np
from flask_cors import CORS
import logging

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})

logging.basicConfig(level=logging.DEBUG)

# Load the model
with open('random_forest_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

def calculate_features(data):
    # Initialize counters
    total_steps = 0
    move_hand_count = 0
    collect_coin_count = 0
    next_level_count = 0
    configuration_count = 0
    use_hint_count = 0
    change_timer_count = 0
    change_cell_count = 0
    rate_level_count = 0
    collect_use_powerup_count = 0
    open_inventory_count = 0
    change_lives_count = 0
    visit_leaderboard_count = 0
    perform_undo_count = 0
    accept_gift_count = 0
    change_interface_count = 0
    reject_gift_count = 0
    open_rules_count = 0  
    perform_reset_count = 0  

    # Parse JSON data
    events = json.loads(data)

    # Count events
    for event in events:
        total_steps += 1

        if event['type'] == 'gameplay' and event['action'] == 'move hand':
            move_hand_count += 1
        elif event['type'] == 'gameplay' and event['action'] == 'collect coin':
            collect_coin_count += 1
        elif event['type'] == 'gameplay' and event['action'] == 'next level':
            next_level_count += 1
        elif event['type'] == 'configuration':
            configuration_count += 1
        elif event['action'] == 'use hint':
            use_hint_count += 1
        elif event['type'] == 'configuration' and 'timer' in event:
            change_timer_count += 1
        elif event['action'] == 'change cell count':
            change_cell_count += 1
        elif event['action'] == 'rate level':
            rate_level_count += 1
        elif event['action'] in ['sell powerup', 'use powerup', 'collect powerup']:
            collect_use_powerup_count += 1
        elif event['action'] == 'open inventory':
            open_inventory_count += 1
        elif event['type'] == 'gameplay' and event['action'] == 'open rules':
            open_rules_count += 1
        elif event['type'] == 'configuration' and event['action'] == 'change color scheme':
            change_interface_count += 1
        elif event['type'] == 'configuration' and event['action'] == 'change lives count':
            change_lives_count += 1
        elif event['action'] in ['leaderboard', 'visit leaderboard']:
            visit_leaderboard_count += 1
        elif event['type'] == 'gameplay' and event['action'] == 'perform reset':
            perform_reset_count += 1
        elif event['type'] == 'gameplay' and event['action'] == 'perform undo':
            perform_undo_count += 1
        elif event['type'] == 'gameplay' and event['action'] == 'accept gift':
            accept_gift_count += 1
        elif event['type'] == 'gameplay' and event['action'] == 'reject gift':
            reject_gift_count += 1

    # Calculate average steps per level (avoid division by zero)
    average_steps = total_steps / (next_level_count + 1) if next_level_count > 0 else total_steps

    # Create a dictionary of features with default values for missing features.
    features = {
        "Average step": average_steps,
        "Total Steps": total_steps,
        "Move hand": move_hand_count,
        "Collect coin": collect_coin_count,
        "Next level": next_level_count,
        "Configuration": configuration_count,
        "Hint": use_hint_count,
        "Change timer": change_timer_count,
        "Change cell count": change_cell_count,
        "Rate level": rate_level_count,
        "change levels COUNT":0,
        "Collect & use powerup": collect_use_powerup_count,
        "Open rules": open_rules_count,  
        "Open inventory": open_inventory_count,
        "Change interface look like (color, cell, clock shape)": change_interface_count,
        "Change lives count": change_lives_count,
        "Visit leaderboard": visit_leaderboard_count,
        "Perform reset": perform_reset_count,  
        "Perform undo": perform_undo_count,
        "Accept gift": accept_gift_count,
        "Reject gift": reject_gift_count,
    }

    return features

@app.route('/predict', methods=['POST'])
def predict():
    app.logger.info("Received prediction request")
    if not request.json or 'session' not in request.json:
        app.logger.error("Missing session data in request")
        return jsonify({'error': 'Missing session data'}), 400

    data = request.json['session']
    
    try:
        features = calculate_features(data)
        df = pd.DataFrame([features])
        prediction = loaded_model.predict(df)
        prediction_value = int(prediction[0]) if isinstance(prediction[0], np.int64) else prediction[0]
        app.logger.info(f"Prediction result: {prediction_value}")
        return jsonify({'prediction': prediction_value})
    
    except Exception as e:
        app.logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    app.run(debug=True)  # Set to True for development
