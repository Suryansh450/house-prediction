from Flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import json

app = Flask(__name__)

# Load the trained model
with open('Banglore_home_prices_model_pickle', 'rb') as f:
    model = pickle.load(f)

# Get the feature names from the model
feature_names = model.feature_names_in_

# Extract location names (excluding the first 3 features: total_sqft, bath, bhk)
locations = [feature for feature in feature_names[3:]]

@app.route('/')
def index():
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        total_sqft = float(request.form['total_sqft'])
        bath = int(request.form['bath'])
        bhk = int(request.form['bhk'])
        location = request.form['location']
        
        # Create input array with zeros for all features
        input_data = np.zeros(len(feature_names))
        
        # Set the basic features
        input_data[0] = total_sqft  # total_sqft
        input_data[1] = bath        # bath
        input_data[2] = bhk         # bhk
        
        # Set location feature to 1 if it exists in the model
        if location in feature_names:
            location_index = np.where(feature_names == location)[0][0]
            input_data[location_index] = 1
        
        # Make prediction
        prediction = model.predict([input_data])[0]
        
        # Convert to lakhs (assuming the model predicts in lakhs)
        price_lakhs = round(prediction, 2)
        
        return jsonify({
            'success': True,
            'price': price_lakhs,
            'details': {
                'total_sqft': total_sqft,
                'bath': bath,
                'bhk': bhk,
                'location': location
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)