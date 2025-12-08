from flask import Flask, request, jsonify
from flask_cors import CORS
from model import generate_forecast

# Initialize the Flask application
app = Flask(__name__)
# Enable CORS for all routes, allowing frontend access
CORS(app)

@app.route('/forecast', methods=['POST'])
def forecast():
    """
    API endpoint to generate a travel cost forecast.
    Expects a JSON payload with 'destinationCountry', 'durationDays', and 'month'.
    """
    if not request.json:
        return jsonify({"error": "Missing JSON in request"}), 400

    required_params = ['destinationCountry', 'durationDays', 'month']
    if not all(param in request.json for param in required_params):
        return jsonify({"error": "Missing one or more required parameters"}), 400

    try:
        params = request.get_json()

        # Call the machine learning model to get the forecast
        result = generate_forecast(params)

        return jsonify(result)

    except FileNotFoundError as e:
        # Handle cases where a model for a specific country isn't found
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        # Generic error handler for other issues
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    # Running in debug mode is not recommended for production
    # For a real deployment, a production-ready WSGI server like Gunicorn or uWSGI should be used.
    app.run(debug=True, port=5000)
