from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
from keplergl import KeplerGl
import os
import openai

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Set OpenAI API key
openai.api_key = "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"  # Replace with your actual OpenAI API key

# Load the dataset
file_path = 'data\Barcelona_Sale.csv'  # Update with your file path
data = pd.read_csv(file_path)

OUTPUT_DIR = "output_maps"  # Directory to store map files
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.route("/process-prompt", methods=["POST"])
def process_prompt():
    prompt = request.json.get("prompt")

    # Validate prompt
    if not prompt or not prompt.strip():
        return jsonify({"error": "The query is empty. Please enter a valid query."}), 400

    # Pre-prompt to guide OpenAI's response
    pre_prompt = f"""
    You are provided with a dataset that contains the following columns: {', '.join(data.columns)}.
    The dataset represents real estate listings in Barcelona, and the columns include:
    - LATITUDE: The latitude of the property location.
    - LONGITUDE: The longitude of the property location.
    - PRICE: The price of the property in Euros.

    Based on the user's request, write Python code to filter the dataset. For example:
    - If the user mentions "south part of Barcelona," filter rows where LATITUDE is less than the median latitude of the dataset.
    - Ensure the result is stored in a Pandas DataFrame named 'filtered_data'.
    - Return only Python code without any additional text or comments.

    User prompt: "{prompt}"
    """

    try:
        # OpenAI API Call
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a Python code generation assistant."},
                {"role": "user", "content": pre_prompt}
            ],
            temperature=0
        )
        generated_code = response['choices'][0]['message']['content']
        print("Raw Generated Code:\n", generated_code)  # Log the raw response for debugging

        # Extract Python code using regex
        import re
        match = re.search(r"```python(.*?)```", generated_code, re.DOTALL)
        if match:
            sanitized_code = match.group(1).strip()
        else:
            sanitized_code = generated_code.strip()  # Fallback if no code block is found

        print("Sanitized Code:\n", sanitized_code)  # Log the sanitized code

        # Execute the sanitized code
        local_vars = {
            "data": data.copy(),
            "pd": pd
        }
        exec(sanitized_code, {}, local_vars)
        filtered_data = local_vars.get("filtered_data")

        # Check if filtered_data is valid
        if filtered_data is None or filtered_data.empty:
            return jsonify({"error": "No data matches the query."}), 400

        # Set fixed latitude, longitude, and zoom for Barcelona
        barcelona_lat = 41.3851  # Central latitude of Barcelona
        barcelona_lon = 2.1734   # Central longitude of Barcelona
        zoom_level = 12          # Default zoom level for the city

        # Generate a new map
        filtered_map = KeplerGl(height=600)
        filtered_map.add_data(data=filtered_data[['LATITUDE', 'LONGITUDE', 'PRICE']], name="Filtered Houses")

        # Update map configuration to always focus on Barcelona
        config = {
            "version": "v1",
            "config": {
                "mapState": {
                    "latitude": barcelona_lat,
                    "longitude": barcelona_lon,
                    "zoom": zoom_level
                }
            }
        }
        filtered_map.config = config

        # Save the map
        new_map_file = os.path.join(OUTPUT_DIR, "filtered_map.html")
        filtered_map.save_to_html(file_name=new_map_file)
        return jsonify({"filteredMap": "filtered_map.html"})
    
    except Exception as e:
        print("Error:", e)
        return jsonify({"error": f"Failed to process the query. Error: {str(e)}"}), 500

@app.route("/maps/<filename>")
def get_map(filename):
    return send_from_directory(OUTPUT_DIR, filename)

if __name__ == "__main__":
    app.run(debug=True)
