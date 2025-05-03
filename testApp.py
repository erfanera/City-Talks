from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
from keplergl import KeplerGl
import os
from openai import OpenAI
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point
import numpy as np

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize OpenAI client
client = OpenAI(api_key="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")

# Set up data directories
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(WORKSPACE_DIR, "data")
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, "output_maps")

print(f"Workspace Directory: {WORKSPACE_DIR}")
print(f"Data Directory: {DATA_DIR}")
print(f"Output Directory: {OUTPUT_DIR}")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load datasets
try:
    houses_data = pd.read_csv(os.path.join(DATA_DIR, "cleaned_barcelona_houses.csv"))
    print(f"Successfully loaded houses data")
except Exception as e:
    print(f"Error loading houses data: {str(e)}")
    houses_data = pd.DataFrame()

try:
    restaurants_data = pd.read_csv(os.path.join(DATA_DIR, "cleaned_barcelona_restaurants.csv"))
    print(f"Successfully loaded restaurants data")
except Exception as e:
    print(f"Error loading restaurants data: {str(e)}")
    restaurants_data = pd.DataFrame()

try:
    supermarkets_data = pd.read_csv(os.path.join(DATA_DIR, "cleaned_barcelona_supermarkets.csv"))
    print(f"Successfully loaded supermarkets data")
except Exception as e:
    print(f"Error loading supermarkets data: {str(e)}")
    supermarkets_data = pd.DataFrame()

def fetch_osm_data(prompt):
    """Fetch POI data from OpenStreetMap based on AI analysis of the prompt."""
    try:
        # Define the bounding box for Barcelona
        north, south, east, west = 41.4695, 41.3203, 2.2267, 2.0652
        
        # Use GPT to analyze the prompt and determine relevant OSM tags
        osm_prompt = f"""
        Available OpenStreetMap data types in Barcelona:

        Healthcare:
        - Hospitals: {{"tags": {{"amenity": "hospital"}}, "name": "Hospitals"}}
        - Clinics: {{"tags": {{"amenity": "clinic"}}, "name": "Clinics"}}
        - Pharmacies: {{"tags": {{"amenity": "pharmacy"}}, "name": "Pharmacies"}}

        Entertainment:
        - Bars: {{"tags": {{"amenity": "bar"}}, "name": "Bars"}}
        - Nightclubs: {{"tags": {{"amenity": "nightclub"}}, "name": "Nightclubs"}}
        - Cinemas: {{"tags": {{"amenity": "cinema"}}, "name": "Cinemas"}}
        - Theatres: {{"tags": {{"amenity": "theatre"}}, "name": "Theatres"}}

        Education:
        - Schools: {{"tags": {{"amenity": "school"}}, "name": "Schools"}}
        - Universities: {{"tags": {{"amenity": "university"}}, "name": "Universities"}}
        - Libraries: {{"tags": {{"amenity": "library"}}, "name": "Libraries"}}

        Transportation:
        - Metro Stations: {{"tags": {{"station": "subway"}}, "name": "Metro Stations"}}
        - Bus Stops: {{"tags": {{"highway": "bus_stop"}}, "name": "Bus Stops"}}
        - Train Stations: {{"tags": {{"station": "train"}}, "name": "Train Stations"}}

        Green Spaces:
        - Parks: {{"tags": {{"leisure": "park"}}, "name": "Parks"}}
        - Gardens: {{"tags": {{"leisure": "garden"}}, "name": "Gardens"}}
        - Playgrounds: {{"tags": {{"leisure": "playground"}}, "name": "Playgrounds"}}

        Based on the query: "{prompt}"
        Return ONLY ONE data type configuration that best matches the query.
        Return ONLY the JSON object for that ONE type, no explanation.
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "system",
                "content": "You are a precise OSM tag selector. Return only ONE exact JSON configuration that best matches the query."
            },
            {
                "role": "user",
                "content": osm_prompt
            }],
            temperature=0
        )
        
        import json
        osm_config = json.loads(response.choices[0].message.content.strip())
        print(f"Selected OSM Config: {osm_config}")
        
        # Fetch the data using OSMnx for just the selected type
        gdf = ox.features_from_bbox(north, south, east, west, osm_config['tags'])
        
        if gdf is not None and not gdf.empty:
            # Convert to DataFrame and extract coordinates
            df = pd.DataFrame({
                'name': gdf['name'].fillna(osm_config['name']),
                'latitude': gdf.geometry.centroid.y,
                'longitude': gdf.geometry.centroid.x,
                'type': osm_config['name']
            })
            print(f"Found {len(df)} {osm_config['name']}")
            return df, f"Found {len(df)} {osm_config['name']} in Barcelona"
        return pd.DataFrame(), f"No {osm_config['name']} found"
    except Exception as e:
        print(f"Error fetching OSM data: {str(e)}")
        return pd.DataFrame(), f"Error: {str(e)}"

@app.route("/process-prompt", methods=["POST"])
def process_prompt():
    try:
        prompt = request.json.get("prompt")
        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400

        # First, classify the query type with a more specific prompt
        classify_prompt = f"""You are a strict classifier that MUST follow these rules:

1. ONLY output one of these exact words: houses, restaurants, supermarkets, osm
2. DO NOT add any explanation, description, or additional text
3. DO NOT use punctuation or spaces
4. Bars, pubs, clubs = osm
5. Restaurants = only food-serving establishments
6. When in doubt = osm

Query: "{prompt}"

Valid responses are ONLY:
houses
restaurants
supermarkets
osm

Any other response format is invalid."""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "system",
                "content": "You are a strict classifier that only outputs single words from a predefined list."
            },
            {
                "role": "user", 
                "content": classify_prompt
            }],
            temperature=0
        )
        
        # Clean and validate the response
        category = response.choices[0].message.content.strip().lower()
        category = ''.join(c for c in category if c.isalnum()).lower()  # Remove any non-alphanumeric characters
        
        # Force bars and pubs queries to use OSM
        if any(word in prompt.lower() for word in ['bar', 'bars', 'pub', 'pubs', 'club', 'clubs', 'nightclub']):
            category = 'osm'
            
        print(f"Query classified as: {category}")

        # Validate the category is one of the expected values
        valid_categories = ['houses', 'restaurants', 'supermarkets', 'osm']
        if category not in valid_categories:
            print(f"Invalid category received: '{category}'. Defaulting to 'osm'")
            category = 'osm'

        filtered_data = pd.DataFrame()
        summary = ""

        if category == "osm":
            # Use OSM agent to fetch data
            filtered_data, summary = fetch_osm_data(prompt)
            if filtered_data.empty:
                return jsonify({"error": "No data found matching the criteria"}), 400
        else:
            # Handle regular dataset queries
            dataset = {
                'houses': houses_data,
                'restaurants': restaurants_data,
                'supermarkets': supermarkets_data
            }.get(category)
            
            if dataset is None or dataset.empty:
                return jsonify({"error": f"No data available for {category}"}), 400

            # Get filtering criteria from GPT
            filter_prompt = f"""
            For the query "{prompt}", extract filtering criteria for {category}.
            Available columns: {', '.join(dataset.columns)}
            Return a JSON object with:
            1. conditions: list of dictionaries with column, operator (==, >, <, >=, <=, in, contains), and value
            2. sort_by: column name to sort by (optional)
            3. sort_order: "asc" or "desc" (optional)
            4. limit: number of results to return (optional)
            Example: {{"conditions": [{{"column": "price", "operator": "<", "value": 300000}}], "sort_by": "price", "sort_order": "asc"}}
            """
            
            filter_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": filter_prompt}],
                temperature=0
            )
            
            import json
            filter_criteria = json.loads(filter_response.choices[0].message.content.strip())
            
            # Apply filters
            filtered_data = dataset.copy()
            for condition in filter_criteria.get('conditions', []):
                column = condition['column']
                operator = condition['operator']
                value = condition['value']
                
                if operator == '==':
                    filtered_data = filtered_data[filtered_data[column] == value]
                elif operator == '>':
                    filtered_data = filtered_data[filtered_data[column] > value]
                elif operator == '<':
                    filtered_data = filtered_data[filtered_data[column] < value]
                elif operator == '>=':
                    filtered_data = filtered_data[filtered_data[column] >= value]
                elif operator == '<=':
                    filtered_data = filtered_data[filtered_data[column] <= value]
                elif operator == 'in':
                    filtered_data = filtered_data[filtered_data[column].isin(value)]
                elif operator == 'contains':
                    filtered_data = filtered_data[filtered_data[column].str.contains(value, case=False, na=False)]
            
            # Apply sorting
            if 'sort_by' in filter_criteria:
                sort_ascending = filter_criteria.get('sort_order', 'asc') == 'asc'
                filtered_data = filtered_data.sort_values(
                    by=filter_criteria['sort_by'],
                    ascending=sort_ascending
                )
            
            # Apply limit
            if 'limit' in filter_criteria:
                filtered_data = filtered_data.head(filter_criteria['limit'])

            summary = f"Found {len(filtered_data)} matching {category}"

        if filtered_data.empty:
            return jsonify({"error": "No data found matching the criteria"}), 400

        # Generate map
        barcelona_lat = 41.3851
        barcelona_lon = 2.1734
        zoom_level = 12

        filtered_map = KeplerGl(height=600)
        filtered_map.add_data(data=filtered_data[['latitude', 'longitude']], name=category.title())

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
        output_map_path = os.path.join(OUTPUT_DIR, "filtered_map.html")
        filtered_map.save_to_html(file_name=output_map_path)

        return jsonify({
            "success": True,
            "filteredMap": "filtered_map.html",
            "count": len(filtered_data),
            "summary": summary
        })

    except Exception as e:
        print(f"Error in process-prompt: {str(e)}")
        import traceback
        print(f"Traceback:\n{traceback.format_exc()}")
        return jsonify({
            "error": str(e),
            "summary": "An error occurred while processing your request."
        }), 500

@app.route("/maps/<filename>")
def get_map(filename):
    return send_from_directory(OUTPUT_DIR, filename)

if __name__ == "__main__":
    app.run(debug=True, port=5000) 