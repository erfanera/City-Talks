from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import folium
from folium import plugins
import os
from openai import OpenAI
import hashlib
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg before importing pyplot
import matplotlib.pyplot as plt
import io
import base64
import seaborn as sns
from keplergl import KeplerGl
import json
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point
from scipy.spatial import cKDTree
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import time

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize OpenAI client
client = OpenAI(api_key="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")

# Set up data directories
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(WORKSPACE_DIR)  # Just go up one level to interactive-map
DATA_DIR = os.path.join(PROJECT_ROOT, "data")  # Now points to D:\IaaC\DataVis\data_vis_project\code\interactive-map\data
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, "data", "output_maps")

print(f"Workspace Directory: {WORKSPACE_DIR}")
print(f"Project Root: {PROJECT_ROOT}")
print(f"Data Directory: {DATA_DIR}")
print(f"Output Directory: {OUTPUT_DIR}")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load amenity types
try:
    amenity_types_df = pd.read_csv(os.path.join(DATA_DIR, "barcelona_amenity_types.csv"))
    AMENITY_TYPES = amenity_types_df['amenity_type'].tolist()
    print(f"Successfully loaded {len(AMENITY_TYPES)} amenity types")
except Exception as e:
    print(f"Error loading amenity types: {e}")
    AMENITY_TYPES = []

# Add Barcelona locations database
BARCELONA_LOCATIONS = {
    # Educational Institutions
    "iaac": {"lat": 41.3984, "lon": 2.1892, "type": "education"},
    "institute for advanced architecture": {"lat": 41.3984, "lon": 2.1892, "type": "education"},
    "institute for advanced architecture of catalonia": {"lat": 41.3984, "lon": 2.1892, "type": "education"},
    
    # Metro stations (Line 4 - Yellow Line)
    "maragall metro station": {"lat": 41.4225, "lon": 2.1809, "type": "metro_station"},
    "llucmajor metro station": {"lat": 41.4285, "lon": 2.1744, "type": "metro_station"},
    "verdaguer metro station": {"lat": 41.4022, "lon": 2.1701, "type": "metro_station"},
    "joanic metro station": {"lat": 41.4068, "lon": 2.1701, "type": "metro_station"},
    "alfons x metro station": {"lat": 41.4068, "lon": 2.1701, "type": "metro_station"},
    "guinardó metro station": {"lat": 41.4192, "lon": 2.1781, "type": "metro_station"},
    
    # Major landmarks
    "sagrada familia": {"lat": 41.4036, "lon": 2.1744, "type": "landmark"},
    "park guell": {"lat": 41.4145, "lon": 2.1527, "type": "park"},
    "la rambla": {"lat": 41.3851, "lon": 2.1700, "type": "street"},
    "plaza catalunya": {"lat": 41.3874, "lon": 2.1700, "type": "plaza"},
    "camp nou": {"lat": 41.3809, "lon": 2.1228, "type": "stadium"},
    
    # Neighborhoods
    "gracia": {"lat": 41.4033, "lon": 2.1526, "type": "neighborhood"},
    "eixample": {"lat": 41.3915, "lon": 2.1709, "type": "neighborhood"},
    "gothic quarter": {"lat": 41.3833, "lon": 2.1777, "type": "neighborhood"},
    "barceloneta": {"lat": 41.3766, "lon": 2.1900, "type": "neighborhood"},
    "poblenou": {"lat": 41.4037, "lon": 2.1997, "type": "neighborhood"}
}

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    # Radius of earth in kilometers
    r = 6371
    return c * r

def find_nearby_amenities(gdf, center_lat, center_lon, radius_km=1.0):
    """
    Find amenities within a specified radius using KDTree for efficient search.
    
    Args:
        gdf: GeoDataFrame with amenity data
        center_lat: latitude of the center point
        center_lon: longitude of the center point
        radius_km: search radius in kilometers (default 1km)
    
    Returns:
        GeoDataFrame containing only the amenities within the radius
    """
    if gdf.empty:
        return gdf
        
    # Extract coordinates
    coords = np.array([[geom.centroid.y if hasattr(geom, 'centroid') else geom.y,
                       geom.centroid.x if hasattr(geom, 'centroid') else geom.x] 
                      for geom in gdf.geometry])
    
    # Build KD-tree
    tree = cKDTree(coords)
    
    # Convert radius to approximate degrees (rough approximation)
    # 1 degree ≈ 111 km at the equator
    radius_deg = radius_km / 111.0
    
    # Find all points within radius
    indices = tree.query_ball_point([center_lat, center_lon], radius_deg)
    
    if not indices:
        return gpd.GeoDataFrame(columns=gdf.columns)
    
    # Filter the original GeoDataFrame
    nearby_gdf = gdf.iloc[indices].copy()
    
    # Calculate exact distances using Haversine formula
    nearby_gdf['distance_km'] = nearby_gdf.geometry.apply(
        lambda geom: haversine_distance(
            center_lat, center_lon,
            geom.centroid.y if hasattr(geom, 'centroid') else geom.y,
            geom.centroid.x if hasattr(geom, 'centroid') else geom.x
        )
    )
    
    # Filter using exact distances
    nearby_gdf = nearby_gdf[nearby_gdf['distance_km'] <= radius_km]
    
    # Sort by distance
    return nearby_gdf.sort_values('distance_km')

def search_location_in_barcelona(query):
    """
    Search for a location in Barcelona using Nominatim API.
    Returns (lat, lon, display_name) or None if not found.
    """
    geolocator = Nominatim(user_agent="barcelona_amenities_app")
    try:
        # First try with Barcelona context
        search_query = f"{query}, Barcelona, Spain"
        
        # Add retry mechanism
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Search with area restriction to Barcelona's bounding box
                location = geolocator.geocode(
                    search_query,
                    viewbox=[2.0525, 41.3200, 2.2275, 41.4900],
                    bounded=True,
                    language="en"
                )
                
                if location:
                    print(f"Found location: {location.address}")
                    return location.latitude, location.longitude, location.address
                
                # If not found, try without strict bounding
                location = geolocator.geocode(
                    search_query,
                    viewbox=[2.0525, 41.3200, 2.2275, 41.4900],
                    bounded=False,
                    language="en"
                )
                
                if location:
                    if (2.0525 <= location.longitude <= 2.2275 and 
                        41.3200 <= location.latitude <= 41.4900):
                        print(f"Found location: {location.address}")
                        return location.latitude, location.longitude, location.address
                break
                
            except GeocoderTimedOut:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                raise
        
        print(f"Location not found: {query}")
        return None
        
    except Exception as e:
        print(f"Error in location search: {str(e)}")
        return None

def find_location_in_database(query):
    """
    Find a location using both local database and real-time search.
    Returns (lat, lon, location_name) or None if not found.
    """
    query = query.lower().strip()
    
    # First check our local database for common locations
    if query in BARCELONA_LOCATIONS:
        loc = BARCELONA_LOCATIONS[query]
        print(f"Found in local database: {query}")
        return loc["lat"], loc["lon"], query
        
    # Try to match parts in local database
    for loc_name, loc_data in BARCELONA_LOCATIONS.items():
        # Check if any part of the query matches any part of the location name
        query_parts = set(query.split())
        loc_parts = set(loc_name.split())
        if query_parts & loc_parts:  # If there's any overlap in words
            print(f"Found partial match in local database: {loc_name}")
            return loc_data["lat"], loc_data["lon"], loc_name
    
    # If not found in local database, try real-time search
    print(f"Searching online for location: {query}")
    result = search_location_in_barcelona(query)
    if result:
        lat, lon, address = result
        # Cache this result for future use
        BARCELONA_LOCATIONS[query] = {
            "lat": lat,
            "lon": lon,
            "type": "searched_location"
        }
        return lat, lon, address
    
    return None

def extract_location_and_amenity(query):
    """
    Smart agent that autonomously understands and classifies location queries.
    Uses advanced context understanding to determine amenity types and locations.
    """
    smart_prompt = f"""You are an advanced AI agent specialized in finding amenities and analyzing mobility patterns in Barcelona.
Your task is to intelligently interpret the user's request, focusing on finding relevant places and transportation insights.

Query: "{query}"

UNDERSTANDING CATEGORIES:
1. Medical Infrastructure:
   - Hospitals: hospital (primary)
   - Clinics: clinic, doctors
   - Pharmacies: pharmacy
   - Healthcare: healthcare, dentist

2. Location Context:
   - North: Above 41.4000 latitude
   - South: Below 41.3700 latitude
   - East: Above 2.1800 longitude
   - West: Below 2.1500 longitude
   - Center: Around (41.3851, 2.1734)

Return a JSON object in this exact format (no additional text):
{{
    "amenity_types": ["primary_type", "alternative_type1", "alternative_type2"],
    "location": "extracted_location or 'all' for city-wide",
    "radius_km": float or null,
    "explanation": "Detailed explanation focusing on amenities and mobility",
    "brand_name": null,
    "mobility_analysis": false
}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": smart_prompt}],
            temperature=0.1
        )
        
        response_text = response.choices[0].message.content.strip()
        print(f"Raw agent response: {response_text}")
        
        try:
            result = json.loads(response_text)
            print(f"Smart agent analysis: {result['explanation']}")
            
            # Get the primary amenity type
            primary_type = result['amenity_types'][0]
            print(f"Selected primary amenity type: {primary_type}")
            
            # Handle location and get coordinates
            location = result['location'].lower()
            bbox = None
            center_coords = None
            radius_km = result.get('radius_km', 5.0)  # Default to 5.0 if not specified
            
            # Special handling for directional locations
            if 'north' in location:
                # North Barcelona coordinates
                center_coords = (41.4300, 2.1734)
                radius_km = 2.0  # Larger radius for area coverage
                radius_deg = radius_km / 111.0
                bbox = [
                    2.1734 - radius_deg,  # minx
                    41.4000,  # miny (north boundary)
                    2.1734 + radius_deg,  # maxx
                    41.4500   # maxy
                ]
            elif location != 'all':
                location_result = find_location_in_database(location)
                if location_result:
                    lat, lon, matched_name = location_result
                    center_coords = (lat, lon)
                    radius_deg = radius_km / 111.0
                    bbox = [
                        lon - radius_deg,
                        lat - radius_deg,
                        lon + radius_deg,
                        lat + radius_deg
                    ]
            else:
                # For city-wide searches, use Barcelona's bounding box
                bbox = [2.0525, 41.3200, 2.2275, 41.4900]  # Barcelona bounding box
                center_coords = (41.3851, 2.1734)  # Barcelona center
                radius_km = 10.0  # Large radius for city-wide search
                print("Using Barcelona bounding box for city-wide search")
            
            return (primary_type, location, bbox, 
                    radius_km, 
                    center_coords, 
                    result['amenity_types'][1:],
                    result.get('brand_name'),
                    result.get('mobility_analysis', False))
                    
        except json.JSONDecodeError as e:
            print(f"Error parsing agent response: {e}")
            print("Falling back to default processing...")
            
            # Default processing for medical queries
            query_lower = query.lower()
            if any(word in query_lower for word in ['medical', 'hospital', 'clinic', 'healthcare']):
                return ('hospital', 'north side of barcelona', None, 2.0, (41.4300, 2.1734), ['clinic', 'doctors'], None, False)
            elif 'pharmacy' in query_lower:
                return ('pharmacy', 'north side of barcelona', None, 2.0, (41.4300, 2.1734), ['chemist'], None, False)
            
            # General fallback
            return ('hospital', 'all', None, 2.0, None, ['clinic', 'doctors'], None, False)
            
    except Exception as e:
        print(f"Error in smart agent: {str(e)}")
        return ('hospital', 'all', None, 2.0, None, ['clinic', 'doctors'], None, False)

def fetch_amenity_data(query, center_lat=41.3851, center_lon=2.1734, radius=5000, alternatives=None, brand_name=None, mobility_analysis=False):
    """
    Fetch amenity data from OpenStreetMap based on smart agent analysis.
    """
    try:
        # Use smart agent to analyze query
        (primary_type, location, bbox, radius_km, 
         center_coords, alternative_types, brand_name, mobility_analysis) = extract_location_and_amenity(query)
        
        if not primary_type:
            print("Smart agent couldn't determine amenity type")
            return pd.DataFrame(columns=['name', 'latitude', 'longitude', 'type', 'category', 'price', 'description', 'address', 'distance_km'])
            
        print(f"Smart agent selected: Primary type: {primary_type}, Alternatives: {alternative_types}")
        print(f"Searching within {radius_km}km radius")
        
        # Initialize empty DataFrame for results
        all_results = pd.DataFrame()
        
        # Set search center
        if center_coords:
            search_lat, search_lon = center_coords
        else:
            search_lat, search_lon = center_lat, center_lon
        
        # Handle parks and leisure areas
        if primary_type == 'park':
            # Try leisure tag for parks
            leisure_tags = {"leisure": ["park", "garden"]}
            try:
                search_radius_km = radius_km * 1.2  # Add 20% to account for edge cases
                center_point = (search_lat, search_lon)
                gdf = ox.features_from_point(center_point, leisure_tags, dist=search_radius_km * 1000)
                
                if not gdf.empty:
                    nearby_gdf = find_nearby_amenities(gdf, search_lat, search_lon, radius_km)
                    if not nearby_gdf.empty:
                        df = process_osm_data(nearby_gdf, 'park', location)
                        df['distance_km'] = nearby_gdf['distance_km']
                        all_results = pd.concat([all_results, df], ignore_index=True)
                        print(f"Found {len(df)} parks/gardens within {radius_km}km")
            except Exception as e:
                print(f"Error with leisure search: {e}")
            
            # Try landuse tag for parks
            landuse_tags = {"landuse": ["grass", "recreation_ground"]}
            try:
                gdf = ox.features_from_point(center_point, landuse_tags, dist=search_radius_km * 1000)
                if not gdf.empty:
                    nearby_gdf = find_nearby_amenities(gdf, search_lat, search_lon, radius_km)
                    if not nearby_gdf.empty:
                        df = process_osm_data(nearby_gdf, 'park', location)
                        df['distance_km'] = nearby_gdf['distance_km']
                        all_results = pd.concat([all_results, df], ignore_index=True)
                        print(f"Found {len(df)} recreational areas within {radius_km}km")
            except Exception as e:
                print(f"Error with landuse search: {e}")
        else:
            # Handle other amenities as before
            tags = {"amenity": primary_type}
            try:
                search_radius_km = radius_km * 1.2
                center_point = (search_lat, search_lon)
                gdf = ox.features_from_point(center_point, tags, dist=search_radius_km * 1000)
                
                if not gdf.empty:
                    nearby_gdf = find_nearby_amenities(gdf, search_lat, search_lon, radius_km)
                    if not nearby_gdf.empty:
                        df = process_osm_data(nearby_gdf, primary_type, location)
                        df['distance_km'] = nearby_gdf['distance_km']
                        all_results = pd.concat([all_results, df], ignore_index=True)
                        print(f"Found {len(df)} {primary_type} locations within {radius_km}km")
            except Exception as e:
                print(f"Error with primary type search: {e}")
        
        # Try alternative types if needed
        if len(all_results) < 3 and alternative_types:
            print("Trying alternative amenity types...")
            for alt_type in alternative_types:
                alt_tags = {"amenity": alt_type}
                try:
                    alt_gdf = ox.features_from_point(center_point, alt_tags, dist=search_radius_km * 1000)
                    if not alt_gdf.empty:
                        nearby_alt_gdf = find_nearby_amenities(alt_gdf, search_lat, search_lon, radius_km)
                        if not nearby_alt_gdf.empty:
                            alt_df = process_osm_data(nearby_alt_gdf, alt_type, location)
                            alt_df['distance_km'] = nearby_alt_gdf['distance_km']
                            all_results = pd.concat([all_results, alt_df], ignore_index=True)
                            print(f"Found {len(alt_df)} {alt_type} locations within {radius_km}km")
                except Exception as e:
                    print(f"Error fetching alternative type {alt_type}: {e}")
                    
        if all_results.empty:
            print("No results found for any amenity type")
            return pd.DataFrame(columns=['name', 'latitude', 'longitude', 'type', 'category', 'price', 'description', 'address', 'distance_km'])
        
        # Sort by distance and remove duplicates
        all_results = all_results.sort_values('distance_km').drop_duplicates(subset=['name', 'latitude', 'longitude'])
        print(f"Found total of {len(all_results)} locations, sorted by distance")
        return all_results
        
    except Exception as e:
        print(f"Error in fetch_amenity_data: {str(e)}")
        return pd.DataFrame(columns=['name', 'latitude', 'longitude', 'type', 'category', 'price', 'description', 'address', 'distance_km'])

def process_osm_data(gdf, amenity_type, location):
    """Helper function to process OSM data into the required format."""
    df = pd.DataFrame(gdf)
    df = df[df.geometry.notna()]
    
    # Extract coordinates
    coords = df.geometry.apply(lambda geom: pd.Series({
        'latitude': geom.centroid.y if hasattr(geom, 'centroid') else geom.y,
        'longitude': geom.centroid.x if hasattr(geom, 'centroid') else geom.x
    }))
    
    df[['latitude', 'longitude']] = coords

    # Handle name field
    df['name'] = df.get('name', pd.Series("Unnamed", index=df.index))

    # Add required columns
    df['type'] = amenity_type
    df['category'] = 'amenity'
    df['price'] = None
    
    # Extract additional information
    df['description'] = df.apply(lambda row: 
        f"Type: {amenity_type}\n"
        f"Opening hours: {row.get('opening_hours', 'Not available')}\n"
        f"Phone: {row.get('phone', 'Not available')}\n"
        f"Website: {row.get('website', 'Not available')}\n"
        f"Location: {location if location != 'all' else 'Barcelona'}", axis=1)
    
    # Construct address
    df['address'] = df.apply(lambda row: 
        f"{row.get('addr:street', '')} {row.get('addr:housenumber', '')}".strip() or 
        f"{row.get('address', '')}".strip() or 
        "Address not available", axis=1)
    
    return df[['name', 'latitude', 'longitude', 'type', 'category', 'price', 'description', 'address']]

# Load datasets
houses_data = pd.DataFrame()
try:
    houses_data = pd.read_csv(os.path.join(DATA_DIR, "cleaned_barcelona_houses.csv"))
    print(f"Successfully loaded houses data from {os.path.join(DATA_DIR, 'cleaned_barcelona_houses.csv')}")
except Exception as e:
    print(f"Error loading houses data from {os.path.join(DATA_DIR, 'cleaned_barcelona_houses.csv')}: {str(e)}")
    houses_data = pd.DataFrame()

# Load CO2 emissions data
co2_data = pd.DataFrame()
try:
    co2_data = pd.read_csv(os.path.join(DATA_DIR, "barcelona_co2_emissions_zones.csv"))
    print(f"Successfully loaded CO2 emissions data from {os.path.join(DATA_DIR, 'barcelona_co2_emissions_zones.csv')}")
except Exception as e:
    print(f"Error loading CO2 emissions data from {os.path.join(DATA_DIR, 'barcelona_co2_emissions_zones.csv')}: {str(e)}")
    co2_data = pd.DataFrame()

restaurants_data = pd.DataFrame()
try:
    restaurants_data = pd.read_csv(os.path.join(DATA_DIR, "cleaned_barcelona_restaurants.csv"))
    print(f"Successfully loaded restaurants data from {os.path.join(DATA_DIR, 'cleaned_barcelona_restaurants.csv')}")
except Exception as e:
    print(f"Error loading restaurants data from {os.path.join(DATA_DIR, 'cleaned_barcelona_restaurants.csv')}: {str(e)}")
    restaurants_data = pd.DataFrame()

supermarkets_data = pd.DataFrame()
try:
    supermarkets_data = pd.read_csv(os.path.join(DATA_DIR, "cleaned_barcelona_supermarkets.csv"))
    print(f"Successfully loaded supermarkets data from {os.path.join(DATA_DIR, 'cleaned_barcelona_supermarkets.csv')}")
except Exception as e:
    print(f"Error loading supermarkets data from {os.path.join(DATA_DIR, 'cleaned_barcelona_supermarkets.csv')}: {str(e)}")
    supermarkets_data = pd.DataFrame()

def generate_map_config():
    """Generate a standard map configuration."""
    barcelona_lat = 41.3851
    barcelona_lon = 2.1734
    zoom_level = 12

    return {
        "version": "v1",
        "config": {
            "mapState": {
                "latitude": barcelona_lat,
                "longitude": barcelona_lon,
                "zoom": zoom_level,
                "pitch": 0,
                "bearing": 0
            },
            "visState": {
                "layers": [
                    {
                        "id": "point_layer",
                        "type": "point",
                        "config": {
                            "dataId": "filtered_data",
                            "label": "Point Data",
                            "columns": {
                                "lat": "latitude",
                                "lng": "longitude"
                            },
                            "isVisible": True,
                            "visConfig": {
                                "radius": 10,
                                "fixedRadius": False,
                                "opacity": 1,
                                "color": [255, 165, 0],
                                "outline": False
                            }
                        }
                    },
                    {
                        "id": "grid_layer",
                        "type": "grid",
                        "config": {
                            "dataId": "filtered_data",
                            "label": "Grid Density",
                            "columns": {
                                "lat": "latitude",
                                "lng": "longitude"
                            },
                            "isVisible": True,
                            "visConfig": {
                                "worldUnitSize": 0.3,
                                "opacity": 0.4,
                                "colorRange": {
                                    "colors": [
                                        "#A7B4E0",
                                        "#7288D4",
                                        "#415CA7",
                                        "#2B3F75",
                                        "#18263F"
                                    ]
                                }
                            }
                        }
                    }
                ],
                "interactionConfig": {
                    "tooltip": {
                        "enabled": True,
                        "fieldsToShow": {
                            "filtered_data": ["name", "latitude", "longitude"]
                        }
                    },
                    "brush": {
                        "size": 1.5,
                        "enabled": False
                    },
                    "coordinate": {
                        "enabled": True
                    },
                    "featureClick": {
                        "enabled": True,
                        "fieldsToShow": {
                            "filtered_data": ["name", "latitude", "longitude"]
                        }
                    }
                }
            },
            "mapStyle": {
                "styleType": "positron-nolabels",
                "topLayerGroups": {
                    "label": True,
                    "road": True,
                    "border": False,
                    "building": False,
                    "land": True,
                    "water": True
                },
                "visibleLayerGroups": {
                    "label": True,
                    "road": True,
                    "land": True,
                    "water": True
                }
            }
        }
    }

@app.route("/maps/<filename>")
def get_map(filename):
    try:
        print(f"Serving map file: {filename}")
        # Ensure the file exists
        file_path = os.path.join(OUTPUT_DIR, filename)
        if not os.path.exists(file_path):
            print(f"Map file not found: {file_path}")
            return "Map file not found", 404
            
        # Set correct MIME type and cache control headers
        response = send_from_directory(
            OUTPUT_DIR, 
            filename, 
            mimetype='text/html'
        )
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        print(f"Successfully serving map file: {filename}")
        return response
    except Exception as e:
        print(f"Error serving map file: {str(e)}")
        return f"Error serving map: {str(e)}", 500

def analyze_prompt(prompt):
    """
    Main agent that analyzes the prompt and determines which specialized agent to use.
    Returns a tuple of (agent_type, specific_query)
    """
    analysis_prompt = f"""You are a query analyzer. Your task is to analyze the following query and determine the type of data being requested.

Query: "{prompt}"

Analyze the query and respond with ONLY a JSON object in the following format:
{{
    "agent": "<agent_type>",
    "query": "<cleaned_query>"
}}

Where <agent_type> must be exactly one of: "houses", "restaurants", "supermarkets", "amenity", "bicing", "co2"
And <cleaned_query> should be the original query, cleaned if needed.

Rules for determining agent_type:
1. houses: for queries about houses, apartments, or properties for sale/rent
2. restaurants: for queries about restaurants, cafes, or dining places
3. supermarkets: for queries about supermarkets or grocery stores
4. bicing: for queries about Bicing bike sharing stations, bike availability, or bike usage patterns
5. co2: for queries about CO2 emissions, air quality, or environmental zones
6. amenity: for queries about any other location or place (like hospitals, schools, bars, etc.)

Examples of valid responses:
{{"agent": "houses", "query": "show me houses under 500k"}}
{{"agent": "bicing", "query": "show bicing stations near sagrada familia"}}
{{"agent": "co2", "query": "show me areas with highest CO2 emissions"}}
{{"agent": "amenity", "query": "find bars in La Rambla"}}"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": analysis_prompt}],
            temperature=0.1
        )
        
        response_text = response.choices[0].message.content.strip()
        
        try:
            result = json.loads(response_text)
            
            if not isinstance(result, dict):
                raise ValueError("Response is not a dictionary")
            if "agent" not in result or "query" not in result:
                raise ValueError("Response missing required fields")
            if result["agent"] not in ["houses", "restaurants", "supermarkets", "amenity", "bicing", "co2"]:
                raise ValueError(f"Invalid agent type: {result['agent']}")
                
            return result["agent"], result["query"]
            
        except json.JSONDecodeError:
            print(f"Invalid JSON response: {response_text}")
            # Check for Bicing-related keywords
            query_lower = prompt.lower()
            if any(word in query_lower for word in ["bicing", "bike sharing", "bike station"]):
                return "bicing", prompt
            return "amenity", prompt
            
    except Exception as e:
        print(f"Error in analyze_prompt: {str(e)}")
        return "amenity", prompt

def fetch_bicing_stations(query):
    """
    Fetch Bicing station data from OpenStreetMap.
    """
    try:
        # Extract location from query
        location = None
        if "near" in query.lower():
            location = query.lower().split("near")[1].strip()
        elif "close to" in query.lower():
            location = query.lower().split("close to")[1].strip()
        elif "around" in query.lower():
            location = query.lower().split("around")[1].strip()
            
        # Set default search parameters
        search_lat, search_lon = 41.3851, 2.1734  # Barcelona center
        radius_km = 1.0
        
        # Try to find the specified location
        if location:
            location_result = find_location_in_database(location)
            if location_result:
                search_lat, search_lon, _ = location_result
                print(f"Found location: {location} at {search_lat}, {search_lon}")
        
        # Search for bicycle rental stations
        tags = {
            "amenity": "bicycle_rental",
            "network": "Bicing"
        }
        
        try:
            # Add 20% to radius for better coverage
            search_radius_km = radius_km * 1.2
            center_point = (search_lat, search_lon)
            
            # Fetch stations from OSM
            gdf = ox.features_from_point(center_point, tags, dist=search_radius_km * 1000)
            
            if not gdf.empty:
                # Find nearby stations
                nearby_gdf = find_nearby_amenities(gdf, search_lat, search_lon, radius_km)
                if not nearby_gdf.empty:
                    # Process the data
                    df = pd.DataFrame()
                    df['name'] = nearby_gdf.get('name', 'Bicing Station')
                    df['latitude'] = nearby_gdf.geometry.apply(lambda p: p.y)
                    df['longitude'] = nearby_gdf.geometry.apply(lambda p: p.x)
                    df['type'] = 'bicing_station'
                    df['category'] = 'transport'
                    df['price'] = None
                    df['description'] = nearby_gdf.apply(
                        lambda row: f"Bicing Station\n"
                                  f"Capacity: {row.get('capacity', 'Unknown')}\n"
                                  f"Operator: Bicing Barcelona", axis=1)
                    df['address'] = nearby_gdf.apply(
                        lambda row: f"{row.get('addr:street', '')} {row.get('addr:housenumber', '')}".strip() or 
                        "Address not available", axis=1)
                    df['distance_km'] = nearby_gdf['distance_km']
                    
                    print(f"Found {len(df)} Bicing stations within {radius_km}km")
                    return df
                    
            print("No Bicing stations found in the specified area")
            return pd.DataFrame()
            
        except Exception as e:
            print(f"Error fetching Bicing stations: {e}")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error in fetch_bicing_stations: {e}")
        return pd.DataFrame()

def fetch_restaurant_data(query, center_lat=41.3851, center_lon=2.1734, radius_km=1.0):
    """
    Fetch restaurant data from OpenStreetMap with cuisine type filtering.
    """
    try:
        # Extract cuisine type and location from query
        cuisine_prompt = f"""Analyze this restaurant query and extract cuisine and location details:

Query: "{query}"

Return ONLY a JSON object in this format:
{{
    "cuisine_type": "extracted cuisine type or null if none specified",
    "location": "extracted location or 'all' if none specified",
    "radius_km": float (default 1.0),
    "explanation": "brief explanation of what was understood from the query"
}}

Example responses:
{{"cuisine_type": "vegan", "location": "sagrada familia", "radius_km": 1.0, "explanation": "Looking for vegan restaurants near Sagrada Familia"}}
{{"cuisine_type": "italian", "location": "gothic quarter", "radius_km": 0.5, "explanation": "Searching for Italian restaurants in Gothic Quarter"}}"""

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": cuisine_prompt}],
            temperature=0.1
        )
        
        result = json.loads(response.choices[0].message.content.strip())
        print(f"Query analysis: {result['explanation']}")
        
        # Set search parameters
        location = result['location']
        cuisine_type = result['cuisine_type']
        radius_km = result.get('radius_km', 1.0)
        
        # Try to find the location
        if location and location != 'all':
            location_result = find_location_in_database(location)
            if location_result:
                center_lat, center_lon, _ = location_result
                print(f"Found location: {location} at {center_lat}, {center_lon}")
        
        # Prepare OSM tags for restaurant search
        tags = {
            "amenity": "restaurant"
        }
        
        if cuisine_type:
            tags["cuisine"] = cuisine_type
            
        try:
            # Add 20% to radius for better coverage
            search_radius_km = radius_km * 1.2
            center_point = (center_lat, center_lon)
            
            # Fetch restaurants from OSM
            gdf = ox.features_from_point(center_point, tags, dist=search_radius_km * 1000)
            
            if not gdf.empty:
                # Find nearby restaurants
                nearby_gdf = find_nearby_amenities(gdf, center_lat, center_lon, radius_km)
                if not nearby_gdf.empty:
                    # Process the data
                    df = pd.DataFrame()
                    df['name'] = nearby_gdf.get('name', 'Unnamed Restaurant')
                    df['latitude'] = nearby_gdf.geometry.apply(lambda p: p.y)
                    df['longitude'] = nearby_gdf.geometry.apply(lambda p: p.x)
                    df['type'] = 'restaurant'
                    df['category'] = 'food'
                    df['cuisine'] = nearby_gdf.get('cuisine', 'Not specified')
                    df['price'] = nearby_gdf.get('price_range', None)
                    df['description'] = nearby_gdf.apply(
                        lambda row: f"Restaurant\n"
                                  f"Cuisine: {row.get('cuisine', 'Not specified')}\n"
                                  f"Opening hours: {row.get('opening_hours', 'Not available')}\n"
                                  f"Price range: {row.get('price_range', 'Not available')}\n"
                                  f"Phone: {row.get('phone', 'Not available')}", axis=1)
                    df['address'] = nearby_gdf.apply(
                        lambda row: f"{row.get('addr:street', '')} {row.get('addr:housenumber', '')}".strip() or 
                        "Address not available", axis=1)
                    df['distance_km'] = nearby_gdf['distance_km']
                    
                    print(f"Found {len(df)} restaurants within {radius_km}km")
                    return df
                    
            # If no results with exact cuisine match, try alternative tags
            if cuisine_type and gdf.empty:
                print(f"No results for cuisine '{cuisine_type}', trying diet-specific tags...")
                diet_tags = {
                    "vegan": {"diet:vegan": "yes"},
                    "vegetarian": {"diet:vegetarian": "yes"},
                    "halal": {"diet:halal": "yes"},
                    "kosher": {"diet:kosher": "yes"}
                }
                
                if cuisine_type.lower() in diet_tags:
                    alt_tags = {"amenity": "restaurant", **diet_tags[cuisine_type.lower()]}
                    gdf = ox.features_from_point(center_point, alt_tags, dist=search_radius_km * 1000)
                    
                    if not gdf.empty:
                        nearby_gdf = find_nearby_amenities(gdf, center_lat, center_lon, radius_km)
                        if not nearby_gdf.empty:
                            df = process_osm_data(nearby_gdf, 'restaurant', location)
                            df['cuisine'] = cuisine_type
                            df['distance_km'] = nearby_gdf['distance_km']
                            print(f"Found {len(df)} {cuisine_type} restaurants within {radius_km}km")
                            return df
            
            print("No restaurants found matching the criteria")
            return pd.DataFrame()
            
        except Exception as e:
            print(f"Error fetching restaurants: {e}")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error in fetch_restaurant_data: {e}")
        return pd.DataFrame()

def process_co2_data(query):
    """
    Process CO2 emissions data based on user query.
    Returns filtered data and visualization information.
    """
    try:
        # Analyze the query to understand what information is requested
        analysis_prompt = f"""Analyze this CO2 emissions query and determine what information to show:

Query: "{query}"

Return ONLY a JSON object in this format:
{{
    "analysis_type": "emissions_by_zone" or "highest_emitters" or "lowest_emitters" or "zone_comparison",
    "sort_order": "ascending" or "descending",
    "limit": number (optional, default: show all),
    "explanation": "brief explanation of what was understood from the query"
}}

Example responses:
{{"analysis_type": "highest_emitters", "sort_order": "descending", "limit": 3, "explanation": "Showing top 3 zones with highest CO2 emissions"}}
{{"analysis_type": "zone_comparison", "sort_order": "ascending", "explanation": "Comparing CO2 emissions across all zones"}}"""

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": analysis_prompt}],
            temperature=0.1
        )
        
        result = json.loads(response.choices[0].message.content.strip())
        print(f"CO2 query analysis: {result['explanation']}")
        
        # Process the data based on analysis type
        if result['analysis_type'] == 'highest_emitters':
            filtered_data = co2_data.nlargest(
                result.get('limit', len(co2_data)),
                'CO2_Emissions_tonnes'
            )
        elif result['analysis_type'] == 'lowest_emitters':
            filtered_data = co2_data.nsmallest(
                result.get('limit', len(co2_data)),
                'CO2_Emissions_tonnes'
            )
        else:
            filtered_data = co2_data.copy()
        
        # Add visualization information
        viz_data = {
            "type": "bar",
            "title": f"CO2 Emissions by Zone",
            "x_column": "Zone",
            "y_column": "CO2_Emissions_tonnes",
            "aggregation": "sum"
        }
        
        return filtered_data, viz_data
        
    except Exception as e:
        print(f"Error processing CO2 data: {e}")
        return pd.DataFrame(), None

def process_with_agent(agent_type, query, logs):
    """
    Process the query using the appropriate specialized agent.
    Returns the filtered data and any additional logs.
    """
    print(f"Processing with {agent_type} agent...")
    logs.append(f"Processing with {agent_type} agent...")
    
    if agent_type == "co2":
        # Use the CO2 agent
        filtered_data, viz_data = process_co2_data(query)
        if filtered_data.empty:
            raise ValueError("No CO2 emissions data available.")
        return filtered_data, "co2"
    elif agent_type == "bicing":
        # Use the Bicing agent
        filtered_data = fetch_bicing_stations(query)
        if filtered_data.empty:
            raise ValueError("No Bicing stations found in the specified area.")
        return filtered_data, "bicing"
    elif agent_type == "restaurants":
        # Use the restaurant agent
        filtered_data = fetch_restaurant_data(query)
        if filtered_data.empty:
            raise ValueError("No restaurants found matching your criteria.")
        return filtered_data, "restaurants"
    elif agent_type == "amenity":
        # Use the amenity agent
        primary_type, location, bbox, radius_km, center_coords, alternatives, brand_name, mobility_analysis = extract_location_and_amenity(query)
        
        # Default to Barcelona bounding box if location is "all" and no radius is provided
        if location == "all" and radius_km is None:
            bbox = [2.0525, 41.3200, 2.2275, 41.4900]  # Barcelona bounding box
            print("Using default Barcelona bounding box for search.")
        
        filtered_data = fetch_amenity_data(primary_type, bbox, radius_km, alternatives, brand_name, mobility_analysis)
        if filtered_data.empty:
            raise ValueError("No matching amenities found in Barcelona.")
        return filtered_data, "amenity"
    else:
        # Use the appropriate dataset
        if agent_type == "houses":
            dataset = houses_data
        elif agent_type == "supermarkets":
            dataset = supermarkets_data
        else:
            raise ValueError(f"Invalid agent type: {agent_type}")

        # Generate and execute filtering code
        prompt_file = os.path.join(WORKSPACE_DIR, "prompts", f"{agent_type}.note")
        with open(prompt_file, 'r') as f:
            pre_prompt = f.read()
        pre_prompt += f"\n\nUser query: \"{query}\""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": pre_prompt}],
            temperature=0
        )
        code = response.choices[0].message.content.strip()
        
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        
        # Execute the filtering code
        local_vars = {"data": dataset.copy(), "pd": pd}
        exec(code, {}, local_vars)
        filtered_data = local_vars.get("filtered_data")

        if filtered_data is None or filtered_data.empty:
            raise ValueError(f"No matching {agent_type} found for your query.")

        return filtered_data, agent_type

def suggest_alternative_query(original_query, error_type):
    """
    Suggests an alternative query based on the original query and error type.
    """
    try:
        suggestion_prompt = f"""As an error recovery expert, analyze this failed query and suggest a working alternative:

Original query: "{original_query}"
Error type: {error_type}

Rules for suggesting alternatives:
1. If location not found: Use a more common nearby location
2. If amenity type not found: Suggest a similar category
3. If no results in radius: Suggest increasing the search radius
4. If syntax error: Fix the query structure

Return ONLY a JSON object in this format:
{{
    "alternative_query": "suggested working query",
    "explanation": "brief explanation of what was changed and why",
    "error_summary": "user-friendly error explanation"
}}"""

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": suggestion_prompt}],
            temperature=0.7
        )
        
        result = json.loads(response.choices[0].message.content.strip())
        return result
    except Exception as e:
        print(f"Error in suggestion generation: {e}")
        return {
            "alternative_query": "show amenities in barcelona center",
            "explanation": "Falling back to a general search in Barcelona center",
            "error_summary": "The original query couldn't be processed. Showing general results instead."
        }

def handle_query_error(original_query, error, logs):
    """
    Handles query errors by suggesting and executing alternative queries.
    """
    try:
        # Get suggestion for alternative query
        suggestion = suggest_alternative_query(original_query, str(error))
        
        print(f"Original query failed: {str(error)}")
        print(f"Trying alternative query: {suggestion['alternative_query']}")
        
        # Try the alternative query
        agent_type, refined_query = analyze_prompt(suggestion['alternative_query'])
        filtered_data, category = process_with_agent(agent_type, refined_query, logs)
        
        # Add error information to the summary
        error_summary = f"""Note: Your original query couldn't be processed exactly as requested.

Error Details: {suggestion['error_summary']}

What we did instead: {suggestion['explanation']}

Showing results for: "{suggestion['alternative_query']}"
"""
        
        return filtered_data, category, error_summary
    except Exception as e:
        print(f"Error in error handling: {e}")
        # Ultimate fallback - show amenities in city center
        fallback_query = "show amenities in barcelona center"
        agent_type, refined_query = analyze_prompt(fallback_query)
        filtered_data, category = process_with_agent(agent_type, refined_query, logs)
        
        error_summary = f"""Note: We couldn't process your original query due to technical limitations.
We're showing general amenities in Barcelona's center instead.
Try refining your search with more specific terms or locations."""
        
        return filtered_data, category, error_summary

@app.route("/process-prompt", methods=["POST"])
def process_prompt():
    logs = []
    try:
        print("\n=== Starting process-prompt ===")
        prompt = request.json.get("prompt")
        print(f"Received prompt: {prompt}")
        
        if not prompt:
            print("Error: No prompt provided")
            return jsonify({"error": "No prompt provided"}), 400

        logs.append(f"Processing prompt: {prompt}")

        # Clean up output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_map_path = os.path.join(OUTPUT_DIR, "filtered_map.html")
        if os.path.exists(output_map_path):
            try:
                os.remove(output_map_path)
                print(f"Removed existing map file: {output_map_path}")
            except Exception as e:
                print(f"Warning: Could not remove existing map file: {e}")

        try:
            # Try original query first
            agent_type, refined_query = analyze_prompt(prompt)
            print(f"Main agent selected: {agent_type} agent")
            logs.append(f"Main agent selected: {agent_type} agent")
            
            filtered_data, category = process_with_agent(agent_type, refined_query, logs)
            error_context = None
        except Exception as e:
            # If original query fails, use error handling agent
            print(f"Original query failed, trying error handling agent...")
            filtered_data, category, error_context = handle_query_error(prompt, e, logs)

        # Generate AI summary and analytics
        print("Generating analytics...")
        logs.append("Generating analytics...")
        analytics_prompt = f"""As an analytics expert, analyze this data and provide insights:
        Query: {prompt}
        Category: {category}
        Number of results: {len(filtered_data)}
        Data sample: {filtered_data.head().to_string() if not filtered_data.empty else 'No data'}
        {f'Error Context: {error_context}' if error_context else ''}

        {
            'For amenity data, focus on:' if category == 'amenity' else 'Focus on:'
        }
        {'''
        1. Distribution of locations across Barcelona
        2. Notable clusters or patterns
        3. Areas with high or low concentration
        ''' if category == 'amenity' else '''
        1. Price distributions and ranges
        2. Location patterns
        3. Notable features or amenities
        '''}

        Format your response as JSON:
        {{
            "summary": "your summary here",
            "visualization": {{
                "possible": true/false,
                "type": "bar/pie",
                "title": "chart title",
                "x_column": "column name for x-axis or categories",
                "y_column": "column name for y-axis or values (optional)",
                "aggregation": "count/mean/sum"
            }}
        }}"""

        analytics_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": analytics_prompt}],
            temperature=0.7
        )
        
        try:
            print("Parsing analytics response...")
            analytics_data = json.loads(analytics_response.choices[0].message.content.strip())
            summary = analytics_data["summary"]
            print(f"Generated summary: {summary}")
            
            # Generate visualization if possible
            viz_data = analytics_data.get("visualization", {})
            if viz_data.get("possible", False) and not filtered_data.empty:
                print("Generating visualization...")
                try:
                    plt.figure(figsize=(10, 6))
                    plt.style.use('bmh')  # Using a built-in style that's modern and clean
                    
                    x_col = viz_data["x_column"]
                    if viz_data["type"] == "bar":
                        if viz_data["aggregation"] == "count":
                            data = filtered_data[x_col].value_counts()
                        else:
                            y_col = viz_data["y_column"]
                            data = filtered_data.groupby(x_col)[y_col].agg(viz_data["aggregation"])
                        
                        bars = plt.bar(data.index, data.values, color='#4a6cf7', alpha=0.8)
                        plt.xticks(rotation=45, ha='right')
                        
                        # Add value labels on top of bars
                        for bar in bars:
                            height = bar.get_height()
                            plt.text(bar.get_x() + bar.get_width()/2., height,
                                   f'{int(height):,}',
                                   ha='center', va='bottom')
                    elif viz_data["type"] == "pie":
                        data = filtered_data[x_col].value_counts()
                        plt.pie(data.values, labels=data.index, autopct='%1.1f%%', 
                               colors=['#6e8efb', '#4a6cf7', '#415ca7', '#2b3f75', '#18263f'])
                    
                    plt.title(viz_data["title"], pad=20, fontsize=12, fontweight='bold')
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    
                    # Save the figure and get base64 data
                    img_path = os.path.join(OUTPUT_DIR, "visualization.png")
                    plt.savefig(img_path, format='png', dpi=300, bbox_inches='tight')
                    plt.close('all')
                    
                    # Convert image to base64
                    with open(img_path, 'rb') as img_file:
                        img_data = base64.b64encode(img_file.read()).decode()
                    
                    # Store visualization data separately
                    visualization_data = {
                        "type": "image",
                        "data": img_data,
                        "title": viz_data["title"]
                    }
                    
                    print("Visualization generated successfully")
                except Exception as e:
                    print(f"Error generating visualization: {e}")
                    visualization_data = None
            else:
                print("No visualization possible for this data")
                if viz_data.get("possible") is False:
                    summary += "\n\nNote: The data structure does not support meaningful visualization."

            # Save filtered data and create map
            print("Saving filtered data...")
            filtered_data.to_csv(os.path.join(OUTPUT_DIR, "current_data.csv"), index=False)
            logs.append("Saved filtered data")

            # Create map with explicit dimensions
            print("Generating map...")
            logs.append("Generating map...")
            try:
                filtered_map = KeplerGl(
                    height=969,
                    width=1920,
                    data={"filtered_data": filtered_data},
                    config=generate_map_config()
                )
                print("Map object created successfully")

                # Save the map
                print(f"Saving map to: {output_map_path}")
                filtered_map.save_to_html(file_name=output_map_path)
                
                # Verify the file was created
                if not os.path.exists(output_map_path):
                    raise Exception("Map file was not created")
                    
                # Read and modify the map HTML
                with open(output_map_path, 'r', encoding='utf-8') as f:
                    map_html = f.read()

                # Create the modified HTML with both map and analytics
                print("Creating final HTML...")
                modified_html = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <meta charset="utf-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1">
                    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                    <style>
                        @font-face {{
                            font-family: 'Bahnschrift';
                            src: local('Bahnschrift');
                        }}
                        
                        body, html {{
                            margin: 0;
                            padding: 0;
                            width: 100vw;
                            height: 100vh;
                            overflow: hidden;
                            font-family: 'Bahnschrift', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen-Sans, Ubuntu, Cantarell, 'Helvetica Neue', sans-serif;
                        }}
                        #kepler-gl {{
                            width: 100% !important;
                            height: 100% !important;
                        }}
                        .kepler-gl {{
                            width: 100% !important;
                            height: 100% !important;
                        }}
                        /* Apply Bahnschrift to all text elements */
                        .kepler-gl * {{
                            font-family: 'Bahnschrift', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
                        }}
                        /* Ensure tooltips and labels use Bahnschrift */
                        .mapboxgl-popup-content,
                        .mapboxgl-ctrl-attrib,
                        .mapboxgl-ctrl-scale {{
                            font-family: 'Bahnschrift', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
                        }}
                    </style>
                </head>
                <body>
                    {map_html}
                    <script>
                        document.addEventListener('DOMContentLoaded', function() {{
                            setTimeout(function() {{
                                const keplerGl = document.querySelector('#kepler-gl');
                                if (keplerGl) {{
                                    keplerGl.style.width = '100%';
                                    keplerGl.style.height = '100vh';
                                }}
                                window.dispatchEvent(new Event('resize'));
                            }}, 100);
                        }});
                    </script>
                </body>
                </html>
                """

                # Save the modified HTML
                print("Writing modified HTML...")
                with open(output_map_path, 'w', encoding='utf-8') as f:
                    f.write(modified_html)
                print("Modified HTML written successfully")
                
                # Verify the file exists and is accessible
                if not os.path.exists(output_map_path):
                    raise Exception("Modified map file was not saved")
                    
                file_size = os.path.getsize(output_map_path)
                print(f"Map file size: {file_size} bytes")
                
                if file_size == 0:
                    raise Exception("Map file is empty")
                    
                print("Map file successfully created and verified")
                
            except Exception as e:
                print(f"Error in map generation: {e}")
                raise

            print("=== Process completed successfully ===")
            return jsonify({
                "success": True,
                "filteredMap": "filtered_map.html",
                "count": len(filtered_data),
                "logs": logs,
                "summary": summary,
                "visualization": visualization_data
            })

        except json.JSONDecodeError as e:
            print(f"Error parsing analytics JSON: {e}")
            print(f"Raw analytics response: {analytics_response.choices[0].message.content.strip()}")
            summary = analytics_response.choices[0].message.content.strip()

        print("=== Process completed successfully ===")
        return jsonify({
            "success": True,
            "filteredMap": "filtered_map.html",
            "count": len(filtered_data),
            "logs": logs,
            "summary": summary,
            "visualization": visualization_data
        })

    except Exception as e:
        print(f"=== Error in process-prompt: {str(e)} ===")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Traceback:\n{traceback.format_exc()}")
        logs.append(f"Error: {str(e)}")
        return jsonify({
            "error": str(e),
            "logs": logs,
            "summary": "An error occurred while processing your request."
        }), 500

def load_bicing_data():
    """
    Load and process Bicing bike sharing data.
    Returns a DataFrame with bike usage statistics.
    """
    try:
        # Read the CSV file
        bicing_df = pd.read_csv('data/2019_01_Gener_BICING_US.csv')
        
        # Convert datetime string to datetime object
        bicing_df['dateTime'] = pd.to_datetime(bicing_df['dateTime'])
        
        # Add additional time-based columns for analysis
        bicing_df['hour'] = bicing_df['dateTime'].dt.hour
        bicing_df['day'] = bicing_df['dateTime'].dt.day
        bicing_df['weekday'] = bicing_df['dateTime'].dt.weekday
        
        # Calculate usage percentages
        bicing_df['electrical_percentage'] = (bicing_df['electricalBikesInUsage'] / bicing_df['bikesInUsage'] * 100).fillna(0)
        bicing_df['mechanical_percentage'] = (bicing_df['mechanicalBikesInUsage'] / bicing_df['bikesInUsage'] * 100).fillna(0)
        
        return bicing_df
    except Exception as e:
        print(f"Error loading Bicing data: {e}")
        return pd.DataFrame()

def get_bicing_stats(location=None, radius_km=1.0):
    """
    Get Bicing statistics for a specific location or the entire city.
    Returns usage patterns and bike availability information.
    """
    try:
        bicing_df = load_bicing_data()
        if bicing_df.empty:
            return None
            
        # Calculate general statistics
        stats = {
            'avg_bikes_in_use': bicing_df['bikesInUsage'].mean(),
            'max_bikes_in_use': bicing_df['bikesInUsage'].max(),
            'avg_electrical_bikes': bicing_df['electricalBikesInUsage'].mean(),
            'avg_mechanical_bikes': bicing_df['mechanicalBikesInUsage'].mean(),
            'peak_hours': bicing_df.groupby('hour')['bikesInUsage'].mean().nlargest(3).index.tolist(),
            'electrical_percentage': bicing_df['electrical_percentage'].mean(),
            'mechanical_percentage': bicing_df['mechanical_percentage'].mean()
        }
        
        return stats
    except Exception as e:
        print(f"Error getting Bicing stats: {e}")
        return None

def analyze_mobility_patterns(query, location=None):
    """
    Analyze mobility patterns combining Bicing data with other amenities.
    """
    try:
        bicing_stats = get_bicing_stats(location)
        if not bicing_stats:
            return "Could not analyze mobility patterns due to missing data."
            
        # Format peak hours for readability
        peak_hours = [f"{hour}:00" for hour in bicing_stats['peak_hours']]
        
        analysis = f"""
        Mobility Analysis:
        - Average bikes in use: {int(bicing_stats['avg_bikes_in_use'])}
        - Peak usage hours: {', '.join(peak_hours)}
        - Bike types: {int(bicing_stats['mechanical_percentage'])}% mechanical, {int(bicing_stats['electrical_percentage'])}% electrical
        """
        
        return analysis
    except Exception as e:
        print(f"Error analyzing mobility patterns: {e}")
        return "Could not analyze mobility patterns."

@app.route("/api/search", methods=["POST"])
def api_search():
    """
    API endpoint for searching amenities in Barcelona.
    Expected JSON body:
    {
        "query": "search query",
        "lat": optional float (default: Barcelona center),
        "lon": optional float (default: Barcelona center),
        "radius_km": optional float (default: 1.0)
    }
    """
    try:
        data = request.json
        if not data or 'query' not in data:
            return jsonify({
                "error": "Missing required parameter: query"
            }), 400

        # Get parameters with defaults
        query = data['query']
        lat = data.get('lat', 41.3851)  # Barcelona center default
        lon = data.get('lon', 2.1734)   # Barcelona center default
        radius_km = data.get('radius_km', 1.0)

        # Validate coordinates are within Barcelona
        if not (41.3200 <= lat <= 41.4900 and 2.0525 <= lon <= 2.2275):
            return jsonify({
                "error": "Coordinates must be within Barcelona's boundaries"
            }), 400

        # Use smart agent to analyze query
        agent_type, refined_query = analyze_prompt(query)
        
        # Process with selected agent
        filtered_data = fetch_amenity_data(refined_query, lat, lon, radius_km * 1000)
        
        if filtered_data.empty:
            return jsonify({
                "success": True,
                "count": 0,
                "results": [],
                "message": "No results found"
            })

        # Convert to list of dictionaries for JSON response
        results = filtered_data.to_dict('records')
        
        # Add distance information
        for result in results:
            result['distance_km'] = haversine_distance(
                lat, lon,
                result['latitude'],
                result['longitude']
            )

        # Sort by distance
        results = sorted(results, key=lambda x: x['distance_km'])

        return jsonify({
            "success": True,
            "count": len(results),
            "results": results,
            "query_info": {
                "original_query": query,
                "refined_query": refined_query,
                "agent_type": agent_type,
                "search_center": {"lat": lat, "lon": lon},
                "radius_km": radius_km
            }
        })

    except Exception as e:
        print(f"API Error: {str(e)}")
        return jsonify({
            "error": str(e),
            "message": "An error occurred while processing your request"
        }), 500

@app.route("/api/amenities", methods=["GET"])
def api_amenities():
    """
    Get a list of supported amenity types.
    """
    try:
        amenity_categories = {
            "food_drink": [
                "restaurant", "cafe", "bar", "pub", "fast_food",
                "food_court", "ice_cream", "bakery"
            ],
            "entertainment": [
                "cinema", "theatre", "nightclub", "casino",
                "arts_centre", "music_venue"
            ],
            "education": [
                "school", "university", "college", "library",
                "language_school", "music_school"
            ],
            "health": [
                "hospital", "clinic", "doctors", "dentist",
                "pharmacy", "veterinary"
            ],
            "transport": [
                "parking", "bicycle_parking", "bus_station",
                "car_rental", "taxi", "fuel"
            ],
            "shopping": [
                "marketplace", "mall", "supermarket",
                "convenience", "clothes", "books"
            ],
            "sports": [
                "sports_centre", "swimming_pool", "gym",
                "stadium", "pitch", "tennis"
            ],
            "tourism": [
                "hotel", "hostel", "guest_house", "museum",
                "gallery", "information"
            ],
            "services": [
                "bank", "atm", "post_office", "police",
                "fire_station", "embassy"
            ],
            "leisure": [
                "park", "garden", "playground", "bbq",
                "picnic_table", "dog_park"
            ]
        }

        return jsonify({
            "success": True,
            "categories": amenity_categories
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "message": "An error occurred while fetching amenity types"
        }), 500

@app.route("/api/location/search", methods=["GET"])
def api_location_search():
    """
    Search for a location in Barcelona.
    Required parameter: q (query string)
    """
    try:
        query = request.args.get('q')
        if not query:
            return jsonify({
                "error": "Missing required parameter: q"
            }), 400

        # Try local database first
        location_result = find_location_in_database(query)
        if location_result:
            lat, lon, name = location_result
            return jsonify({
                "success": True,
                "results": [{
                    "name": name,
                    "lat": lat,
                    "lon": lon,
                    "source": "local_database"
                }]
            })

        # Try online search
        online_result = search_location_in_barcelona(query)
        if online_result:
            lat, lon, name = online_result
            return jsonify({
                "success": True,
                "results": [{
                    "name": name,
                    "lat": lat,
                    "lon": lon,
                    "source": "online_search"
                }]
            })

        return jsonify({
            "success": True,
            "count": 0,
            "results": [],
            "message": "No locations found"
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "message": "An error occurred while searching for locations"
        }), 500

# Add CORS headers
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

if __name__ == "__main__":
    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create an initial empty map
    try:
        initial_map = KeplerGl(height=969, width=1920)
        initial_map_path = os.path.join(OUTPUT_DIR, "filtered_map.html")
        initial_map.save_to_html(file_name=initial_map_path)
        print(f"Initial map created at: {initial_map_path}")
    except Exception as e:
        print(f"Warning: Could not create initial map: {e}")
    
    app.run(debug=True, port=5000)