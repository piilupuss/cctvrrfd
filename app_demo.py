#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrated Flood Monitoring System for Bandung City
DEMO Version (Local Video Streams)
- Multi-threaded LOCAL VIDEO "Stream" Capture
- Simulated Flood Detection
- Road risk analysis based on elevation.
- Real-time 2D Visualization using Streamlit and Plotly
"""

import os
import cv2
import threading
import queue
import numpy as np
import rasterio
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import time
import csv
from datetime import datetime
from shapely.geometry import Point
from flood_detection_3lvl import classify_image
import pandas as pd # Added: For easier DataFrame creation for Streamlit table

# --- Configuration ---
# DEM_PATH = "/Users/nagailab/Documents/Dev/water_level_classifier/data/dem_bandung.tif"  # Path to the Digital Elevation Model
# ROADS_PATH = "/Users/nagailab/Documents/Dev/water_level_classifier/data/roads_bandung.geojson"  # Path to the roads GeoJSON
DEM_PATH = "data/DEMNAS_1209-31_v1.0.tif"  
ROADS_PATH = "data/roads_bandung.geojson"  
OUTPUT_FOLDER = "data/captured_images" # Folder to save captured images
RESULTS_FILE = "data/flood_detection_results_demo.csv" # CSV to log detection results
CCTV_CSV = "data/cctv_locations_demo_video.csv" # CSV with CCTV details (ID, Name, Lat, Lon, URL)
CAPTURE_INTERVAL = 60  # Capture interval in seconds per CCTV
FLOOD_RISK_RADIUS_M = 500 
# IDW_POWER = 2  # Power parameter for IDW interpolation

# --- Global Data Structures ---
# Queue to pass detected flood points from capture threads to the main Streamlit thread
# flood_detection_queue = queue.Queue()
cctv_status_queue = queue.Queue()

# List to store all detected flood points for interpolation and visualization
# This list will be accessed by the Streamlit thread
# global_flood_points = [] # Stores (lon, lat) tuples

# List to store the latest status of all CCTVs for visualization
# Stores dictionaries: {'cctv_id', 'cctv_name', 'lat', 'lon', 'flood_status', 'timestamp'}
global_cctv_statuses = {} # Use a dictionary for easy updates by cctv_id

# --- Helper Functions ---
def cleanup_images(output_folder, max_images=6):
    """Keep only the newest images in the folder"""
    images = sorted(
        [os.path.join(output_folder, f) for f in os.listdir(output_folder) if f.endswith('.jpg')],
        key=os.path.getctime
    )
    if len(images) > max_images:
        for old_img in images[:-max_images]:
            try:
                os.remove(old_img)
                # print(f"Deleted old image: {old_img}") # Uncomment for verbose logging
            except Exception as e:
                print(f"Error deleting {old_img}: {e}")

def initialize_results_file(results_file):
    """Create results file with headers if it doesn't exist"""
    if not os.path.exists(results_file):
        with open(results_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp',
                'cctv_id',
                'cctv_name',
                'latitude',
                'longitude',
                'image_path',
                'flood_status'
            ])

def load_cctv_data(csv_path):
    """Load CCTV data from CSV file"""
    cctv_list = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cctv_list.append({
                'cctv_id': row['cctv_id'],
                'lat': float(row['lat']),
                'lon': float(row['lon']),
                'cctv_name': row['cctv_name'],
                'url': row['url']
            })
    return cctv_list

# --- Threaded CCTV Capture and Detection Function ---
def capture_images_from_stream(stream_url, output_folder, cctv_id, cctv_name, lat, lon, results_file, interval):
    """
    Capture images from a video stream, perform flood detection,
    and send detected flood points to a global queue.
    """
    cctv_output_folder = os.path.join(output_folder, f"cctv_{cctv_id}")
    os.makedirs(cctv_output_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print(f"Error: Unable to open stream {stream_url} for {cctv_name}")
        return
    print(f"Started capture from {cctv_name} (ID: {cctv_id})")
    last_capture_time = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            # If the source is a local file (ends with .mp4, .avi, etc.), loop it.
            # Otherwise, for a URL, try to reconnect.
            if stream_url.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                print(f"Video '{cctv_name}' ended. Looping...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else: # Original behavior for network streams
                print(f"Stream ended or unable to fetch frame from {cctv_name}. Retrying in 5s...")
                time.sleep(5)
                cap = cv2.VideoCapture(stream_url) # Attempt to re-open stream
                continue
        
        current_time = time.time()
        if (current_time - last_capture_time) >= interval:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{cctv_name}_{timestamp}.jpg"
            filepath = os.path.join(cctv_output_folder, filename)

            cv2.imwrite(filepath, frame)
            # print(f"Captured image from {cctv_name}: {filepath}") # Uncomment for verbose logging
            last_capture_time = current_time
            
            flood_status = classify_image(filepath)
            print(f"Flood detection result for {cctv_name}: {flood_status}")
            
            with open(results_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp,
                    cctv_id,
                    cctv_name,
                    lat,
                    lon,
                    filepath,
                    flood_status
                ])

            # Put the detected flood point (lon, lat) into the queue
            # Now putting full CCTV status
            cctv_status_queue.put({
                'cctv_id': cctv_id,
                'cctv_name': cctv_name,
                'lat': lat,
                'lon': lon,
                'flood_status': flood_status,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            print(f"Status for {cctv_name} ({cctv_id}) updated: {flood_status}")
            
            # if flood_status == 'Flood':
            #     # Put the detected flood point (lon, lat) into the queue
            #     flood_detection_queue.put((lon, lat))
            #     print(f"Flood detected at: ({lon}, {lat}) - Sent to queue.")
            
            cleanup_images(cctv_output_folder)
        
        time.sleep(0.1) # Small sleep to reduce CPU usage
    cap.release()

# --- GIS Data Loading and Elevation Helpers ---
@st.cache_data 
def load_gis_data():
    """Load Roads data, DEM raster, and metadata once."""
    roads = None
    dem_data = None
    dem_transform = None
    
    try:
        # Load DEM data
        if os.path.exists(DEM_PATH):
            with rasterio.open(DEM_PATH) as src:
                dem_data = src.read(1) 
                dem_transform = src.transform 
        else:
            st.error(f"DEM file not found at '{DEM_PATH}'. Please check path.")
        
        # Load roads GeoJSON
        if os.path.exists(ROADS_PATH):
            roads = gpd.read_file(ROADS_PATH)
            roads = roads.to_crs("EPSG:4326") # Ensure roads are in WGS84 for Mapbox 
        else:
            st.error(f"Roads file not found at '{ROADS_PATH}'. Please check path.")

        return roads, dem_data, dem_transform
        
    except Exception as e:
        st.error(f"GIS Data loading failed: {str(e)}. Check file integrity and paths.")
        return None, None, None

def get_elevation_at_point(dem_data, dem_transform, lon, lat):
    """Retrieves elevation from DEM at a given lon/lat."""
    if dem_data is None or dem_transform is None:
        return None
    try:
        # Convert world coords (lon, lat) to pixel indices (row, col)
        row, col = rasterio.transform.rowcol(dem_transform, lon, lat)
        row, col = int(row), int(col)
        
        # Check bounds
        if 0 <= row < dem_data.shape[0] and 0 <= col < dem_data.shape[1]:
            elevation = dem_data[row, col]
            # Handle potential NoData values
            return elevation if not np.isnan(elevation) else None
        return None
    except Exception:
        return None

# --- Road Highlighting Logic with Elevation Check (Similar to app_vis.py) ---
@st.cache_data(show_spinner=False)
def get_risk_roads_by_elevation(_roads_gdf, flood_point_df, dem_data, dem_transform, radius_meters=FLOOD_RISK_RADIUS_M):
    """
    Identifies roads within a buffer where the road elevation is lower than 
    the simulated flood point's elevation.
    """
    if _roads_gdf is None or dem_data is None or dem_transform is None:
        return gpd.GeoDataFrame()

    # 1. Project Flood Point for accurate buffering
    flood_geometries = [Point(lon, lat) for lon, lat in zip(flood_point_df['lon'], flood_point_df['lat'])]
    flood_gdf_4326 = gpd.GeoDataFrame(flood_point_df, geometry=flood_geometries, crs="EPSG:4326")
    
    try:
        projected_crs = "EPSG:32748" # UTM Zone 48S for Bandung area
        flood_gdf_projected = flood_gdf_4326.to_crs(projected_crs)
        roads_projected = _roads_gdf.to_crs(projected_crs)
    except Exception as e:
        st.warning(f"Projection setup error: {e}")
        return gpd.GeoDataFrame() 

    # 2. Calculate the 500m buffer and intersect with roads
    flood_buffer = flood_gdf_projected.geometry.buffer(radius_meters).unary_union
    buffered_roads_projected = roads_projected[roads_projected.intersects(flood_buffer)].copy()

    # 3. Check elevation of the road geometry vertices
    risky_roads_list = []
    
    # Use the original WGS84 roads (for elevation lookup and Plotly)
    roads_4326 = _roads_gdf 
    
    for idx, road in buffered_roads_projected.iterrows():
        # Get the geometry of the road segment in 4326 (WGS84)
        geom_4326 = roads_4326.loc[idx].geometry
        
        coords = []
        if geom_4326.geom_type == 'LineString':
            coords.extend(list(geom_4326.coords))
        elif geom_4326.geom_type == 'MultiLineString':
            for line in geom_4326.geoms:
                coords.extend(list(line.coords))

        is_risky = False
        # Risk condition: road elevation is lower than the flood point elevation
        for lon, lat in coords:
            road_elev = get_elevation_at_point(dem_data, dem_transform, lon, lat)
            
            if road_elev is not None and (road_elev < flood_point_df['flood_elevation']).any():
                is_risky = True
                break
        
        if is_risky:
            # Add the original WGS84 road segment to the risk list
            risky_roads_list.append(_roads_gdf.loc[idx].copy()) 
            
    if risky_roads_list:
        risky_gdf = gpd.GeoDataFrame(risky_roads_list, crs="EPSG:4326")
        return risky_gdf
    else:
        return gpd.GeoDataFrame()

# def interpolate_flood_idw(dem, transform, points, power=IDW_POWER):
    """Perform true IDW (Inverse Distance Weighting) interpolation."""
    if not points:
        return None

    # Convert world coordinates (lon, lat) to DEM pixel indices (x_pixel, y_pixel)
    # Note: rasterio transform is (x_origin, x_pixel_size, x_rotation, y_origin, y_rotation, y_pixel_size)
    # For typical north-up rasters, x_rotation=0, y_rotation=0, y_pixel_size is negative
    # x_pixel = (lon - transform.c) / transform.a
    # y_pixel = (lat - transform.f) / transform.e
    # Using transform.rowcol for more robust conversion
    
    # Convert (lon, lat) to (row, col)
    rows, cols = rasterio.transform.rowcol(transform, [p[0] for p in points], [p[1] for p in points])
    
    # Filter out points outside DEM bounds
    valid_indices = [(r, c) for r, c in zip(rows, cols) if 0 <= r < dem.shape[0] and 0 <= c < dem.shape[1]]
    
    if not valid_indices:
        return None # No valid points within DEM extent

    # Extract valid pixel coordinates and their elevations
    px = np.array([c for r, c in valid_indices])
    py = np.array([r for r, c in valid_indices])
    elevations = np.array([dem[r, c] for r, c in valid_indices])

    # Create output grid (interpolated flood surface)
    n_rows, n_cols = dem.shape
    
    # Create a grid of pixel coordinates for interpolation
    x_grid_pixels = np.arange(n_cols)
    y_grid_pixels = np.arange(n_rows)
    xx_pixels, yy_pixels = np.meshgrid(x_grid_pixels, y_grid_pixels)

    flood_grid = np.zeros_like(xx_pixels, dtype=float)

    # Iterate over each pixel in the DEM grid
    for i in range(n_rows): # row index
        for j in range(n_cols): # column index
            x0, y0 = j, i # Current pixel coordinates (col, row)

            # Calculate Euclidean distances from current pixel to all valid flood points
            distances = np.sqrt((px - x0)**2 + (py - y0)**2)

            # Handle cases where a grid point is exactly at a flood point location
            near_zero = distances < 1e-6 # Use a small epsilon to avoid floating point issues
            if np.any(near_zero):
                # If an exact match, assign the elevation of that flood point
                flood_grid[i, j] = elevations[near_zero.argmax()]
                continue

            # Calculate weights based on inverse distance
            weights = 1.0 / (distances**power)
            
            # If all weights are zero (e.g., all distances are infinite or very large and power makes weights tiny)
            if np.sum(weights) == 0:
                flood_grid[i, j] = np.nan # No influence from any point
            else:
                # Calculate the weighted average of elevations
                flood_grid[i, j] = np.sum(weights * elevations) / np.sum(weights)
    return flood_grid

# --- Streamlit Application ---
def main():
    st.set_page_config(layout="wide", page_title="DEMO Bandung Flood Monitoring")
    st.title("DEMO Bandung Flood Monitoring")
    # st.markdown("""
    # Real-time flood spread estimation using CCTV streams, true IDW interpolation, and DEM data.
    # """)
    st.markdown("""
    DEMO flood status from LOCAL VIDEOS as a CCTV Stream in Bandung City.
    """)

    # 1. Load GIS data
    roads, dem_data, dem_transform = load_gis_data()
    
    roads_available = roads is not None and not roads.empty
    elevation_available = dem_data is not None and dem_transform is not None

    if not roads_available or not elevation_available:
        st.warning("GIS data not available. Map cannot be generated.")
        return
    
    # Initialize results file for logging
    initialize_results_file(RESULTS_FILE)

    # Load CCTV data
    cctv_list = load_cctv_data(CCTV_CSV)
    if not cctv_list:
        st.error(f"No CCTV data found in '{CCTV_CSV}'. Please ensure the file exists and is correctly formatted.")
        st.stop()

    # Initialize global_cctv_statuses with initial data from CSV
    # This ensures all CCTVs appear on the map from the start
    for cctv in cctv_list:
        global_cctv_statuses[cctv['cctv_id']] = {
            'cctv_id': cctv['cctv_id'],
            'cctv_name': cctv['cctv_name'],
            'lat': cctv['lat'],
            'lon': cctv['lon'],
            'flood_status': 'Initializing...', # Initial status
            'flood_elevation': get_elevation_at_point(
                dem_data, dem_transform, cctv['lon'], cctv['lat']), # Initial elevation
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    # Start CCTV capture threads (only once)
    if 'cctv_threads_started' not in st.session_state:
        st.session_state.cctv_threads_started = True
        for cctv in cctv_list:
            thread = threading.Thread(
                target=capture_images_from_stream,
                args=(
                    cctv['url'],
                    OUTPUT_FOLDER,
                    cctv['cctv_id'],
                    cctv['cctv_name'],
                    cctv['lat'],
                    cctv['lon'],
                    RESULTS_FILE,
                    CAPTURE_INTERVAL
                ),
                daemon=True # Daemon threads exit when the main program exits
            )
            thread.start()
            print(f"Started capture thread for {cctv['cctv_name']}")

             # 3. Get Flood Point Elevation
            # flood_elevation = get_elevation_at_point(
            #     dem_data, dem_transform, cctv['lon'], cctv['lat']  
            # )
            # if flood_elevation is None:
            #     st.error(f"Could not determine elevation for the flood point. Check if the point is within the DEM boundary.")
            #     return
        st.success("CCTV monitoring threads started.")

    # Placeholder for the dynamic plot
    plot_placeholder = st.empty()
    
    # Placeholder for flood point table
    table_placeholder = st.empty()

    
    
    # New Main Streamlit update loop
    while True:
        # Process new CCTV statuses from the queue
        while not cctv_status_queue.empty():
            new_status = cctv_status_queue.get()
            cctv_id = new_status['cctv_id']
            if cctv_id in global_cctv_statuses:
                # Update existing dictionary to preserve 'flood_elevation'
                global_cctv_statuses[cctv_id].update(new_status)
        # Convert global_cctv_statuses to a DataFrame for Plotly and Streamlit table
        # Ensure it's a list of dictionaries for DataFrame creation
        cctv_status_df = pd.DataFrame(list(global_cctv_statuses.values()))
        # Create the Plotly figure
        fig = go.Figure()
        if not cctv_status_df.empty:
            # Add CCTV points to the map
            # Use color to represent flood status
            color_map = {
                'Flood': 'red',
                'Dry': 'green',
                'Wet': 'yellow',
                'Stream Error': 'gray',
                'Initializing...': 'blue'
            }

            # --- 3. Flood Risk Road Highlighting (with Elevation Check) ---
            risky_roads_gdf = gpd.GeoDataFrame()
            
            # Filter for CCTVs that currently have a 'Flood' status
            flood_cctvs_df = cctv_status_df[cctv_status_df['flood_status'] == 'Flood']
            
            if not flood_cctvs_df.empty and roads_available and elevation_available:
                # Run Road Risk Analysis only if there are flooded CCTVs
                risky_roads_gdf = get_risk_roads_by_elevation(
                    roads, flood_cctvs_df, dem_data, dem_transform
                )

                # --- A. Add Flood Risk Roads (Highlighted Red) ---
                if not risky_roads_gdf.empty:
                    fig.add_trace(go.Scattermapbox(
                        lon=[None], lat=[None], mode='lines', name='Risk Road',
                        line=dict(color='red', width=3), legendgroup='risk_roads', showlegend=True
                    ))
                    # Plot all risky road segments
                    for geom in risky_roads_gdf.geometry:
                        if geom.geom_type == 'LineString':
                            x, y = geom.xy
                            fig.add_trace(go.Scattermapbox(
                                lon=list(x), lat=list(y), mode='lines', 
                                line=dict(color='red', width=3), hoverinfo='text',
                                name='Risk Road', showlegend=False, legendgroup='risk_roads'
                            ))
                        elif geom.geom_type == 'MultiLineString':
                            for line in geom.geoms:
                                x, y = line.xy
                                fig.add_trace(go.Scattermapbox(
                                    lon=list(x), lat=list(y), mode='lines', 
                                    line=dict(color='red', width=3), hoverinfo='none',
                                    name='Risk Road', showlegend=False, legendgroup='risk_roads'
                                ))


            # Ensure 'flood_status' column exists and map colors
            cctv_status_df['color'] = cctv_status_df['flood_status'].map(color_map).fillna('purple') # Default for unknown status
            fig.add_trace(go.Scattermapbox(
                lat=cctv_status_df['lat'],
                lon=cctv_status_df['lon'],
                mode='markers',
                marker=go.scattermapbox.Marker(
                    size=10,
                    color=cctv_status_df['color'],
                    opacity=0.8
                ),
                text=cctv_status_df.apply(lambda row: f"<b>{row['cctv_name']}</b><br>Status: {row['flood_status']}<br>Lat: {row['lat']:.4f}<br>Lon: {row['lon']:.4f}<br>Last Update: {row['timestamp']}", axis=1),
                hoverinfo='text',
                name='CCTV Locations'
            ))
            # Add a legend for flood status colors
            for status, color in color_map.items():
                fig.add_trace(go.Scattermapbox(
                    lat=[None], lon=[None], # Dummy points for legend
                    mode='markers',
                    marker=go.scattermapbox.Marker(size=10, color=color),
                    name=status,
                    showlegend=True
                ))

        # Update layout for better map-like appearance
        fig.update_layout(
            mapbox_style="open-street-map", # Use OpenStreetMap as base map
            mapbox_zoom=11.5, # Adjust zoom level as needed
            mapbox_center={"lat": cctv_status_df['lat'].mean() if not cctv_status_df.empty else -6.9175,
                           "lon": cctv_status_df['lon'].mean() if not cctv_status_df.empty else 107.6191}, # Center map on average CCTV location or Bandung
            margin={"r":0,"t":40,"l":0,"b":0},
            showlegend=True,
            legend=dict(x=1.02, y=1, xanchor='left', yanchor='top'),
        )
        # Display the plot in Streamlit
        with plot_placeholder.container():
            st.plotly_chart(fig, use_container_width=True, key=f"cctv_status_map_{int(time.time())}")
        # Display latest flood points in a table
        with table_placeholder.container():
            st.subheader("CCTV Status Overview")
            if not cctv_status_df.empty:
                # Select and reorder columns for display
                display_df = cctv_status_df[['cctv_name', 'lat', 'lon', 'flood_status', 'timestamp']]
                display_df.columns = ['CCTV Name', 'Latitude', 'Longitude', 'Flood Status', 'Last Updated']
                st.dataframe(display_df, hide_index=True)
            else:
                st.info("Waiting for CCTV data...")
        # Refresh Streamlit every few seconds to update the plot
        time.sleep(CAPTURE_INTERVAL) # Update every 10 seconds


if __name__ == "__main__":
    # Create dummy CCTV locations CSV if it doesn't exist for testing
    if not os.path.exists(CCTV_CSV):
        print(f"Creating dummy '{CCTV_CSV}' for demonstration.")
        with open(CCTV_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['cctv_id', 'lat', 'lon', 'cctv_name', 'url'])
            writer.writerow(['cctv_001', '-6.9176', '107.6191', 'Alun-alun Bandung', 'http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4']) # Example public video
            writer.writerow(['cctv_002', '-6.9123', '107.6254', 'Cikapundung River', 'http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ElephantsDream.mp4']) # Example public video
            writer.writerow(['cctv_003', '-6.9050', '107.6090', 'Gedung Sate Area', 'http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4']) # Example public video
    
    # Ensure 'captured_images' directory exists
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Import pandas here, as it's only needed for the dataframe display in main()
    import pandas as pd 
    
    main()
