
# Road Risk Flood Detection (CCTV RRFD) 🌊🤖

This project implements an **AI-Powered Integrated Flood Risk Monitoring System** for urban areas, specifically targeting Bandung City. It integrates real-time CCTV stream analysis with geospatial data (DEM and road networks) to detect flooding and dynamically highlight high-risk road segments based on elevation.

## Key Features

  * **Multi-threaded CCTV Monitoring:** Captures images from multiple video streams concurrently.
  * **AI Flood Classification:** Uses a simulated classification model (`classify_image`) to determine flood status (`Flood`, `Wet`, `Dry`).
  * **Geospatial Risk Analysis:** Compares flood point elevation (derived from DEM) with nearby road segment elevations.
  * **Dynamic Risk Highlighting:** Visualizes CCTV status and highlights **"Risk Roads"** on an interactive map using Plotly and Streamlit.

## Setup and Installation

### 1\. Prerequisites

You need **Python 3.12+** and **Git** installed.

### 2\. Clone the Repository

```bash
git clone https://github.com/piilupuss/cctvrrfd.git
cd cctvrrfd
```

### 3\. Install Dependencies

Create a `requirements.txt` file (if you haven't already) containing the necessary libraries, and install them:

```bash
# Recommended contents for requirements.txt:
# opencv-python
# tensorflow
# keras
# geopandas
# plotly
# streamlit
# numpy
# pandas
# rasterio

pip install -r requirements.txt
```

### 4\. Download Geospatial Data (Crucial Step\!)

Due to GitHub's file size limitations, the road network GeoJSON file is hosted externally.

1.  **Download the file:** Download `roads_bandung.geojson` from the following Google Drive link:
    🔗 **[Download here](https://drive.google.com/file/d/1NDOvwODVo8GYGTWQ80r18WZiaGVIBBud/view?usp=share_link)**
2.  **Place the file:** Ensure you place the downloaded `roads_bandung.geojson` file inside the existing `data/` directory in your project folder.

**Note:** Ensure your `data/` folder also contains the DEM file (`DEMNAS_1209-31_v1.0.tif`) and the CCTV location data (`cctv_locations.csv`).

## How to Run the Application

The project offers two main scripts to run the application: the Live Stream Version and the Demo Version.

### 1. Live Stream Version (app.py)

This is the primary application designed to connect to live CCTV stream URLs specified in cctv_locations.csv and capture frames every 20 seconds (CAPTURE_INTERVAL).

```bash
streamlit run app.py
```

### 2. Demo Version (app_demo.py) 🎬 (For showcasing flood event)

The `app_demo.py` script is provided to offer a reliable demonstration of the entire system, including a simulated flood occurrence and the resulting road risk analysis, without relying on external, often unstable, live streams.

Dummy CCTV Feeds: This script is configured to use local video files as dummy CCTV feeds for flood. These files is configured in `data/cctv_locations_demo_video.csv` to point to a local path (`data/video/`).

Capture Interval: For demonstration purposes, this demo captures a new frame from the video feeds and refreshes the Streamlit page every 1 minute (60 seconds).

Feature Showcase: This is the recommended script for users to run to quickly visualize the core functionality: AI detection transitioning a point to 'Flood' and the resulting dynamic 'Risk Road' highlight.

```bash
streamlit run app_demo.py
```

## Project Structure

```
cctvrrfd/
├── app.py                   # Main Streamlit application (Live Stream Version)
├── app_demo.py              # Streamlit application (Demo Version, uses local videos)
├── flood_detection_3lvl.py  # Simulated AI classification module
├── requirements.txt         # Python dependency list
├── README.md                # This file
└── data/
    ├── DEMNAS_1209-31_v1.0.tif # Digital Elevation Model (DEM)
    ├── roads_bandung.geojson   # Road network data (requires external download)
    ├── cctv_locations.csv      # CCTV data (ID, Lat, Lon, Stream/Video URL)
    ├── cctv_locations_demo_video.csv # CCTV data (ID, Lat, Lon, Local Video URL)
    └── flood_detection_results.csv # Log file for detection results (auto-generated)
```
