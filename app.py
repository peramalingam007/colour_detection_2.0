import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
from collections import Counter

# --- CORE COLOR DETECTION LOGIC ---

# Cache the color data to prevent reloading on every interaction
@st.cache_data
def load_color_data():
    """Loads color data from colors.csv into a pandas DataFrame."""
    try:
        index = ["color", "color_name", "hex", "R", "G", "B"]
        df = pd.read_csv('colors.csv', names=index, header=None)
        return df
    except FileNotFoundError:
        st.error("Error: 'colors.csv' not found. Please make sure the file is in the root of your repository.")
        return None

# Function to find the closest color name for a given RGB value
def get_color_name(R, G, B, color_data):
    """Calculates the distance between the given RGB and all colors in the dataset to find the closest match."""
    if color_data is None:
        return "Unknown", "#000000"
    minimum = float('inf')
    cname = "Unknown"
    hex_value = "#000000"
    for i in range(len(color_data)):
        d = abs(R - int(color_data.loc[i, "R"])) + abs(G - int(color_data.loc[i, "G"])) + abs(B - int(color_data.loc[i, "B"]))
        if d < minimum:
            minimum = d
            cname = color_data.loc[i, "color_name"]
            hex_value = color_data.loc[i, "hex"]
    return cname, hex_value

# --- NEW: Dominant Color Analysis Function ---
def analyze_image_colors(image_array, k=5):
    """
    Analyzes an image to find the K most dominant colors using K-Means clustering.
    Returns a list of tuples, each containing (RGB color, percentage, color_name, hex_code).
    """
    # For performance, resize the image to a smaller size.
    # This doesn't significantly affect the dominant color results.
    h, w, _ = image_array.shape
    w_new = 200
    h_new = int(h / w * w_new)
    image_small = cv2.resize(image_array, (w_new, h_new), interpolation=cv2.INTER_AREA)

    # Reshape the image to be a list of pixels (N_pixels, 3)
    pixels = image_small.reshape(-1, 3)

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(pixels)

    # Get the cluster centers (dominant colors) and labels
    colors = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_

    # Count the pixels in each cluster to find the percentage
    counts = Counter(labels)
    total_pixels = len(pixels)
    percentages = {i: counts[i] / total_pixels for i in range(k)}

    # Get color names for each dominant color
    color_info = []
    for i in range(k):
        rgb = colors[i]
        percent = percentages[i]
        name, hex_code = get_color_name(rgb[0], rgb[1], rgb[2], color_df)
        color_info.append((rgb, percent, name, hex_code))

    # Sort colors by percentage (most dominant first)
    color_info.sort(key=lambda x: x[1], reverse=True)
    return color_info

# --- STREAMLIT UI SETUP ---

st.set_page_config(page_title="Color Palette Analyzer", page_icon="🎨", layout="wide")
color_df = load_color_data()

# --- SIDEBAR ---
st.sidebar.title("Color Palette Analyzer")
st.sidebar.write("---")
st.sidebar.header("Options")
app_mode = st.sidebar.selectbox("Choose an analysis mode:", ["About", "Analyze Image Palette", "Analyze Live Video"])
st.sidebar.write("---")

# --- MAIN PAGE CONTENT ---

if app_mode == "About":
    st.title("Welcome to the Color Palette Analyzer! 🎨")
    st.markdown("This app automatically detects the dominant colors in an image or live video feed.")
    st.markdown("Select a mode from the sidebar to begin:")
    st.markdown("1.  **Analyze Image Palette:** Upload an image to extract its main color palette.")
    st.markdown("2.  **Analyze Live Video:** Use your webcam to analyze the colors in your environment in real-time.")
    st.image("https://images.unsplash.com/photo-1558244402-286dd748c595?w=900", caption="Extract the palette from any source.")

elif app_mode == "Analyze Image Palette":
    st.header("Analyze an Image's Color Palette")
    num_colors = st.sidebar.slider("Number of Colors to Detect", 2, 15, 5)
    uploaded_file = st.file_uploader("Upload an Image (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])

    if uploaded_file and color_df is not None:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        with col2:
            st.subheader("Dominant Color Palette")
            with st.spinner("Analyzing image... this may take a moment."):
                image = Image.open(uploaded_file).convert('RGB')
                img_array = np.array(image)
                dominant_colors = analyze_image_colors(img_array, k=num_colors)

                for color in dominant_colors:
                    rgb, percent, name, hex_val = color
                    
                    # Create a layout for each color result
                    c_col1, c_col2, c_col3 = st.columns([1, 4, 2])
                    with c_col1:
                        st.markdown(f'<div style="width:50px; height:50px; background-color:rgb({rgb[0]},{rgb[1]},{rgb[2]}); border: 1px solid #d3d3d3; border-radius: 8px;"></div>', unsafe_allow_html=True)
                    with c_col2:
                        st.write(f"**{name}**")
                        st.write(f"RGB: ({rgb[0]}, {rgb[1]}, {rgb[2]}) | HEX: {hex_val}")
                    with c_col3:
                        st.progress(percent)
                        st.write(f"{percent:.1%}")

elif app_mode == "Analyze Live Video":
    st.header("Analyze Live Video Feed")
    num_colors_video = st.sidebar.slider("Number of Colors to Detect", 2, 10, 4)
    st.info("Allow webcam access. The dominant colors in the frame will be analyzed in real-time.")
    
    run = st.checkbox('Start Webcam')
    
    if run and color_df is not None:
        col1, col2 = st.columns([2, 1])
        with col1:
            FRAME_WINDOW = st.image([])
        with col2:
            results_placeholder = st.empty()

        camera = cv2.VideoCapture(0)
        frame_count = 0
        
        while run:
            ret, frame = camera.read()
            if not ret:
                st.error("Failed to capture image from webcam.")
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame_rgb)
            
            # Optimization: Analyze every Nth frame to avoid lag
            if frame_count % 15 == 0:
                with results_placeholder.container():
                    st.subheader("Live Palette")
                    dominant_colors = analyze_image_colors(frame_rgb, k=num_colors_video)
                    for color in dominant_colors:
                        rgb, percent, name, hex_val = color
                        
                        st.markdown(f"""
                        <div style="display: flex; align-items: center; margin-bottom: 10px;">
                            <div style="width:25px; height:25px; background-color:rgb({rgb[0]},{rgb[1]},{rgb[2]}); border-radius: 5px; margin-right: 10px;"></div>
                            <div>
                                <div style="font-weight: bold;">{name}</div>
                                <div style="font-size: 0.8em;">{percent:.0%}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            
            frame_count += 1
        
        camera.release()
    else:
        st.write("Webcam is stopped.")

