import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
from collections import Counter

@st.cache_data
def load_color_data():
"""Loads color data from colors.csv into a pandas DataFrame."""
try:
index = ["color", "color_name", "hex", "R", "G", "B"]
df = pd.read_csv('colors.csv', names=index, header=None)
return df
except FileNotFoundError:
st.error("Error: 'colors.csv' not found. Please ensure the file is in the root of your repository.")
return None

def get_color_name(R, G, B, color_data):
"""Finds the closest color name for a given RGB value."""
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

def analyze_colors(image_array, k=5):
"""Analyzes an image to find K dominant colors using K-Means clustering."""
h, w, _ = image_array.shape
w_new = min(w, 200)
h_new = int(h / w * w_new)
image_small = cv2.resize(image_array, (w_new, h_new), interpolation=cv2.INTER_AREA)
pixels = image_small.reshape(-1, 3)

kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')  
kmeans.fit(pixels)  

colors = kmeans.cluster_centers_.astype(int)  
labels = kmeans.labels_  
counts = Counter(labels)  
total_pixels = len(pixels)  
percentages = {i: counts[i] / total_pixels for i in range(k)}  

color_info = []  
for i in range(k):  
    rgb = colors[i]  
    percent = percentages[i]  
    name, hex_code = get_color_name(rgb[0], rgb[1], rgb[2], color_df)  
    color_info.append((rgb, percent, name, hex_code))  

color_info.sort(key=lambda x: x[1], reverse=True)  
return color_info, kmeans

def create_highlighted_image(original_image_array, kmeans_model, selected_cluster_index):
"""Generates an image where only the selected dominant color is shown."""
pixels = original_image_array.reshape(-1, 3)
full_size_labels = kmeans_model.predict(pixels)
mask = (full_size_labels == selected_cluster_index).reshape(original_image_array.shape[:2])

grayscale_image = cv2.cvtColor(original_image_array, cv2.COLOR_RGB2GRAY)  
grayscale_image_3ch = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2RGB)  
highlighted_image = np.where(mask[:, :, np.newaxis], original_image_array, grayscale_image_3ch)  
return highlighted_image

def hex_to_rgb(hex_code):
"""Converts a hex color string to an (R, G, B) tuple."""
hex_code = hex_code.lstrip('#')
return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))

st.set_page_config(page_title="Color Palette Analyzer", page_icon="🎨", layout="wide")
color_df = load_color_data()

if 'highlight_index' not in st.session_state:
st.session_state.highlight_index = None

st.sidebar.title("Color Palette Analyzer")
st.sidebar.write("---")
st.sidebar.header("Options")
app_mode = st.sidebar.selectbox("Choose an analysis mode:", ["About", "Analyze Image Palette", "Analyze Live Video", "Manual Color Picker"])
st.sidebar.write("---")

if app_mode == "About":
st.title("Welcome to the Color Palette Analyzer! 🎨")
st.markdown("This app brings together multiple tools for color analysis. Select a mode from the sidebar:")
st.markdown("- Analyze Image Palette: Upload an image to find its dominant colors and highlight them.")
st.markdown("- Analyze Live Video: Use your webcam for real-time color palette detection.")
st.markdown("- Manual Color Picker: Choose a color from a palette to instantly identify its name.")
st.image("https://images.unsplash.com/photo-1558244402-286dd748c595?w=900", caption="Extract and visualize the palette from any source.")

elif app_mode == "Analyze Image Palette":
st.header("Analyze an Image's Color Palette")
num_colors = st.sidebar.slider("Number of Colors to Detect", 2, 15, 5, key="img_slider")
uploaded_file = st.file_uploader("Upload an Image (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])

if uploaded_file and color_df is not None:  
    image = Image.open(uploaded_file).convert('RGB')  
    img_array = np.array(image)  
    col1, col2 = st.columns([2, 1])  

    with col1:  
        image_placeholder = st.empty()  
    with col2:  
        st.subheader("Dominant Color Palette")  
        with st.spinner("Analyzing image..."):  
            dominant_colors, kmeans_model = analyze_colors(img_array, k=num_colors)  
            st.session_state.kmeans_model = kmeans_model  
            for i, color in enumerate(dominant_colors):  
                rgb, percent, name, hex_val = color  
                cols = st.columns([1, 4, 2])  
                with cols[0]:  
                    st.markdown(f'<div style="width:30px; height:30px; background-color:{hex_val}; border: 1px solid #d3d3d3; border-radius: 5px;"></div>', unsafe_allow_html=True)  
                with cols[1]:  
                    st.write(f"**{name}** ({percent:.1%})")  
                with cols[2]:  
                    if st.button(f"Highlight", key=f"highlight_{i}"):  
                        st.session_state.highlight_index = i  
        if st.session_state.highlight_index is not None:  
            if st.button("Show Original Image"):  
                st.session_state.highlight_index = None  
      
    if st.session_state.highlight_index is not None:  
        with st.spinner("Generating highlight..."):  
            highlighted_img = create_highlighted_image(img_array, st.session_state.kmeans_model, st.session_state.highlight_index)  
            image_placeholder.image(highlighted_img, caption=f"Highlighting: {dominant_colors[st.session_state.highlight_index][2]}", use_column_width=True)  
    else:  
        image_placeholder.image(img_array, caption="Original Uploaded Image", use_column_width=True)

elif app_mode == "Analyze Live Video":
st.header("Analyze Live Video Feed")
num_colors_video = st.sidebar.slider("Number of Colors to Detect", 2, 10, 4, key="video_slider")
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
            camera.release()  
            break  
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
        FRAME_WINDOW.image(frame_rgb)  
        if frame_count % 15 == 0:  
            with results_placeholder.container():  
                st.subheader("Live Palette")  
                dominant_colors, _ = analyze_colors(frame_rgb, k=num_colors_video)  
                for color in dominant_colors:  
                    rgb, percent, name, hex_val = color  
                    st.markdown(f"""<div style="display: flex; align-items: center; margin-bottom: 10px;"><div style="width:25px; height:25px; background-color:{hex_val}; border-radius: 5px; margin-right: 10px;"></div><div><div style="font-weight: bold;">{name}</div><div style="font-size: 0.8em;">{percent:.0%}</div></div></div>""", unsafe_allow_html=True)  
        frame_count += 1  
    camera.release()  
else:  
    st.write("Webcam is stopped.")

elif app_mode == "Manual Color Picker":
st.header("Manual Color Picker")
st.write("Use the color wheel to select a color and find its closest name.")

picked_color_hex = st.color_picker("Pick a color", "#FFFFFF")  

if picked_color_hex and color_df is not None:  
    r, g, b = hex_to_rgb(picked_color_hex)  
      
    color_name, _ = get_color_name(r, g, b, color_df)  
      
    st.write("---")  
    st.subheader("Result")  
    col1, col2 = st.columns([1, 4])  
    with col1:  
        st.markdown(f'<div style="width:80px; height:80px; background-color:{picked_color_hex}; border: 1px solid #d3d3d3; border-radius: 8px;"></div>', unsafe_allow_html=True)  
    with col2:  
        st.markdown(f"### Closest Match: **{color_name}**")  
        st.write(f"**HEX:** {picked_color_hex}")  
        st.write(f"**RGB:** ({r}, {g}, {b})")

