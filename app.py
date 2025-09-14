import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image

# --- CORE COLOR DETECTION LOGIC ---

# Load the color data from the CSV file
# This is cached to prevent reloading on every interaction
@st.cache_data
def load_color_data():
    """
    Loads color data from colors.csv into a pandas DataFrame.
    """
    try:
        index = ["color", "color_name", "hex", "R", "G", "B"]
        # The file is expected to be in the same directory as the script
        df = pd.read_csv('colors.csv', names=index, header=None)
        return df
    except FileNotFoundError:
        st.error("Error: 'colors.csv' not found. Please make sure the file is in the root of your repository.")
        return None

# Function to find the closest color name for a given RGB value
def get_color_name(R, G, B, color_data):
    """
    Calculates the distance between the given RGB and all colors in the dataset
    to find the closest match.
    """
    if color_data is None:
        return "Unknown", "#000000"

    minimum = float('inf')
    cname = "Unknown"
    hex_value = "#000000"
    for i in range(len(color_data)):
        # The distance is calculated using the sum of absolute differences (a simple and effective metric)
        d = abs(R - int(color_data.loc[i, "R"])) + \
            abs(G - int(color_data.loc[i, "G"])) + \
            abs(B - int(color_data.loc[i, "B"]))
        if d < minimum:
            minimum = d
            cname = color_data.loc[i, "color_name"]
            hex_value = color_data.loc[i, "hex"]
    return cname, hex_value

# --- STREAMLIT UI SETUP ---

# Page Configuration
st.set_page_config(
    page_title="Color Detector Pro",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the data once
color_df = load_color_data()

# --- SIDEBAR ---
st.sidebar.title("Color Detector Pro")
st.sidebar.write("---")
st.sidebar.header("Options")
app_mode = st.sidebar.selectbox(
    "Choose a detection mode:",
    ["About", "Detect from Image", "Detect from Live Video", "Manual Color Picker"]
)
st.sidebar.write("---")
st.sidebar.info("This app uses a dataset of over 800 common color names to find the closest match for a selected color.")

# --- MAIN PAGE CONTENT ---

if app_mode == "About":
    st.title("Welcome to Color Detector Pro! 🎨")
    st.markdown("This application allows you to identify colors in three different ways. Select a mode from the sidebar to get started:")
    st.markdown("1.  **Detect from Image:** Upload your own image and find the color of any pixel.")
    st.markdown("2.  **Detect from Live Video:** Use your webcam to see color names in real-time.")
    st.markdown("3.  **Manual Color Picker:** Choose a color from a palette to learn its name.")
    st.image("https://images.unsplash.com/photo-1502691851199-522b516ac238?w=900", caption="A vibrant spectrum of colors.")


elif app_mode == "Detect from Image":
    st.header("Detect Color from an Image")
    uploaded_file = st.file_uploader("Upload an Image (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        st.subheader("Color Detection Result")
        col1, col2 = st.columns([1, 1])

        with col1:
            st.write("Your Image:")
            # We use PIL to open the image and then convert to a NumPy array
            image = Image.open(uploaded_file).convert('RGB')
            img_array = np.array(image)
            st.image(image, use_column_width=True)

        with col2:
            st.write("Click on the image to select a color:")
            # A simple trick to get click coordinates: wrap the image in a form
            # This is a known workaround in Streamlit for getting mouse events
            # Note: This is a placeholder for a more advanced component if one were available.
            # For now, we will detect the color from the center as a default.
            
            height, width, _ = img_array.shape
            center_y, center_x = height // 2, width // 2
            
            # Extract the RGB values from the center pixel
            r, g, b = img_array[center_y, center_x]
            
            st.info("Note: Streamlit doesn't support direct image clicks yet. The color below is from the center of the image.")

            color_name, hex_code = get_color_name(r, g, b, color_df)

            # Display the results in a visually appealing way
            st.markdown("### Detected Color")
            result_col1, result_col2 = st.columns([1, 3])
            with result_col1:
                 st.markdown(f'<div style="width:75px; height:75px; background-color:rgb({r},{g},{b}); border: 2px solid #d3d3d3; border-radius: 8px;"></div>', unsafe_allow_html=True)
            with result_col2:
                 st.write(f"**Name:** {color_name}")
                 st.write(f"**RGB:** ({r}, {g}, {b})")
                 st.write(f"**HEX:** {hex_code}")


elif app_mode == "Detect from Live Video":
    st.header("Detect Color from Live Video")
    st.info("Allow webcam access to start detecting colors from the center of the video feed.")
    
    run = st.checkbox('Start Webcam')
    FRAME_WINDOW = st.image([])
    
    if run:
        camera = cv2.VideoCapture(0)
        while run:
            ret, frame = camera.read()
            if not ret:
                st.error("Failed to capture image from webcam.")
                break
            
            h, w, _ = frame.shape
            
            # A circle in the center to guide the user
            cx, cy = w // 2, h // 2
            cv2.circle(frame, (cx, cy), 5, (255, 255, 255), 2)
            cv2.circle(frame, (cx, cy), 6, (0, 0, 0), 2) # black outline
            
            # Get BGR color from the center and convert to RGB
            b_val, g_val, r_val = frame[cy, cx]
            r, g, b = int(r_val), int(g_val), int(b_val)
            
            color_name, _ = get_color_name(r, g, b, color_df)
            
            # Create a header bar to display the color info
            header_text = f"Color: {color_name} | RGB: ({r},{g},{b})"
            text_color = (0, 0, 0) if r + g + b > 384 else (255, 255, 255) # Black or white text
            
            # Draw the info bar on the frame
            cv2.rectangle(frame, (0,0), (w, 40), (b, g, r), -1)
            cv2.putText(frame, header_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
            
            # Convert the frame back to RGB for Streamlit display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame_rgb)
        
        camera.release()
    else:
        st.write("Webcam is stopped.")

elif app_mode == "Manual Color Picker":
    st.header("Pick a Color Manually")
    st.info("Use the color picker below to select any color and find its name.")
    
    picked_color = st.color_picker("Choose a color:", "#FFFFFF")
    
    # The color picker returns a hex string, so we need to convert it to RGB
    hex_color = picked_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    color_name, hex_code = get_color_name(r, g, b, color_df)
    
    st.markdown("### Selected Color")
    res_col1, res_col2 = st.columns([1, 3])
    with res_col1:
         st.markdown(f'<div style="width:75px; height:75px; background-color:rgb({r},{g},{b}); border: 2px solid #d3d3d3; border-radius: 8px;"></div>', unsafe_allow_html=True)
    with res_col2:
         st.write(f"**Name:** {color_name}")
         st.write(f"**RGB:** ({r}, {g}, {b})")
         st.write(f"**HEX:** {picked_color.upper()}")

