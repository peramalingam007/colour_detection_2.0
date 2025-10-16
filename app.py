import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image

# Function to get color name from RGB
def get_color_name(rgb, colors_df):
    r, g, b = rgb
    distances = np.sqrt((colors_df['R'] - r)**2 + (colors_df['G'] - g)**2 + (colors_df['B'] - b)**2)
    closest_color_index = distances.argmin()
    return colors_df.loc[closest_color_index, 'Color Name']

# Load the color dataset
try:
    colors_df = pd.read_csv('colours_rgb_shades.csv')
except FileNotFoundError:
    st.error("Error: 'colours_rgb_shades.csv' not found. Please make sure the file is in the same directory.")
    st.stop()

st.title("Multiple Color Detection Application - Level 2")
st.write("This application can detect multiple colors (Red, Green, Blue, Yellow, Orange, Purple) in images or videos.")

# Define color ranges in HSV
color_ranges = {
    'red': [
        (np.array([0, 100, 100]), np.array([10, 255, 255])),
        (np.array([160, 100, 100]), np.array([179, 255, 255]))
    ],
    'green': [(np.array([36, 100, 100]), np.array([86, 255, 255]))],
    'blue': [(np.array([100, 150, 0]), np.array([140, 255, 255]))],
    'yellow': [(np.array([20, 100, 100]), np.array([30, 255, 255]))],
    'orange': [(np.array([10, 100, 100]), np.array([20, 255, 255]))],
    'purple': [(np.array([125, 100, 100]), np.array([155, 255, 255]))]
}

# File uploader
uploaded_file = st.file_uploader("Upload an Image (JPG, PNG, JPEG) or a Video (MP4, AVI)", type=['jpg', 'png', 'jpeg', 'mp4', 'avi'])

# Debug mode toggle
debug_mode = st.checkbox("Enable Debug Mode (Show HSV Masks)", value=False)

if uploaded_file is not None:
    file_type = uploaded_file.type.split('/')[0]

    if file_type == 'image':
        image = Image.open(uploaded_file)
        img_np = np.array(image.convert('RGB'))
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        
        detected_colors = []
        labeled_img = img_bgr.copy()
        
        if debug_mode:
            st.subheader("HSV Masks (Debug View)")
            mask_columns = st.columns(len(color_ranges))
            
        for i, (color_name, hsv_ranges) in enumerate(color_ranges.items()):
            mask = np.zeros(hsv_img.shape[:2], dtype="uint8")
            for lower, upper in hsv_ranges:
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv_img, lower, upper))
            
            if debug_mode:
                mask_columns[i].image(mask, caption=f"{color_name.capitalize()} Mask", use_column_width=True)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if cv2.contourArea(contour) > 100:  # Contour area threshold
                    x, y, w, h = cv2.boundingRect(contour)
                    roi = labeled_img[y:y+h, x:x+w]
                    
                    # Calculate average color in the region
                    avg_rgb = np.mean(roi, axis=(0, 1)).astype(int)
                    color_label = get_color_name(avg_rgb, colors_df)
                    detected_colors.append((color_label, avg_rgb))
                    
                    # Draw bounding box and label
                    cv2.rectangle(labeled_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(labeled_img, color_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Check for solid color fallback
        if not detected_colors:
            avg_hsv = np.mean(hsv_img, axis=(0, 1)).astype(int)
            solid_color_found = False
            for color_name, hsv_ranges in color_ranges.items():
                for lower, upper in hsv_ranges:
                    if (lower <= avg_hsv).all() and (upper >= avg_hsv).all():
                        avg_rgb = np.mean(img_bgr, axis=(0, 1)).astype(int)
                        color_label = get_color_name(avg_rgb, colors_df)
                        detected_colors.append((f"Solid {color_label}", avg_rgb))
                        solid_color_found = True
                        break
                if solid_color_found:
                    break

        st.subheader("Processed Image")
        st.image(cv2.cvtColor(labeled_img, cv2.COLOR_BGR2RGB), caption="Detected Colors", use_column_width=True)
        
        st.subheader("Detected Colors and RGB Values")
        if detected_colors:
            for label, rgb in detected_colors:
                st.write(f"- **{label}**: RGB {tuple(rgb)}")
        else:
            st.write("No specified colors detected.")
            
    elif file_type == 'video':
        st.error("Video processing is not yet implemented in this version. Please upload an image.")