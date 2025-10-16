import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image  # Changed 'pil' to 'PIL'
import tempfile
import os

# function to find closest color name
def find_closest_color(rgb, color_df):
    r, g, b = rgb
    differences = []
    for index, row in color_df.iterrows():
        try:
            # Ensure 'r;g;b dec' is a string before splitting
            color_rgb_str = str(row['r;g;b dec'])
            color_rgb = np.array([int(x) for x in color_rgb_str.split(';')])
            diff = np.sqrt(sum((rgb - color_rgb) ** 2))
            differences.append((diff, row['color name']))
        except (ValueError, KeyError): # Changed 'valueerror', 'keyerror' to 'ValueError', 'KeyError'
            continue
    return min(differences, key=lambda x: x[0])[1] if differences else "Unknown Color" # Changed "unknown color" to "Unknown Color"

# load color dataset
@st.cache_data # Decorators should not have parentheses when used like this unless passing arguments
def load_color_dataset():
    try:
        color_df = pd.read_csv('colours_rgb_shades.csv')
        if not all(col in color_df.columns for col in ['color name', 'r;g;b dec']):
            raise ValueError("Invalid color dataset format. Required columns: 'color name', 'r;g;b dec'") # Changed 'valueerror' to 'ValueError'
        
        # Filter rows where 'r;g;b dec' is a valid RGB string
        color_df = color_df[color_df['r;g;b dec'].astype(str).str.match(r'^\d+;\d+;\d+$', na=False)] # Corrected na=false to na=False
        
        if color_df.empty:
            raise ValueError("No valid RGB data found in the dataset") # Changed 'valueerror' to 'ValueError'
        return color_df
    except FileNotFoundError: # Changed 'filenotfounderror' to 'FileNotFoundError'
        st.error("colours_rgb_shades.csv file not found. Please ensure it is in the same directory.")
        return None # Changed 'none' to 'None'
    except Exception as e:
        st.error(f"Error loading colours_rgb_shades.csv: {str(e)}")
        return None # Changed 'none' to 'None'

# process a single frame for color detection
def process_frame(frame, color_df):
    # Convert to HSV
    hsvframe = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # Changed 'cvtcolor', 'color_bgr2hsv' to 'cvtColor', 'COLOR_BGR2HSV'

    # Define wider HSV ranges
    red_lower1 = np.array([0, 100, 50], np.uint8)
    red_upper1 = np.array([15, 255, 255], np.uint8)
    red_lower2 = np.array([165, 100, 50], np.uint8)
    red_upper2 = np.array([180, 255, 255], np.uint8)
    green_lower = np.array([35, 30, 30], np.uint8)
    green_upper = np.array([85, 255, 255], np.uint8)
    blue_lower = np.array([90, 100, 50], np.uint8)
    blue_upper = np.array([140, 255, 255], np.uint8)

    # Create masks
    red_mask1 = cv2.inRange(hsvframe, red_lower1, red_upper1) # Changed 'inrange' to 'inRange'
    red_mask2 = cv2.inRange(hsvframe, red_lower2, red_upper2) # Changed 'inrange' to 'inRange'
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    green_mask = cv2.inRange(hsvframe, green_lower, green_upper) # Changed 'inrange' to 'inRange'
    blue_mask = cv2.inRange(hsvframe, blue_lower, blue_upper) # Changed 'inrange' to 'inRange'

    # Dilate masks
    kernel = np.ones((5, 5), "uint8")
    red_mask = cv2.dilate(red_mask, kernel)
    green_mask = cv2.dilate(green_mask, kernel)
    blue_mask = cv2.dilate(blue_mask, kernel)

    # Set to store unique detected colors (color_name, rgb)
    detected_colors = set()

    # Debug: sample HSV values at various points
    h, w, _ = hsvframe.shape
    debug_info = "Sample HSV values:\n"
    sample_points = [(0, 0), (w//2, h//2), (w-1, h-1)]
    for x, y in sample_points:
        hsv_value = hsvframe[y, x]
        debug_info += f"Pixel ({x}, {y}): HSV {hsv_value}\n"

    # Fallback: check if the entire image is a solid color
    avg_hsv = np.mean(hsvframe, axis=(0, 1)).astype(int)
    debug_info += f"Average HSV of image: {avg_hsv}\n"
    if (red_lower1[0] <= avg_hsv[0] <= red_upper1[0] or red_lower2[0] <= avg_hsv[0] <= red_upper2[0]) and \
       avg_hsv[1] >= 100 and avg_hsv[2] >= 50:
        avg_rgb = np.mean(frame, axis=(0, 1)).astype(int)
        b, g, r = avg_rgb
        rgb = np.array([r, g, b])
        color_name = find_closest_color(rgb, color_df)
        detected_colors.add((color_name, tuple(rgb)))
    elif green_lower[0] <= avg_hsv[0] <= green_upper[0] and avg_hsv[1] >= 30 and avg_hsv[2] >= 30:
        avg_rgb = np.mean(frame, axis=(0, 1)).astype(int)
        b, g, r = avg_rgb
        rgb = np.array([r, g, b])
        color_name = find_closest_color(rgb, color_df)
        detected_colors.add((color_name, tuple(rgb)))
    elif blue_lower[0] <= avg_hsv[0] <= blue_upper[0] and avg_hsv[1] >= 100 and avg_hsv[2] >= 50:
        avg_rgb = np.mean(frame, axis=(0, 1)).astype(int)
        b, g, r = avg_rgb
        rgb = np.array([r, g, b])
        color_name = find_closest_color(rgb, color_df)
        detected_colors.add((color_name, tuple(rgb)))

    # Function to process contours and label colors
    def process_contours(mask, box_color):
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # Changed 'findcontours', 'retr_tree', 'chain_approx_simple' to 'findContours', 'RETR_TREE', 'CHAIN_APPROX_SIMPLE'
        debug_info_contours = f"Number of contours (area > 100): {len([c for c in contours if cv2.contourArea(c) > 100])}\n" # Changed 'contourarea' to 'contourArea'
        for contour in contours:
            area = cv2.contourArea(contour) # Changed 'contourarea' to 'contourArea'
            if area > 100:  # Lowered threshold
                x, y, w, h = cv2.boundingRect(contour) # Changed 'boundingrect' to 'boundingRect'
                roi = frame[y:y+h, x:x+w]
                if roi.size == 0:
                    continue
                avg_color = np.mean(roi, axis=(0, 1)).astype(int)
                b, g, r = avg_color
                rgb = np.array([r, g, b])
                color_name = find_closest_color(rgb, color_df)
                detected_colors.add((color_name, tuple(rgb)))
                cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
                cv2.putText(frame, f"{color_name}", (x, y - 10), # Changed 'puttext' to 'putText'
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, box_color, 2) # Added thickness and changed 'font_hershey_simplex' to 'FONT_HERSHEY_SIMPLEX'
        return debug_info_contours

    # Process each color and collect debug info
    debug_info += "Red Mask:\n" + process_contours(red_mask, (0, 0, 255))
    debug_info += "Green Mask:\n" + process_contours(green_mask, (0, 255, 0))
    debug_info += "Blue Mask:\n" + process_contours(blue_mask, (255, 0, 0))

    return frame, detected_colors, debug_info

# main app
def main():
    st.title("Multiple Color Detection Application")
    st.write("Upload an image or video to detect colors (Red, Green, Blue regions).")

    # Load color dataset
    color_df = load_color_dataset()
    if color_df is None: # Changed 'none' to 'None'
        return

    # File upload
    uploaded_file = st.file_uploader("Choose an image or video", type=['png', 'jpg', 'jpeg', 'mp4', 'avi'])

    if uploaded_file is not None: # Changed 'none' to 'None'
        # Determine file type
        file_type = uploaded_file.type
        is_image = file_type.startswith('image')

        # Save uploaded file to a temporary location
        # Use a more robust way for suffix to avoid issues with .mp4 for images or .png for videos
        suffix = '.mp4' if not is_image else os.path.splitext(uploaded_file.name)[1]
        if not suffix: # Fallback if file has no extension
            suffix = '.tmp'
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        try:
            if is_image:
                # Process as image
                image = cv2.imread(tmp_file_path)
                if image is None: # Changed 'none' to 'None'
                    st.error("Failed to load image. Please check the file format.")
                    return
                processed_frame, detected_colors, debug_info = process_frame(image, color_df)
                st.image(processed_frame, channels="BGR", caption="Detected Colors") # Changed 'bgr' to 'BGR'

                # Display detected colors below the image
                if detected_colors:
                    detected_colors = sorted(detected_colors, key=lambda x: x[0])
                    color_text = " | ".join([f"{color_name} {rgb}" for color_name, rgb in detected_colors])
                    st.write("**Detected Colors:**")
                    st.text(color_text)
                else:
                    st.write("**Detected Colors:** None")

                # Display debug info
                st.write("**Debug Information:**")
                st.text(debug_info)

            else:
                # Process as video
                cap = cv2.VideoCapture(tmp_file_path) # Changed 'videocapture' to 'VideoCapture'
                if not cap.isOpened(): # Changed 'isopened' to 'isOpened'
                    st.error("Failed to load video. Please check the file format.")
                    return

                stframe = st.empty()
                color_display = st.empty()
                debug_display = st.empty()
                while cap.isOpened(): # Changed 'isopened' to 'isOpened'
                    ret, frame = cap.read()
                    if not ret:
                        break
                    processed_frame, detected_colors, debug_info = process_frame(frame, color_df)
                    stframe.image(processed_frame, channels="BGR", caption="Detected Colors") # Changed 'bgr' to 'BGR'

                    # Display detected colors below the video frame
                    if detected_colors:
                        detected_colors = sorted(detected_colors, key=lambda x: x[0])
                        color_text = " | ".join([f"{color_name} {rgb}" for color_name, rgb in detected_colors])
                        color_display.write(f"**Detected Colors:** {color_text}")
                    else:
                        color_display.write("**Detected Colors:** None")

                    # Display debug info
                    debug_display.write(f"**Debug Information:** {debug_info}")
                cap.release()

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

if __name__ == "__main__":
    main()
