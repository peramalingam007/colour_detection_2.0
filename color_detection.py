import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans # For dominant color analysis
from collections import Counter # For counting pixel occurrences

# Global variables for mouse event handling (for image mode)
clicked = False
r = g = b = x_pos = y_pos = 0

# Load the color data once globally to avoid reloading on every click/frame
try:
    index = ["color", "color_name", "hex", "R", "G", "B"]
    COLOR_CSV_DATA = pd.read_csv('colors.csv', names=index, header=None)
except FileNotFoundError:
    print("Error: 'colors.csv' not found. Please make sure the file is in the correct directory.")
    COLOR_CSV_DATA = None # Set to None if file not found to handle gracefully

def get_color_name(R, G, B):
    """
    Finds the closest matching color name for a given RGB value from the pre-loaded CSV data.
    """
    if COLOR_CSV_DATA is None:
        return "Unknown" # Handle case where CSV was not loaded

    minimum = 10000
    cname = "Unknown"
    # Iterate over the pre-loaded DataFrame
    for i in range(len(COLOR_CSV_DATA)):
        d = abs(R - int(COLOR_CSV_DATA.loc[i, "R"])) + abs(G - int(COLOR_CSV_DATA.loc[i, "G"])) + abs(B - int(COLOR_CSV_DATA.loc[i, "B"]))
        if d <= minimum:
            minimum = d
            cname = COLOR_CSV_DATA.loc[i, "color_name"]
    return cname

def analyze_dominant_colors(image_array, k=5):
    """
    Analyzes an image to find k dominant colors using K-Means clustering.
    Returns a list of (RGB, percentage, name, hex) for each dominant color.
    """
    if image_array is None or image_array.size == 0:
        return []

    # Resize to a smaller image for faster clustering
    h, w, _ = image_array.shape
    max_dim = 100 # Max dimension for the smaller image to cluster
    if h > w:
        new_h = max_dim
        new_w = int(w / h * new_h)
    else:
        new_w = max_dim
        new_h = int(h / w * new_w)
    
    # Ensure dimensions are at least 1x1
    new_h = max(1, new_h)
    new_w = max(1, new_w)

    image_small = cv2.resize(image_array, (new_w, new_h), interpolation=cv2.INTER_AREA)
    pixels = image_small.reshape(-1, 3)

    # Handle cases where there might be fewer unique colors than k
    num_unique_pixels = np.unique(pixels, axis=0).shape[0]
    n_clusters = min(k, num_unique_pixels)
    if n_clusters == 0:
        return []

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    kmeans.fit(pixels)

    colors = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_
    counts = Counter(labels)
    total_pixels = len(pixels)
    
    color_info = []
    for i in range(n_clusters):
        rgb = colors[i]
        percent = counts[i] / total_pixels
        name = get_color_name(rgb[2], rgb[1], rgb[0]) # OpenCV stores as BGR, convert to RGB for get_color_name
        hex_val = '#%02x%02x%02x' % (rgb[2], rgb[1], rgb[0]) # Assuming BGR from OpenCV for cluster centers
        color_info.append(((rgb[2], rgb[1], rgb[0]), percent, name, hex_val)) # Store as RGB tuple
    
    color_info.sort(key=lambda x: x[1], reverse=True)
    return color_info

def draw_info_on_frame(frame, dominant_colors):
    """Draws dominant color information on the frame."""
    y_offset = 30
    for i, color_data in enumerate(dominant_colors):
        rgb_color, percent, name, hex_val = color_data
        
        # Create a small color patch
        cv2.rectangle(frame, (frame.shape[1] - 180, y_offset + i * 40), (frame.shape[1] - 150, y_offset + 25 + i * 40), rgb_color, -1)
        cv2.rectangle(frame, (frame.shape[1] - 180, y_offset + i * 40), (frame.shape[1] - 150, y_offset + 25 + i * 40), (255, 255, 255), 1) # Border

        text = f"{name} ({percent:.0%})"
        text_color = (255, 255, 255) # White text for better visibility
        # Add a dark background for the text for better contrast
        cv2.rectangle(frame, (frame.shape[1] - 145, y_offset + i * 40), (frame.shape[1] - 10, y_offset + 25 + i * 40), (0, 0, 0, 150), -1)
        cv2.putText(frame, text, (frame.shape[1] - 145, y_offset + 20 + i * 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    return frame

def process_image_mode(image_path):
    """
    Processes an image for single-click color detection (original functionality).
    """
    global clicked, r, g, b, x_pos, y_pos
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}. Please check the path.")
        return

    window_name = "Color Detection - Image Mode (Double Click)"
    cv2.namedWindow(window_name)
    params = {"image": img}
    cv2.setMouseCallback(window_name, draw_function, params)

    last_drawn_img = img.copy() # Keep a copy for consistent display
    
    while True:
        display_img = img.copy() # Start fresh each frame for drawing

        if clicked:
            # Draw a rectangle with the detected color
            cv2.rectangle(display_img, (20, 20), (750, 60), (b, g, r), -1)

            # Get the color name and construct the text string
            color_name = get_color_name(r, g, b)
            text = f"{color_name} R={r} G={g} B={b}"

            # Choose text color based on the background color for readability
            text_color = (0, 0, 0) if (r + g + b) / 3 >= 180 else (255, 255, 255) # Avg pixel value threshold

            # Put the text on the image
            cv2.putText(display_img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2, cv2.LINE_AA)
            
            last_drawn_img = display_img.copy() # Update the last drawn image
            clicked = False # Reset clicked after drawing once
        
        cv2.imshow(window_name, last_drawn_img) # Always show the last drawn image (or img if nothing clicked)

        key = cv2.waitKey(1) & 0xFF
        if key == 27: # ESC key
            break
    cv2.destroyAllWindows()


def process_live_video_mode(num_colors_to_detect=5):
    """
    Processes live webcam feed to detect dominant colors in real-time.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream or file. Check webcam connection.")
        return

    window_name = "Color Detection - Live Video Mode"
    cv2.namedWindow(window_name)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # Process dominant colors less frequently for performance
        dominant_colors = []
        if frame_count % 30 == 0: # Analyze every 30 frames (approx. 1 second at 30fps)
            # OpenCV captures in BGR, analyze_dominant_colors expects BGR as well for now
            # (get_color_name inside converts it to RGB)
            dominant_colors = analyze_dominant_colors(frame, k=num_colors_to_detect)
        
        # Display current dominant colors
        if dominant_colors:
            frame = draw_info_on_frame(frame, dominant_colors)
        else:
             cv2.putText(frame, "Analyzing...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(window_name, frame)

        frame_count += 1
        key = cv2.waitKey(1) & 0xFF
        if key == 27: # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # --- Create a sample test image if it doesn't exist ---
    test_image_path = 'test_image.jpg'
    try:
        with open(test_image_path, 'rb') as f:
            pass # File exists
    except FileNotFoundError:
        print(f"Creating a sample image: {test_image_path}")
        test_image = np.zeros((400, 600, 3), dtype=np.uint8)
        # Draw some distinct color areas for testing (BGR format)
        test_image[:, :200] = (255, 0, 0)   # Blue
        test_image[:, 200:400] = (0, 255, 0) # Green
        test_image[:, 400:] = (0, 0, 255)   # Red
        cv2.rectangle(test_image, (100, 100), (300, 300), (0, 255, 255), -1) # Yellow
        cv2.imwrite(test_image_path, test_image)

    # --- Ensure 'colors.csv' exists for this to work correctly ---
    # If you don't have one, create a simple one in the same directory:
    # color,color_name,hex,R,G,B
    # red,Red,#FF0000,255,0,0
    # green,Green,#00FF00,0,255,0
    # blue,Blue,#0000FF,0,0,255
    # yellow,Yellow,#FFFF00,255,255,0
    # white,White,#FFFFFF,255,255,255
    # black,Black,#000000,0,0,0
    # pink,Pink,#FFC0CB,255,192,203
    # orange,Orange,#FFA500,255,165,0
    # purple,Purple,#800080,128,0,128
    # brown,Brown,#A52A2A,165,42,42
    # gray,Gray,#808080,128,128,128
    # cyan,Cyan,#00FFFF,0,255,255
    # magenta,Magenta,#FF00FF,255,0,255


    print("\nSelect a mode:")
    print("1. Image Mode (Double-click to detect color)")
    print("2. Live Video Mode (Dominant colors)")
    choice = input("Enter your choice (1 or 2): ")

    if choice == '1':
        print("\nStarting Image Mode. Double-click on the image to detect colors. Press 'ESC' to exit.")
        process_image_mode(test_image_path)
    elif choice == '2':
        print("\nStarting Live Video Mode. Dominant colors will be shown in real-time. Press 'ESC' to exit.")
        process_live_video_mode(num_colors_to_detect=5) # You can change the number of colors
    else:
        print("Invalid choice. Exiting.")
