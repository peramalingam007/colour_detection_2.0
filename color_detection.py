import cv2
import numpy as np
import pandas as pd
import os # Added for checking file existence

# Global variables for mouse click event
clicked = False
r = g = b = x_pos = y_pos = 0

def get_color_name(R, G, B):
    """
    Finds the closest matching color name for a given RGB value from 'colors.csv'.
    """
    # Check if colors.csv exists
    if not os.path.exists('colors.csv'):
        print("Error: 'colors.csv' not found. Please make sure the file is in the same directory as the script.")
        # Provide a default message or handle more gracefully
        return "Unknown Color (CSV not found)"

    try:
        # Define column names for the CSV, assuming no header in the CSV
        index = ["color_id", "color_name", "hex", "R_val", "G_val", "B_val"] # Changed 'color', 'R', 'G', 'B' to avoid conflict and be more descriptive
        csv_df = pd.read_csv('colors.csv', names=index, header=None)
    except Exception as e: # Catch broader exceptions during CSV loading
        print(f"Error loading colors.csv: {e}")
        return "Unknown Color (CSV Load Error)"

    minimum = 10000
    cname = "Unknown Color"
    for i in range(len(csv_df)):
        # Ensure values are integers before comparison
        # Use try-except for robust parsing in case of bad data in CSV
        try:
            csv_R = int(csv_df.loc[i, "R_val"])
            csv_G = int(csv_df.loc[i, "G_val"])
            csv_B = int(csv_df.loc[i, "B_val"])
            d = abs(R - csv_R) + abs(G - csv_G) + abs(B - csv_B)
            if d < minimum: # Changed <= to < to prefer earlier entries if diff is same, or just for consistency
                minimum = d
                cname = csv_df.loc[i, "color_name"]
        except ValueError:
            # Skip rows with invalid integer conversion
            continue
    return cname

def draw_function(event, x, y, flags, param):
    """
    Callback function to handle mouse events for color detection.
    """
    global b, g, r, x_pos, y_pos, clicked
    if event == cv2.EVENT_LBUTTONDBLCLK:
        clicked = True
        x_pos = x
        y_pos = y
        # Access the image directly from param dictionary
        b_val, g_val, r_val = param["image"][y, x]
        b = int(b_val)
        g = int(g_val)
        r = int(r_val)

def process_image(image_path):
    """
    Processes an image for color detection.
    Returns the image with color information drawn on it.
    """
    global clicked, r, g, b # Added r, g, b to global to reset for each new image
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}. Please check the path and file format.")
        return None

    # Reset clicked state and color for a new image processing
    clicked = False
    r = g = b = 0

    window_name = "Color Detection"
    cv2.namedWindow(window_name)
    params = {"image": img} # Pass the image to the callback function
    cv2.setMouseCallback(window_name, draw_function, params)

    while True:
        # Create a copy of the original image for drawing, so previous drawings don't persist
        display_img = img.copy()

        if clicked:
            # Draw rectangle to show detected color
            cv2.rectangle(display_img, (20, 20), (750, 60), (b, g, r), -1)

            color_name = get_color_name(r, g, b)
            text = f"{color_name} R={r} G={g} B={b}" # Using f-string for cleaner text formatting

            # Determine text color based on background luminance
            # If background is bright, use black text; otherwise, white text
            if r + g + b >= 600: # Threshold for brightness
                text_color = (0, 0, 0) # Black
            else:
                text_color = (255, 255, 255) # White
            
            cv2.putText(display_img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2, cv2.LINE_AA) # Changed font and added thickness

        cv2.imshow(window_name, display_img) # Show the image with drawings

        # Wait for a key press (20ms delay), check for 'Esc' key (ASCII 27)
        key = cv2.waitKey(20) & 0xFF
        if key == 27: # ASCII for 'Esc' key
            break
        elif key != 255: # Any other key press, reset click
             clicked = False # Reset clicked state after processing one click

    cv2.destroyAllWindows()
    return img # Return the original image (or you could return the last display_img)

if __name__ == '__main__':
    # Create a test image dynamically or specify an existing image path
    test_image_name = "sample_image.jpg"
    
    # Create a simple test image (e.g., a gradient or blocks of color)
    img_height, img_width = 400, 600
    test_image_data = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    
    # Fill with some colors for testing
    test_image_data[:, :img_width//3] = (255, 0, 0)      # Blue part (BGR)
    test_image_data[:, img_width//3:2*img_width//3] = (0, 255, 0) # Green part (BGR)
    test_image_data[:, 2*img_width//3:] = (0, 0, 255)    # Red part (BGR)

    cv2.imwrite(test_image_name, test_image_data)
    
    # Process the created test image
    process_image(test_image_name)

    # You can also use an existing image by providing its path:
    # process_image('path/to/your/image.jpg')
