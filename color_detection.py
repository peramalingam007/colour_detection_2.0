import cv2
import numpy as np
import pandas as pd

clicked = False
r = g = b = x_pos = y_pos = 0

def get_color_name(R, G, B):
    """
    Finds the closest matching color name for a given RGB value from a CSV file.
    """
    try:
        index = ["color", "color_name", "hex", "R", "G", "B"]
        csv = pd.read_csv('colors.csv', names=index, header=None)
    except FileNotFoundError:
        print("Error: 'colors.csv' not found. Please make sure the file is in the correct directory.")
        return "Unknown"

    minimum = 10000
    cname = "Unknown"
    for i in range(len(csv)):
        d = abs(R - int(csv.loc[i, "R"])) + abs(G - int(csv.loc[i, "G"])) + abs(B - int(csv.loc[i, "B"]))
        if d <= minimum:
            minimum = d
            cname = csv.loc[i, "color_name"]
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
        b, g, r = param["image"][y, x]
        b = int(b)
        g = int(g)
        r = int(r)

def process_image(image_path):
    """
    Processes an image for color detection.
    Returns the image with color information drawn on it.
    """
    global clicked
    img = cv2.imread(image_path)
    if img is None:
        return None

    window_name = "Color Detection"
    cv2.namedWindow(window_name)
    params = {"image": img}
    cv2.setMouseCallback(window_name, draw_function, params)

    while True:
        cv2.imshow(window_name, img)

        if clicked:
            cv2.rectangle(img, (20, 20), (750, 60), (b, g, r), -1)

            color_name = get_color_name(r, g, b)
            text = color_name + ' R=' + str(r) + ' G=' + str(g) + ' B=' + str(b)

            if r + g + b >= 600:
                cv2.putText(img, text, (50, 50), 2, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(img, text, (50, 50), 2, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            clicked = False

        if cv2.waitKey(20) & 0xFF == 27:
            break

    cv2.destroyAllWindows()
    return img

if __name__ == '__main__':
    test_image = np.zeros((400, 600, 3), dtype=np.uint8)
    test_image[:] = (255, 255, 255) # White background
    cv2.imwrite("test_image.jpg", test_image)
    process_image('test_image.jpg')
