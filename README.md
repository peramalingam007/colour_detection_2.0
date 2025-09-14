🎨 Interactive Color Detection Application
This is a web application built with Python, Streamlit, and OpenCV that allows you to detect colors from both images and live video streams from your webcam.

✨ Features
Image Color Detection: Upload an image (JPG, PNG) and the app will identify the color at the center of the image.

Live Video Color Detection: Use your webcam to detect colors in real-time from the video feed. The color at the center of the frame is continuously identified.

User-Friendly Interface: A simple and intuitive web interface powered by Streamlit.

Comprehensive Color Data: Uses a dataset of over 1,000 color names for accurate identification.

🚀 How to Run this Project
Prerequisites
Python 3.7 or higher

An IDE or code editor (like VS Code)

Git installed on your system

1. Clone the Repository
First, clone this repository to your local machine:

git clone <your-repository-url>
cd <repository-directory>

2. Set Up a Virtual Environment (Recommended)
It's a good practice to create a virtual environment to manage project dependencies.

# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

3. Install the Dependencies
Install all the necessary libraries using the requirements.txt file:

pip install -r requirements.txt

4. Get the Full colors.csv File
The colors.csv file provided here is a sample. You need a complete dataset for better results. You can find a good one here:

Download colors.csv from this GitHub Gist

Download it and place it in the root directory of your project.

5. Run the Streamlit Application
Now, you can run the Streamlit app with the following command:

streamlit run app.py

Your web browser should open a new tab with the application running at http://localhost:8501.

📦 Deploying on Streamlit Community Cloud
You can deploy this application for free on Streamlit Community Cloud.

Push your project to a GitHub repository. Make sure your repository includes:

app.py

colors.csv

requirements.txt

This README.md file

Sign up for Streamlit Community Cloud: Go to share.streamlit.io and sign up using your GitHub account.

Deploy the app:

Click on the "New app" button.

Choose your repository and the branch (usually main).

Make sure the "Main file path" is set to app.py.

Click "Deploy!".

Streamlit will then build and deploy your application, and you'll get a public URL to share your project.

💡 How It Works
OpenCV: This library is used for all the image and video processing tasks, like reading the image/video frames and extracting pixel data.

Pandas: It's used to efficiently read and process the colors.csv file, which contains our color dataset.

Streamlit: This framework allows us to create the interactive web application with just Python code, handling things like file uploads, webcam access, and displaying results.

Color Logic: For a given RGB value from a pixel, the application calculates the Euclidean distance to every color in the colors.csv dataset and identifies the color with the minimum distance as the closest match.

Feel free to contribute to this project by forking the repository and submitting a pull request. If you encounter any issues, please open an issue on GitHub.
