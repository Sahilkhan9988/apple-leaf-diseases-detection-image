import os
import streamlit as st
import cv2
import numpy as np
import tempfile
from PIL import Image
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
from ultralytics import YOLO
import datetime
import plotly.express as px
import pandas as pd

# Base directory and output directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "runs")

# Create output directories for predictions
for subdir in ["predict_image", "predict_video", "predict_webcamera"]:
    os.makedirs(os.path.join(OUTPUT_DIR, subdir), exist_ok=True)

# Load the apple diseases detection model
apple_detection_model = YOLO("best11.pt")

# Function to add timestamp to a frame
def add_timestamp(frame):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    position = (10, 30)  # Top left corner
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    color = (0, 255, 0)  # Green color
    thickness = 2
    cv2.putText(frame, timestamp, position, font, font_scale, color, thickness)
    return frame

# Function to perform apple diseases detection on an image
def predict_image(model, img):
    results = model.predict(img)
    result_images = []
    for r in results:
        im_array = r.plot()
        im_array = add_timestamp(im_array)  # Add timestamp to each detected image
        result_images.append(im_array)
    return result_images

# VideoProcessor class for processing videos
class VideoProcessor:
    def __init__(self, video_path, output_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.output_path = output_path

    def __del__(self):
        self.cap.release()

    def process_video(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            # Perform apple diseases detection on the frame
            frame = detect_apple(apple_detection_model, frame)
            frame = add_timestamp(frame)  # Add timestamp to the frame
            # Save processed video frame
            filename = f"frame_{len(os.listdir(self.output_path))}.jpg"
            cv2.imwrite(os.path.join(self.output_path, filename), frame)
            # Yield the processed frame
            yield frame

# Function to perform apple diseases detection on a frame
def detect_apple(model, frame):
    results = model.predict(frame)
    if results:
        for result in results:
            frame = result.plot()  # Overlay detections on the frame
    return frame

# AppleDetectionProcessor class for webcam feed processing
class AppleDetectionProcessor:
    def __init__(self, output_path):
        self.output_path = output_path

    def __call__(self):
        return self

    def recv(self, frame):
        frame = frame.to_ndarray(format="bgr24")
        frame = detect_apple(apple_detection_model, frame)
        frame = add_timestamp(frame)  # Add timestamp to the frame
        # Save processed frame
        filename = f"detection_{len(os.listdir(self.output_path))}.jpg"
        cv2.imwrite(os.path.join(self.output_path, filename), frame)
        return av.VideoFrame.from_ndarray(frame, format="bgr24")
    
def compare_images(original_img, detected_img):
    st.subheader("Comparison")
    col1, col2 = st.columns(2)
    with col1:
        st.image(original_img, caption='Original Image', use_column_width=True)
    with col2:
        st.image(detected_img, caption='Detected Image', use_column_width=True)


# Home page function
def home_page():
    st.title("Apple Leaf Disease Detection Application")
    st.write("Welcome to our innovative Apple Leaf Detection app! Revolutionize your orchard management with precise leaf analysis at your fingertips.")
    image_path = "1-s2.0-S0168169922004100-gr8.jpg"
    st.image(image_path, caption='Your Profile Picture', use_column_width=True, )
    
def detection():
    st.title("Apple Diseases Detection")
    uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "jpeg", "png", "mp4"])

    if uploaded_file is not None:
        file_extension = uploaded_file.name.split('.')[-1].lower()

        if file_extension in ['jpg', 'jpeg', 'png']:
            # Image upload
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            img_array = np.array(image)
            detection_result_images = predict_image(apple_detection_model, img_array)

            # Compare the original image with the detected version
            for idx, detected_image in enumerate(detection_result_images, start=1):
                compare_images(img_array, detected_image)

            # Save detected images
            if st.button("Save Detected Images"):
                save_folder = os.path.join(OUTPUT_DIR, "predict_image")
                os.makedirs(save_folder, exist_ok=True)
                existing_files = os.listdir(save_folder)
                max_num = max([int(file.split('_')[1].split('.')[0]) for file in existing_files]) if existing_files else 0
                for idx, detected_image in enumerate(detection_result_images, start=1):
                    Image.fromarray(detected_image).save(os.path.join(save_folder, f"detection_{max_num + idx}.jpg"))
                st.success("Detected images saved successfully!")

        elif file_extension == 'mp4':
            # Video upload
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(uploaded_file.read())
            output_path = os.path.join(OUTPUT_DIR, "predict_video")
            video_processor = VideoProcessor(temp_file.name, output_path)

            # Display each frame of the processed video
            for frame in video_processor.process_video():
                st.image(frame, caption='Processed Frame', use_column_width=True)

    else:
        # Webcam feed
        st.write("Web Camera Feed")
        output_path = os.path.join(OUTPUT_DIR, "predict_webcamera")
        detector = AppleDetectionProcessor(output_path)
        webrtc_streamer(key="example", video_processor_factory=detector, rtc_configuration=RTCConfiguration(iceServers=[{"urls": ["stun:stun.l.google.com:19302"]}]))


# About page function
def about_page():
    st.header("About Page")
    st.write("""It looks like you have a well-structured application for 
             apple leaf disease detection using YOLO and Streamlit for the web-based interface. 
             Let’s go through the code and identify some aspects and theories behind the coding choices made in your application:""")
    
    st.header("Modularity and Separation of Concerns")
    st.write("""*** Your code demonstrates a clear separation of concerns with different functions and classes
              handling specific tasks such as processing images, videos, and webcam feeds separately.""")
    st.write("""*** Functions like predict_image(), process_video(), and classes like AppleDetectionProcessor 
             and VideoProcessor each focus on one specific task,which makes your code more modular and easier to maintain.""")
    st.header("Use of Libraries:")
    st.write("*** Libraries like Streamlit, OpenCV (cv2), PIL, and Plotly help with interfacing, image and video manipulation, and data visualization.")
    st.write("*** The ultralytics library is used for the YOLO model, which is designed for object detection and makes use of pre-trained models to perform detection.")
    st.header("Handling Different File Types:")
    st.write("*** The code appropriately handles different file types (image and video) using conditionals based on the file extension (jpg, jpeg, png, and mp4). This allows the application to process different file types correctly.")
    st.write("*** By using a file_uploader, users can upload images or videos directly from the interface.")
    st.header("Real-time Processing with WebRTC:")
    st.write("*** The application supports real-time processing of webcam feeds using webrtc_streamer, enabling live detection from the user's webcam.")
    st.write("*** The AppleDetectionProcessor class integrates the YOLO model to perform real-time detection.")
    st.header("Code for Data Management")
    st.write("*** The application uses specific directories (OUTPUT_DIR) to manage the outputs (predicted images, videos) from the model. This organization helps in structuring data.")
    st.write("*** Using Python's os module to manage file paths and directories keeps the code flexible and platform-independent.")
    st.header("Timestamping and Result Display:")
    st.write("*** Adding timestamps to frames can be useful for tracking when images or frames were processed.")
    st.write("*** Detected results are displayed to the user within the application using Streamlit, which provides a friendly and interactive interface.")
    st.header("Saving Detected Images")
    st.write("*** The code saves detected images and video frames to specified directories, ensuring persistent storage of the model's outputs.")
    st.header("UI and Navigation")
    st.write("*** Streamlit’s sidebar is used for navigation, allowing users to switch between different pages (Home, Detect, About).")

    st.header("Visualization")
    # Create a list of image paths
    image_paths = [
        "results.png",
        "R_curve.png",
        "confusion_matrix_normalized.png",
        "F1_curve.png",
        "labels_correlogram.jpg",
        "labels.jpg",
        "newplot (1).png",
        "newplot (2).png",
        "P_curve.png",
        "PR_curve.png",
        "train_batch1522.jpg",
        "val_batch2_pred.jpg",
    ]

    # Display images in a vertical two-by-two format
    for i in range(0, len(image_paths), 2):
        # Create columns for each pair of images
        col1, col2 = st.columns(2)
        
        # Display the first image in the left column
        with col1:
            st.image(image_paths[i], caption='Image', use_column_width=True)
        
        # Display the second image in the right column if it exists
        if i + 1 < len(image_paths):
            with col2:
                st.image(image_paths[i + 1], caption='Image', use_column_width=True)

 


# Main function for navigation and page selection
def app():
    st.sidebar.title("Detection")
    page = st.sidebar.selectbox("Choose a page", ["Home", "Detect", "About"])

    if page == "Home":
        home_page()
    elif page == "Detect":
        detection()
    elif page == "About":
        about_page()

if __name__ == "__main__":
    app()
