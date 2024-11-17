import streamlit as st
import torch
import cv2
import tempfile
import numpy as np
import os
import time

# Define the model path
model_path = './custom_model/weights/best.pt'  # Ensure the path to your YOLO model is correct

# Attempt to load the custom YOLOv5 model
try:
    model = torch.hub.load('.', 'custom', path=model_path, source='local')
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    raise e  # Raise the exception to highlight critical issues

# Sidebar for user guidance
st.sidebar.title("🎾 Tennis Player Tracking Guide")
st.sidebar.markdown(
    """
    Welcome to the *Tennis Tracking App*! Analyze and track players in your tennis videos with ease.

    ### How to Use:
    1. *Upload Your Video*  
       - Supported formats: MP4, AVI, MOV.
       - Choose a video from your device to get started.
    
    2. *Processing*  
       - The app will detect and track players. This may take some time based on video size.
    
    3. *Download the Result*  
       - Once processed, you can download the video with player tracking.

    ### Notes:
    - Ensure best.pt is in the correct directory.
    - For support, refer to the documentation or reach out to us.

    *Enjoy analyzing your game!*
    """
)

# Main app interface
st.title("🎾 Tennis Tracking Application")
st.write("Upload your tennis video to detect and track players in real-time.")

# File uploader for video input
uploaded_video = st.file_uploader("Upload your video file:", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    # Save uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
        temp_video.write(uploaded_video.read())
        temp_video_path = temp_video.name

    # Open video capture
    cap = cv2.VideoCapture(temp_video_path)
    stframe = st.empty()

    # Set up the output video
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as output_temp:
        output_video_path = output_temp.name

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Progress bar setup
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)
    frame_count = 0

    st.write("🔄 Processing video, please wait...")

    # Process video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection model
        if torch.cuda.is_available():
            with torch.amp.autocast(device_type='cuda'):
                results = model(frame)
        else:
            results = model(frame)

        frame = np.squeeze(results.render())  # Render detection boxes on the frame

        # Write processed frame to output file
        out.write(frame)

        # Convert BGR to RGB for Streamlit display
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Show the processed frame in Streamlit
        stframe.image(frame, channels='RGB', use_container_width=True)

        # Update progress bar
        frame_count += 1
        progress_bar.progress(frame_count / total_frames)

        # Ensure consistent playback
        time.sleep(1 / fps)

    # Release resources
    cap.release()
    out.release()

    st.success("✅ Video processing completed!")

    # Provide download button
    st.write("📥 Download your processed video:")
    with open(output_video_path, 'rb') as f:
        st.download_button(
            label="Download Processed Video",
            data=f,
            file_name="processed_video.mp4",
            mime="video/mp4"
        )

    # Clean up temporary files
    os.remove(temp_video_path)
    os.remove(output_video_path)