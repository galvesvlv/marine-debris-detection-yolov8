# imports
import os
import requests
import streamlit as st

# Page config
st.set_page_config(
                   page_title="Marine Debris Detection (YOLOv8)",
                   page_icon="🌊",
                   layout="centered"
                   )

# Title and Description
st.markdown(
            "<h1 style='text-align: center;'>🌊 Marine Debris Detection with YOLOv8</h1>",
            unsafe_allow_html=True
            )

st.markdown(

"""
Upload an **image** or **video** and run object detection for marine debris.
The annotated output will be returned directly by the API.

- Images are displayed on screen and can be downloaded.
- Videos are processed frame-by-frame and returned as a downloadable file.
"""

)

# Sidebar
st.sidebar.header("🚀 Instructions")
st.sidebar.markdown(

"""
1. Choose **Image** or **Video**.
2. Upload a supported file.
3. Wait for the API to process the input.
4. View or download the annotated result.
"""

)

st.sidebar.markdown("---")
st.sidebar.markdown("**Supported formats**")
st.sidebar.markdown(

"""
- Images: JPG, PNG  
- Videos: MP4, AVI, MOV, MKV
"""

)

# API Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Input type selector
input_type = st.radio(
                      "Select input type:",
                      options=["Image", "Video"],
                      horizontal=True
                      )

# Image inference
if input_type == "Image":
    uploaded_file = st.file_uploader(
                                     "Upload an image",
                                     type=["jpg", "jpeg", "png"]
                                     )

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Original image", width=700)

        if st.button("Run inference"):
            with st.spinner("Running YOLOv8 inference on image..."):
                response = requests.post(
                                         f"{API_URL}/predict/image",
                                         files={"file": uploaded_file},
                                         )

            if response.status_code != 200:
                st.error(f"API error: {response.text}")
            else:
                st.success("Inference completed.")

                annotated_image = response.content

                st.image(
                         annotated_image,
                         caption="Annotated image",
                         width=700
                         )

                st.download_button(
                                   label="Download annotated image",
                                   data=annotated_image,
                                   file_name="annotated_image.jpg",
                                   mime="image/jpeg"
                                   )

# Video inference
elif input_type == "Video":
    uploaded_file = st.file_uploader(
                                     "Upload a video",
                                     type=["mp4", "avi", "mov", "mkv"]
                                     )

    if uploaded_file is not None:
        st.info("Video uploaded. Click below to start processing.")

        if st.button("Run inference"):
            with st.spinner(
                            "Processing video... This may take some time depending on its length."
                            ):
                response = requests.post(
                                         f"{API_URL}/predict/video",
                                         files={"file": uploaded_file},
                                         )

            if response.status_code != 200:
                st.error(f"API error: {response.text}")
            else:
                st.success("Video processing completed.")

                annotated_video = response.content

                st.download_button(
                                   label="Download annotated video",
                                   data=annotated_video,
                                   file_name="annotated_video.mp4",
                                   mime="video/mp4"
                                   )