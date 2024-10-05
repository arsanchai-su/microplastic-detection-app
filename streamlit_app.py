#Import required libraries
import PIL

import streamlit as st
from ultralytics import YOLO
import time  # Add the time module


# Replace the relative path to your weight file
model_path = 'weights/t29.pt' # Change to T29 weigth

# Setting page layout
st.set_page_config(
    page_title="Microplastic Detection using YOLOv8",  # Setting page title
    page_icon="📃",     # Setting page icon
    layout="wide",      # Setting layout to wide
    initial_sidebar_state="expanded"    # Expanding sidebar by default
)

# Creating sidebar
with st.sidebar:
    st.header("Upload Image")     # Adding header to sidebar
    # Adding file uploader to sidebar for selecting images
    source_img = st.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    # Model Options
    confidence = float(st.slider(
        "Select Model Confidence", 25, 100, 40)) / 100

# Creating main page heading
st.title("Microplastic Detection using YOLOv8 ")
st.write("Welcome to the Microplastic Detection Application!")
st.write("This web application allows you to detect and classify microplastics using the model described in the paper:")
st.write("'A Detection and Classification of Microplastics Based on YOLOv8 and YOLO-NAS'.")
st.markdown("Visit GitHub:<a href='https://github.com/arsanchai-su/microplastic-detection' target='_blank'>GitHub Repository </a>", unsafe_allow_html=True)


# Creating two columns on the main page
col1, col2 = st.columns(2)

# Adding image to the first column if image is uploaded
with col1:
    if source_img:
        # Opening the uploaded image
        uploaded_image = PIL.Image.open(source_img)
        # Adding the uploaded image to the page with a caption
        st.image(source_img,
                 caption="Uploaded Image",
                 use_column_width=True
                 )

try:
    model = YOLO(model_path)
except Exception as ex:
    st.error(
        f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

if st.sidebar.button('Detect Objects'):
    # Capture start time
    start_time = time.time()
    
    res = model.predict(uploaded_image,
                        conf=confidence
                        )
    
    # Capture end time and calculate prediction time
    end_time = time.time()
    prediction_time = end_time - start_time
    
    boxes = res[0].boxes
    class_idx = res[0].boxes.cls.cpu().numpy().astype(int)
    
    label_names = {
    0: 'Fibers',
    1: 'Films',
    2: 'Fragments',
    3: 'Pallets'
    }
    label_counts = {}

    for label in class_idx:
        label_name = label_names.get(label, 'unknown')  # Get the label name from the dictionary
        if label_name in label_counts:
            label_counts[label_name] += 1
        else:
            label_counts[label_name] = 1
    
    res_plotted = res[0].plot()[:, :, ::-1]
    with col2:
        st.image(res_plotted,
                 caption='Detected Image',
                 use_column_width=True
                 )
        st.write("<span style='font-size: 20px;'>Type of microplastic: Q'ty</span>", unsafe_allow_html=True)
        for idx, (label, count) in enumerate(label_counts.items()):
            color = 'green' if idx % 2 == 0 else 'blue'
            st.markdown(f"<span style='color: {color}; font-size: 20px;'>{label}: {count}</span>", unsafe_allow_html=True)

        # Display the prediction time
        st.write(f"Prediction Time: {prediction_time:.2f} seconds")
            
        try:
            with st.expander("Detection Results"):
                for box in boxes:
                    st.write(box.xywh)
                    
                    
        except Exception as ex:
            st.write("No image is uploaded yet!")