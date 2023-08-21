# Python In-built packages
from pathlib import Path
import PIL

# External packages
import streamlit as st

# Local Modules
import settings
import helper
from collections import Counter
import time

# Setting page layout
st.set_page_config(
    page_title="Object Detection Microplastic using YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Object Detection Microplastic using YOLOv8")

coil, coli1, coli2 = st.columns(3)
coil.metric("Precission", "99 %")
coli1.metric("Recall", "98 %")
coli2.metric("mPA", "95 %")

# Sidebar
st.sidebar.header("ML DETECTION")

# Model Options
model_type = st.sidebar.radio(
    "Select Task", ['Detection'])        #select  add Segmentation 

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40)) / 100

# Selecting Detection Or Segmentation
if model_type == 'Detection':
    model_path = Path(settings.DETECTION_MODEL)
#elif model_type == 'Segmentation':
    #model_path = Path(settings.SEGMENTATION_MODEL)

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Image Microplastic")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)

#def count_class_names(classes):
    #class_counter = Counter(classes)
    #return class_counter

source_img = None
# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)
    #class_counts = {} 

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
        #st.error(ex)
            
                        
    #from collections import Count ...
       
    with col2:
        
        if source_img is None:
            
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image',
                     use_column_width=True)
        
        else:
            st.sidebar.button('Detect Objects')
                       
            res = model.predict(uploaded_image,
                                    conf=confidence
                                    )
            
            boxes = res[0].boxes
            #boxes_np = boxes.xyxy.cpu().numpy()
            res_plotted = res[0].plot()[:, :, ::-1]
            st.image(res_plotted, caption='Detected Image',
                         use_column_width=True)
            
            
            class_counts = Counter([model.names[int(cls)] for cls in boxes.cls])
   
            try:
                
                 with st.expander("Detection Results"):
                     
                     #for box in boxes:
                         #st.write(box.cls)
                     
                    for class_name, count in class_counts.items():     #box in boxes:
                        st.write(f"{count} {class_name}s")                                             
                        #st.write(box.data)
            except Exception as ex:
                              
                 st.write("No image is uploaded yet!")
        
                     
                                     
st.header("Microplastics")   
st.markdown('Microplastics are tiny particles of plastic, typically measuring less than 5 millimeters in size. They are the result of the fragmentation or degradation of larger plastic items such as bottles, bags, packaging, and synthetic fibers. Microplastics can also be intentionally manufactured for use in various products like cosmetics, personal care items, and industrial applications.')




if st.button('Fiber'):
    st.markdown('Long Filaments with a constant diameter.')



if st.button('Film'):
    st.markdown('Thin continuous polymeric substance.')
   
    
    
if st.button('Fragment'):
    st.markdown('Thick particles with irregular dimensions.')
    
    
       
if st.button('Pellets'):
    st.markdown('Small beads of primary microplastic')
    


#elif source_radio == settings.VIDEO:
    #helper.play_stored_video(confidence, model)

#elif source_radio == settings.WEBCAM:
    #helper.play_webcam(confidence, model)

#elif source_radio == settings.RTSP:
    #helper.play_rtsp_stream(confidence, model)

#elif source_radio == settings.YOUTUBE:
    #helper.play_youtube_video(confidence, model)

#else:
    #st.error("Please select a valid source type!")
