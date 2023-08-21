# Microplastic-detection by using YOLO

## Object detection model 

This github is contain model after train of paper `A detection and classification of microplastics based on YOLOv8 and YOLO-NAS` . 

<br />

## Using in colab

You can use example code for inferance model at `example.ipynb` in this github. <br />

You can acess and our model in this github .
```
weights/microplastic-detection-yolo8m.pt
```


## Web application. 
You can acess our example web application in below link. 
```
https://microplastic.onrender.com/
```

## Run with streamlit in your local computer
**Step1**
Clone this repository. 
```
git clone https://github.com/arsanchai-su/microplastic-detection-app.git
```
**Step2**
Change working directory 
```
cd microplastic-detection-app
```
**Step3**
 Install packages with pip. 
```
pip install -r requirements.txt
```
**Step4**
Run  streamlit by using command. 
```
streamlit run streamlit_app.py 
```
**Step5**
Access you localhost.
```
http://localhost:8501/
```

