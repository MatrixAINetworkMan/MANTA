import os
import streamlit as st
from PIL import Image
from detector import detect_class, detect_stereo

st.set_option('deprecation.showfileUploaderEncoding', False)

# Constants for sidebar dropdown
SIDEBAR_OPTION_PROJECT_INFO = "Show Project Info"
SIDEBAR_OPTION_IMAGE_CLASSIFICATION = "Image Classification"
SIDEBAR_OPTION_STEREO_MATCHING = "Depth Estimation"
SIDEBAR_OPTIONS = [SIDEBAR_OPTION_PROJECT_INFO, SIDEBAR_OPTION_IMAGE_CLASSIFICATION, SIDEBAR_OPTION_STEREO_MATCHING]

st.sidebar.title("Menu")
CKPT_ROOT = 'checkpoints'
app_mode = st.sidebar.selectbox("Please select from the following", SIDEBAR_OPTIONS)
CLASS_MODELS = ['OFA_MBV3']
class_checkpoints = ['checkpoints/demo_class']
DEPTH_MODELS = ['OFA_AANet']
depth_checkpoints = ['checkpoints/final']

def update_class_ckpts():

    ckpts = os.listdir(CKPT_ROOT)
    ckpts = [ckpt for ckpt in ckpts if 'class' in ckpt]
    print(ckpts)
    return ckpts

def update_stereo_ckpts():

    ckpts = os.listdir(CKPT_ROOT)
    ckpts = [ckpt for ckpt in ckpts if 'stereo' in ckpt]
    print(ckpts)
    return ckpts

if app_mode == SIDEBAR_OPTION_PROJECT_INFO:

    st.sidebar.write(" ------ ")
    st.sidebar.success("Project information showing on the right!")
    with open("Project_Info.md", "r") as f:
        contents = f.read()
        st.write(contents)

elif app_mode == SIDEBAR_OPTION_IMAGE_CLASSIFICATION:
    
    class_checkpoints = update_class_ckpts()

    st.sidebar.write(" ------ ")
    st.sidebar.title("Model")
    models = st.sidebar.selectbox("Please select from the following", CLASS_MODELS)
    st.sidebar.title("Weights")
    ckpt = st.sidebar.selectbox("Please select from the following", class_checkpoints)
    ckpt = os.path.join(CKPT_ROOT, ckpt)
    st.title("Image Classification by AutoML")
    st.write("")

    file_up = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if file_up is not None:
        image = Image.open(file_up)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Just a second...")
        labels = detect_class(file_up, model=models, ckpt=ckpt)
    
        # print out the top 5 prediction labels with scores
        for i in labels:
            #st.write("Prediction (index, name)", i[0], ",   Score: ", i[1])
            st.write(i[0], ",   Score: %.2f" % i[1])

elif app_mode == SIDEBAR_OPTION_STEREO_MATCHING:

    depth_checkpoints = update_stereo_ckpts()

    st.sidebar.write(" ------ ")
    st.sidebar.title("Model")
    models = st.sidebar.selectbox("Please select from the following", DEPTH_MODELS)
    st.sidebar.title("Weights")
    ckpt = st.sidebar.selectbox("Please select from the following", depth_checkpoints)
    ckpt = os.path.join(CKPT_ROOT, ckpt)
    st.title("Depth Estimation by AutoML")
    st.write("")

    file_left = st.sidebar.file_uploader("Upload the left image", type=["jpg", "jpeg", "png"])
    file_right = st.sidebar.file_uploader("Upload the right image", type=["jpg", "jpeg", "png"])

    left_column, right_column = st.columns(2)
    if file_left is not None:
        left_image = Image.open(file_left).convert('RGB')
        left_column.image(left_image, caption = "Left Image")

    if file_right is not None:
        right_image = Image.open(file_right).convert('RGB')
        right_column.image(right_image, caption = "Right Image")

    pressed = st.sidebar.button('RUN!')
    if pressed:
        st.empty()
        st.write('Please wait for the magic to happen! This may take up to a minute.')
        st.image(detect_stereo(left_image, right_image, ckpt=ckpt), caption = "Predicted Left Depth")
