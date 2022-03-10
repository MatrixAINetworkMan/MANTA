import streamlit as st
from PIL import Image
import time
from detector import detect_class, detect_stereo
import net_utils.settings as settings
from trainer import train_class, train_stereo, query_running, query_stopped, query_all, stop_job, check_job_name

st.set_option('deprecation.showfileUploaderEncoding', False)

# Constants for sidebar dropdown
SIDEBAR_OPTION_PROJECT_INFO = "Show Project Info"
SIDEBAR_OPTION_IMAGE_CLASSIFICATION = "Image Classification"
SIDEBAR_OPTION_STEREO_MATCHING = "Depth Estimation"
SIDEBAR_OPTIONS = [SIDEBAR_OPTION_PROJECT_INFO, SIDEBAR_OPTION_IMAGE_CLASSIFICATION, SIDEBAR_OPTION_STEREO_MATCHING]

st.sidebar.title("Menu")
app_mode = st.sidebar.selectbox("Please select from the following", SIDEBAR_OPTIONS)
CLASS_MODELS = ['OFA_MBV3']
CLASS_DATASETS = ['ImageNet']
class_checkpoints = ['checkpoints/ofa_mbv3_d234_e346_k357_w1.0']
DEPTH_MODELS = ['OFA_AANet']
DEPTH_DATASETS = ['FlyingThings3D', 'Monkaa']
node_list = ['host%d' % (i+1) for i in range(2)]

#st.sidebar.write(" ------ ")

if app_mode == SIDEBAR_OPTION_PROJECT_INFO:

    #st.sidebar.success("Project information showing on the right!")
    with open("Project_Info.md", "r") as f:
        contents = f.read()
        st.write(contents)

if app_mode == SIDEBAR_OPTION_IMAGE_CLASSIFICATION or app_mode == SIDEBAR_OPTION_STEREO_MATCHING:
    st.sidebar.title("GPU Nodes")
    node_sts = []
    for node in node_list:
        node_sts.append(st.sidebar.checkbox(node))

    st.sidebar.title("Job Name")
    job_name = st.sidebar.text_input('Please type in the following', '')

if app_mode == SIDEBAR_OPTION_IMAGE_CLASSIFICATION:
    
    st.sidebar.title("Model")
    models = st.sidebar.selectbox("Please select from the following", CLASS_MODELS)
    st.sidebar.title("Datasets")
    datasets = st.sidebar.selectbox("Please select from the following", CLASS_DATASETS)

    # training parameters
    st.sidebar.title("Training Parameters")
    st.sidebar.write("Batch Size")
    bs_set = st.sidebar.radio("", [2, 4, 8, 16])
    st.sidebar.write("Learning Rate")
    lr_set = st.sidebar.radio("", [0.01, 0.02, 0.04, 0.08])

    # node list
    node_chosen = [node_list[i] for i in range(len(node_list)) if node_sts[i] == True]
    print(node_chosen)
    st.title("Image Classification by AutoML")

    st.write("Job List")
    #st.write(running_info)
    job_list, js_list = query_all()
    #st.write(stopped_info)
    # You can use a column just like st.sidebar:
    left_col, right_col = st.columns(2)
    job = left_col.selectbox("Please select from the following", [jn+" "+js for jn, js in zip(job_list, js_list)])

    pressed = right_col.button('Refresh')
    if pressed:
        pass

    pressed = st.sidebar.button('Run!')
    if pressed:
        st.empty()
        jcs = check_job_name(job_name)
        if jcs == 0 or jcs == 1: # empty name or running name
            st.warning('You have input an invalid job name. Try another.')
        else:
            if len(node_chosen) == 0:
                st.warning('Please choose at least one node for training.')
            else:
                st.warning('Please wait for the magic to happen! This may take up to a minute.')
                node_str = [n+":2" for n in node_chosen]
                node_str = ",".join(node_str)
                print(node_str)
                train_class(name=job_name, bs=bs_set, num_nodes=len(node_chosen), hosts=node_str, lr=lr_set)

    if job != None:
        j, js = job.split()
        pressed = right_col.button('Stop')
        if pressed:
            stop_job(j)

        logArea = st.empty()
        with open("logs/%s_class.log" % j, "r") as f:
            st.download_button(label="Download the complete log file", data=f)

        with open("logs/%s_class.log" % j, "r") as f:
            contents = f.readlines()[-20:]
            contents = [line.strip() for line in contents]
            logArea.text('\n'.join(contents))

elif app_mode == SIDEBAR_OPTION_STEREO_MATCHING:

    st.sidebar.title("Model")
    models = st.sidebar.selectbox("Please select from the following", DEPTH_MODELS)
    st.sidebar.title("Datasets")
    datasets = st.sidebar.selectbox("Please select from the following", DEPTH_DATASETS)

    # training parameters
    st.sidebar.title("Training Parameters")
    st.sidebar.write("Batch Size")
    bs_set = st.sidebar.radio("", [1, 2])
    st.sidebar.write("Learning Rate")
    lr_set = st.sidebar.radio("", [0.001, 0.002])

    # node list
    node_chosen = [node_list[i] for i in range(len(node_list)) if node_sts[i] == True]
    print(node_chosen)
    st.title("Depth Estimation by AutoML")

    st.write("Job List")
    #st.write(running_info)
    job_list, js_list = query_all(jtype='stereo')
    #st.write(stopped_info)
    # You can use a column just like st.sidebar:
    left_col, right_col = st.columns(2)
    job = left_col.selectbox("Please select from the following", [jn+" "+js for jn, js in zip(job_list, js_list)])

    pressed = right_col.button('Refresh')
    if pressed:
        pass

    pressed = st.sidebar.button('Run!')
    if pressed:
        st.empty()
        jcs = check_job_name(job_name, 'stereo')
        if jcs == 0 or jcs == 1: # empty name or running name
            st.warning('You have input an invalid job name. Try another.')
        else:
            if len(node_chosen) == 0:
                st.warning('Please choose at least one node for training.')
            else:
                st.warning('Please wait for the magic to happen! This may take up to a minute.')
                node_str = [n+":2" for n in node_chosen]
                node_str = ",".join(node_str)
                print(node_str)
                train_stereo(name=job_name, bs=bs_set, num_nodes=len(node_chosen), hosts=node_str, lr=lr_set)

    if job != None:
        j, js = job.split()
        pressed = right_col.button('Stop')
        if pressed:
            stop_job(j, 'stereo')

        logArea = st.empty()
        with open("logs/%s_stereo.log" % j, "r") as f:
            st.download_button(label="Download the complete log file", data=f)

        with open("logs/%s_stereo.log" % j, "r") as f:
            contents = f.readlines()[-20:]
            contents = [line.strip() for line in contents]
            logArea.text('\n'.join(contents))

