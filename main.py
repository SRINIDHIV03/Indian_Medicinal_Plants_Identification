import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import base64

model = load_model('Model/model.h5',compile=False)
class_dict = np.load("artifacts/class_names.npy")


def predict(image):
    IMG_SIZE = (1, 224, 224, 3)

    img = image.resize(IMG_SIZE[1:-1])
    img_arr = np.array(img)
    img_arr = img_arr.reshape(IMG_SIZE)

    pred_proba = model.predict(img_arr)
    pred = np.argmax(pred_proba)
    return pred, class_dict[pred], pred_proba[0][pred]  # Return prediction, class name, and probability

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

custom_style = """
    <style>
        body {
            font-family: 'Roboto', sans-serif;
        }
        .title-to-right {
             text-align: center;
             font-weight:bold;
             font-family:'Rethink Sans';
        }
        
        .predict-para {
        
            color:green;
            text-align:center;
        }
        
        .predict-button, .know-more-button {
            text-align: center;
            margin-top: 10px;
        }
        
        .center-image {
            display: block;
            margin-left: auto;
            margin-right: auto;
        }
        
    </style>
"""

contnt = "<p class='content-to-right' style='color: black;'>Herbal medicines are preferred in both developing and developed countries as an alternative to " \
         "synthetic drugs mainly because of no side effects. Recognition of these plants by human sight will be " \
         "tedious, time-consuming, and inaccurate.</p> " \
         "<p style='color: black;'>Applications of image processing and computer vision " \
         "techniques for the identification of the medicinal plants are very crucial as many of them are under " \
         "extinction as per the IUCN records. Hence, the digitization of useful medicinal plants is crucial " \
         "for the conservation of biodiversity.</p>"

if __name__ == '__main__':
    add_bg_from_local("artifacts/Leaf2.jpg")
    new_title = '<p class="title-to-right" style="font-family:sans-serif; color:black; font-size: 42px;">Medicinal Leaf Classification</p>'
    st.markdown(custom_style, unsafe_allow_html=True)
    st.markdown(new_title, unsafe_allow_html=True)
    st.markdown(contnt, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is not None:
        img = Image.open(uploaded_file)

        img = img.resize((300, 300))
        st.image(img)

        if st.button("Predict"):
            pred, name, prob = predict(img)  # Get prediction, class name, and probability

            result = f'<p class="predict-para"style="font-family:sans-serif; color:Black; font-size: 16px;">The given image is {name}</p>'
            st.markdown(result, unsafe_allow_html=True)

            # Add "Know More" button with hyperlink
            name1=name+" leaf meadicinal uses"
            know_more_text = f'<a href="https://www.google.com/search?q={name1}" target="_blank">Know More</a>'  # Google search on new tab
            st.markdown(know_more_text, unsafe_allow_html=True)
