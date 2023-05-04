import streamlit as st
import operator
import cv2
from keras.models import model_from_json
import numpy as np
from streamlit_option_menu import option_menu

from PIL import Image

# Open image file
accuracy = Image.open('acc.png')
loss = Image.open('los.png')
confusion_matrix = Image.open('confusion_matrix.png')

# 1. as sidebar menu
with st.sidebar:
    selected2 = option_menu("Main Menu", ["Home", "recognize/start", 'accuracy graph', 'loss graph', 'confusion matrix'],
                            icons=['house', "camera", 'diagram-3', 'diagram-3', 'gear'], menu_icon="cast", default_index=0,
                            styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"},
        "nav-link": {"font-size": "25px", "text-align": "left", "margin": "0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "green"},
    }
    )

if selected2 == "Home":
    st.title("Welcome to sign language recognition system")
if selected2 == "recognize/start":
    st.title("Sign language recognition")

    # Load the model from disk
    json_file = open("model5.json", "r")
    model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(model_json)
    loaded_model.load_weights("model5.h5")
    st.write("Place your hand inside the rectangle")

    # Start the webcam and set the region of interest
    cap = cv2.VideoCapture(0)
    x1, y1, x2, y2 = 400, 100, 620, 320

    # Create a streamlit placeholder to display the video stream
    frame_placeholder = st.empty()

    # Create a streamlit placeholder to display the prediction
    prediction_placeholder = st.empty()

    while True:
        # Read a frame from the video stream and flip it horizontally
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        # Draw a rectangle around the region of interest and extract it
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        roi = frame[y1:y2, x1:x2]

        # Preprocess the extracted region for prediction
        roi = cv2.resize(roi, (128, 128))
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi = cv2.GaussianBlur(roi, (7, 7), 0)
        _, roi = cv2.threshold(
            roi, 10, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)

        # Predict the sign language gesture and display the result
        result = loaded_model.predict(roi.reshape(1, 128, 128, 3))
        prediction = {'Zero': result[0][0],
                      'One': result[0][1],
                      'Two': result[0][2],
                      'Three': result[0][3],
                      'Four': result[0][4],
                      'Five': result[0][5],
                      'A': result[0][6],
                      'B': result[0][7],
                      'C': result[0][8],
                      'D': result[0][9],
                      'E': result[0][10],
                      'F': result[0][11],
                      'G': result[0][12],
                      'H': result[0][13],
                      'I': result[0][14],
                      'J': result[0][15],
                      'Null': result[0][16]
                      }
        prediction = sorted(prediction.items(),
                            key=operator.itemgetter(1), reverse=True)
        prediction_text = prediction[0][0]
        prediction_placeholder.write(f"Predicted Gesture: {prediction_text}")

        # Display the video stream and prediction using streamlit
        frame_placeholder.image(frame, channels="BGR")

        # Wait for the user to press the Esc key to exit the app
        interrupt = cv2.waitKey(10)
        if interrupt & 0xFF == 27:
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()


if selected2 == "accuracy graph":
    st.image(accuracy, caption='Accuracy graph image')

if selected2 == "loss graph":
    st.image(loss, caption='Loss graph image')

if selected2 == "confusion matrix":
    st.image(confusion_matrix, caption='Confusion matrix image')

