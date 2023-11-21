import streamlit as st
from fastai.vision.all import Path,load_learner
import pandas as pd
import PIL.Image
import numpy as np
import pandas as pd
import cv2
from mtcnn.mtcnn import MTCNN

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

st.set_page_config(page_title='Mood Music AI App')
# favicon being an object of the same kind as the one you should provide st.image() with (ie. a PIL array for example) or a string (url or local file path)

# load model
path = Path('export.pkl')
learn = load_learner(path)
detector = MTCNN()

lables= ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

df = pd.read_csv("data_moods.csv")

# Create a dictionary to map prediction labels to moods
prediction_to_mood = {
    'anger': 'Calm',
    'contempt': 'Calm',
    'disgust': 'Energetic',
    'fear': 'Sad',  # You can map 'fear' to 'Sad' based on your dataset
    'happy': 'Happy',
    'neutral': 'Happy',  # You can create a mapping for 'neutral' if needed
    'sad': 'Sad',
    'surprise': 'Happy'  # You can map 'surprise' to 'Happy' based on your dataset
}

# Define the function to detect faces and emotions (you can reuse your existing function)
def detect_faces_emo(pil_img, prediction, detection_confidence=0.99, min_face_size=10):
    # Load the image using OpenCV
    image = np.asarray(pil_img)#cv2.imread(image_path)

    # Create an MTCNN detector instance
    detector = MTCNN()

    # Convert the prediction tensor to a list of probabilities
    predicted_emo = prediction[-1].tolist()

    # Loop over the detected faces
    for face in detector.detect_faces(image):
        # Check the confidence score of the detection
        if face['confidence'] < detection_confidence:
            continue
        # Extract the bounding box coordinates
        x, y, width, height = face['box']
        # Check the size of the bounding box
        if min(width, height) < min_face_size:
            continue
        # Extract the face region from the image
        face_image = image[y:y+height, x:x+width]
        # Resize the face image to 96x96
        face_image_resized = cv2.resize(face_image, (96, 96))
        # Reshape the face image to match the input shape of the model
        face_image_reshaped = face_image_resized.reshape((1, 96, 96, 3))
        
        # Draw the predicted emotion label on the rectangle around the face
        predicted_emo_sorted = sorted(list(enumerate(predicted_emo)), key=lambda x: x[1], reverse=True)
        label = lables[predicted_emo_sorted[0][0]].title()  # Get the label for the highest predicted emotion
        cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX,5, (0, 255, 0), 10)
        
        # Draw a square rectangle around the face
        face_size = min(width, height)
        x_center = x + int(width / 2)
        y_center = y + int(height / 2)
        x1 = x_center - int(face_size / 2)
        y1 = y_center - int(face_size / 2)
        x2 = x_center + int(face_size / 2)
        y2 = y_center + int(face_size / 2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (127, 255, 0), 10)

    # Save the image with the detected faces and predicted emotions to a file
    cv2.imwrite("detected_faces.jpg",cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    # Return the path to the saved file
    return "detected_faces.jpg"

def recommend_song_by_facial_mood(pil_img):
    img_np = np.asarray(pil_img)
    pred = learn.predict(img_np)
    image_path = detect_faces_emo(pil_img,pred)
    temp_df = df[df['mood']==prediction_to_mood[pred[0]]]
    temp_df = temp_df[['name','artist','mood','popularity']].sort_values(by='popularity',ascending=False).reset_index(drop=True)
    return image_path, temp_df.head(5),pred[0]

st.image("moodmusic.png", use_column_width=True,)
html_temp = """
    <div style="background-color:#5247fe;padding:6px;margin-bottom: 15px">
    <h4 style="color:white;text-align:center;">MUSIC RECOMMENDATION SYSTEM BASED FACIAL EXPRESSION</h4>
    <p style="color:white;text-align:center;" >Discover the perfect tunes for your every emotion in seconds. Simply let our app read your facial expressions, and voilà – your ideal playlist, all under 25 seconds!</p>
    </div>
    """
st.markdown(html_temp,unsafe_allow_html=True)

# Option to select camera or upload an image
input_option = st.radio("Select Input Option:", ("Upload Image","Camera"))

if input_option == "Camera":
    # Camera input
    camera_image = st.camera_input(label="Capture Your Image")
    if camera_image is not None:
        # Display the captured image
        st.image(camera_image, caption="Captured Image", use_column_width=True)
        
        # Detect faces and emotions
        pil_img = PIL.Image.open(camera_image)
        img_np_c = np.asarray(pil_img)
        detector = MTCNN()
        faces = detector.detect_faces(img_np_c)

        if len(faces) == 0:
            st.warning("No faces detected in the captured image.")
        else:
            # Detect faces and emotions and recommend a song
            image_path, reco_song,pred = recommend_song_by_facial_mood(pil_img)

            # Display the predicted image with mood annotations
            st.image(image_path, caption="Predicted Image", use_column_width=True)

            html_temp = f"""
                    <div style="background-color:#5247fe;padding:5px;margin-bottom: 15px">
                    <p style="color:white;text-align:center;">Detected Emotion</p>
                    <h3 style="color:white;text-align:center;">{pred.upper()}</h3>
                    </div>
                    """
            st.markdown(html_temp,unsafe_allow_html=True)

            html_temp = """
                    <div style="background-color:purple;padding:0px;margin-bottom: 1px">
                    <p style="color:white;text-align:center;">Based On Your Emotion Top songs Picked For You</p>
                    </div>
                    """
            st.markdown(html_temp,unsafe_allow_html=True)
            # Display recommended song information
            st.table(reco_song)
else:
    # Upload an image
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_image is not None:
        # Detect faces and emotions
        pil_img = PIL.Image.open(uploaded_image)
        img_np_p = np.asarray(pil_img)
        faces = detector.detect_faces(img_np_p)

        if len(faces) == 0:
            st.warning("No faces detected in the captured image.")
        else:
            # Detect faces and emotions and recommend a song
            image_path, reco_song, pred = recommend_song_by_facial_mood(pil_img)

            # Display the predicted image with mood annotations
            st.image(image_path, caption="Predicted Image", use_column_width=True)

            html_temp = f"""
                    <div style="background-color:#5247fe;padding:5px;margin-bottom: 15px">
                    <p style="color:white;text-align:center;">Detected Emotion</p>
                    <h3 style="color:white;text-align:center;">{pred.upper()}</h3>
                    </div>
                    """
            st.markdown(html_temp,unsafe_allow_html=True)

            html_temp = """
                    <div style="background-color:purple;padding:0px;margin-bottom: 1px">
                    <p style="color:white;text-align:center;">Based On Your Emotion Top songs Picked For You</p>
                    </div>
                    """
            st.markdown(html_temp,unsafe_allow_html=True)
            st.table(reco_song)

html_temp1 = """
    <div style="background-color:#5247fe">
    <p style="color:white;text-align:center;" >Designe & Developed By: <b>Ankita Verma</b> </p>
    </div>
    """
st.markdown(html_temp1,unsafe_allow_html=True)
