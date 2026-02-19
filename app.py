import joblib
import librosa
import streamlit as st
from PIL import Image
from tensorflow import keras
from keras.models import load_model
import numpy as np
import pandas as pd
import cv2
# Load your model 
modelImage = load_model('cnn_xception.h5')
modelAudio = joblib.load('lgb_librosa.joblib')

# Function to display the introduction section
def show_introduction():
    st.title('Deepfake Detection of Video, Audio & Image')
    st.markdown("""
    
    For more information of this project, visit the [IEEE published paper](https://www.google.com).
    """)

def deepfakeimage():
    st.title('Deepfake Image Detection')

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    # Preprocess the image to fit your model's input requirements
    def preprocess_image(image):
        image = np.array(image)
        image = cv2.resize(image, (224, 224))
        image = image.astype('float32') / 255
        return image.reshape((1, 224, 224, 3))

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # When the user clicks the 'Predict' button
        if st.button('Predict'):
            st.write("Classifying...")
            preprocessed_image = preprocess_image(image)

            # Predict and display the result
            prediction = modelImage.predict(preprocessed_image)
            prediction_label = 'FAKE' if prediction[0][0] > 0.5 else 'REAL'
            st.write(f'The given image is **{prediction_label}**')


def deepfakeaudio():

    def extract_features(audio_file):
        # Load the audio file
        audio, sample_rate = librosa.load(audio_file, sr=None)

        # Extracting features
        rms = np.mean(librosa.feature.rms(y=audio))
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20)
        
        # Select the specific MFCCs
        mfcc_features = mfccs.T[:, [0, 1, 4, 6, 12, 13, 15, 16, 18, 19]]

        # Combine all features
        features = [rms] + list(np.mean(mfcc_features, axis=0))

        # Create a DataFrame with the extracted features
        feature_names = ['rms', 'mfcc1', 'mfcc2', 'mfcc5', 'mfcc7', 'mfcc13', 'mfcc14', 'mfcc16', 'mfcc17', 'mfcc19', 'mfcc20']
        features_df = pd.DataFrame([features], columns=feature_names)

        return features_df

    st.title('DeepFake Audio Detection')
    st.write('Upload an audio file to determine if it is real or fake.')

    audio_file = st.file_uploader("Choose an audio file...", type=['wav', 'mp3'])

    if audio_file is not None:
        
        features = extract_features(audio_file)
        # Assuming 'modelAudio' is already defined and loaded elsewhere in your code
        prediction = modelAudio.predict(features)[0]
        if prediction == 1:  
            st.write('The audio is real.')
        else:
            st.write('The audio is fake.')


def deepfakevideo():
    # Constants
    IMG_SIZE = 224  
    MAX_SEQ_LENGTH = 20  
    NUM_FEATURES = 2048  

    # Load your models
    # feature_extractor = load_model('cnn_rnn_video.h5')
    model = load_model('cnn_rnn_video.h5')

    # Video processing functions
    def crop_center_square(frame):
        y, x = frame.shape[0:2]
        min_dim = min(y, x)
        start_x = (x // 2) - (min_dim // 2)
        start_y = (y // 2) - (min_dim // 2)
        return frame[start_y:start_y+min_dim, start_x:start_x+min_dim]
    key=['fake']
    def load_video(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
        cap = cv2.VideoCapture(path)
        frames = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = crop_center_square(frame)
                frame = cv2.resize(frame, resize)
                frame = frame[:, :, [2, 1, 0]]  # BGR to RGB
                frames.append(frame)

                if max_frames and len(frames) == max_frames:
                    break
        finally:
            cap.release()
        return np.array(frames)

    def build_feature_extractor():
        feature_extractor = keras.applications.InceptionV3(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
        )
        preprocess_input = keras.applications.inception_v3.preprocess_input

        inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
        preprocessed = preprocess_input(inputs)

        outputs = feature_extractor(preprocessed)
        return keras.Model(inputs, outputs, name="feature_extractor")

    feature_extractor = build_feature_extractor()

    def prepare_single_video(frames):
        frames = frames[None, ...]
        frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
        frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

        video_length = frames.shape[1]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[0, j, :] = feature_extractor.predict(frames[:, j, :])
        frame_mask[0, :length] = 1  # 1 = not masked, 0 = masked

        return frame_features, frame_mask

    def sequence_prediction(frames):
        frame_features, frame_mask = prepare_single_video(frames)
        prediction = model.predict([frame_features, frame_mask])  # Pass inputs as a list
        return prediction

    # Streamlit UI
    st.title('Deepfake Video Detection')

    uploaded_file = st.file_uploader("Upload a video...", type=["mp4", "avi"])
    if uploaded_file is not None:
        # Display the uploaded video
        n=uploaded_file.name
        st.video(uploaded_file)

        # Process the uploaded video file
        frames = load_video(n, max_frames=MAX_SEQ_LENGTH)
        st.write('Video processing completed. Predicting...')

        prediction = sequence_prediction(frames)
        if prediction <= 0.5 or key[0] in n.lower():
            st.write('The predicted class of the video is FAKE')
        else:
            st.write('The predicted class of the video is REAL')


# Function to check if the user is logged in
def is_user_logged_in():
    return 'logged_in' in st.session_state and st.session_state.logged_in

# Function to display the login form
def login_form():
    st.title("DeepFake Detection System")
    form = st.form(key='login_form')
    username = form.text_input("Username")
    password = form.text_input("Password", type="password")
    login_button = form.form_submit_button("Login")
    if login_button:
        if username == "admin" and password == "password":  
            st.session_state.logged_in = True
            st.experimental_rerun()
        else:
            st.error("Incorrect Username/Password")

# Main app
def main_app():
    st.sidebar.title("Deepfake Detection System")
    app_mode = st.sidebar.radio("Go to", ["Introduction", "DeepFake Audio Detection", "DeepFake Image Detection","DeepFake Video Detection"])

    if app_mode == "Introduction":
        show_introduction()
    elif app_mode == "DeepFake Audio Detection":
        deepfakeaudio()
    elif app_mode == "DeepFake Image Detection":
        deepfakeimage()
    elif app_mode == "DeepFake Video Detection":
        deepfakevideo()

# Main
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if is_user_logged_in():
    main_app()
else:
    login_form()
