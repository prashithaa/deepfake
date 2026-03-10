import os
import tempfile
import urllib.request
from pathlib import Path

import cv2
import joblib
import librosa
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from tensorflow import keras
from keras.models import load_model


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Deepfake Detection",
    page_icon="🛡️",
    layout="wide",
)

# =========================
# CUSTOM CSS
# =========================
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
    }

    .main-title {
        font-size: 3.2rem;
        font-weight: 800;
        color: #111827;
        margin-bottom: 0.3rem;
    }

    .sub-text {
        font-size: 1.1rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }

    .hero-box {
        background: rgba(255,255,255,0.82);
        border: 1px solid rgba(255,255,255,0.7);
        border-radius: 24px;
        padding: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
    }

    .info-card {
        background: white;
        border-radius: 20px;
        padding: 1.2rem;
        box-shadow: 0 8px 22px rgba(0,0,0,0.06);
        border: 1px solid #eef2f7;
        text-align: center;
        min-height: 140px;
    }

    .result-real {
        padding: 1rem;
        border-radius: 16px;
        background: #ecfdf5;
        border: 1px solid #86efac;
        color: #166534;
        font-weight: 700;
        font-size: 1.1rem;
    }

    .result-fake {
        padding: 1rem;
        border-radius: 16px;
        background: #fef2f2;
        border: 1px solid #fca5a5;
        color: #991b1b;
        font-weight: 700;
        font-size: 1.1rem;
    }

    .section-box {
        background: rgba(255,255,255,0.88);
        border-radius: 22px;
        padding: 1.4rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.06);
        border: 1px solid #edf2f7;
        margin-top: 1rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 12px;
        padding: 10px 18px;
        border: 1px solid #e5e7eb;
    }

    .stTabs [aria-selected="true"] {
        background: #111827 !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# MODEL CONFIG
# =========================
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

IMAGE_MODEL_PATH = MODEL_DIR / "cnn_xception.h5"
VIDEO_MODEL_PATH = MODEL_DIR / "cnn_rnn_video.h5"
AUDIO_MODEL_PATH = MODEL_DIR / "lgb_librosa.joblib"

DROPBOX_DIRECT_LINK_IMAGE = "https://www.dropbox.com/scl/fi/qj1bdyg9gumbzzw22mgp0/cnn_xception.h5?rlkey=65rmw61a1iorse30j33zr1h6s&st=x5ok4qlq&dl=1"
DROPBOX_DIRECT_LINK_VIDEO = "https://www.dropbox.com/scl/fi/icbqnswrhsuo7xipmo3cg/cnn_rnn_video.h5?rlkey=zyxp3s6r33nuavtpka2xump7c&st=csrrtge2&dl=1"
DROPBOX_DIRECT_LINK_AUDIO = "https://www.dropbox.com/scl/fi/xw3lbf3r2do9jf5e1fsr1/lgb_librosa.joblib?rlkey=7t9wks8jxbdzk7it1y3oeesp8&st=ubvnbaj7&dl=1"


# =========================
# HELPERS
# =========================
def download_file(url: str, output_path: Path):
    if not output_path.exists():
        with st.spinner(f"Downloading {output_path.name}..."):
            urllib.request.urlretrieve(url, output_path)


@st.cache_resource(show_spinner=False)
def build_feature_extractor():
    img_size = 224
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(img_size, img_size, 3),
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((img_size, img_size, 3))
    preprocessed = preprocess_input(inputs)
    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")


@st.cache_resource(show_spinner=False)
def load_all_models():
    download_file(DROPBOX_DIRECT_LINK_IMAGE, IMAGE_MODEL_PATH)
    download_file(DROPBOX_DIRECT_LINK_VIDEO, VIDEO_MODEL_PATH)
    download_file(DROPBOX_DIRECT_LINK_AUDIO, AUDIO_MODEL_PATH)

    image_model = load_model(IMAGE_MODEL_PATH)
    video_model = load_model(VIDEO_MODEL_PATH)
    audio_model = joblib.load(AUDIO_MODEL_PATH)

    return image_model, video_model, audio_model


def show_result(label: str):
    if label == "REAL":
        st.markdown(
            f'<div class="result-real">✅ Prediction: {label}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="result-fake">⚠️ Prediction: {label}</div>',
            unsafe_allow_html=True,
        )


def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = np.array(image)
    image = cv2.resize(image, (224, 224))
    image = image.astype("float32") / 255.0
    return image.reshape((1, 224, 224, 3))


def extract_audio_features(audio_file):
    audio, sample_rate = librosa.load(audio_file, sr=None)

    rms = np.mean(librosa.feature.rms(y=audio))
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20)
    mfcc_features = mfccs.T[:, [0, 1, 4, 6, 12, 13, 15, 16, 18, 19]]

    features = [rms] + list(np.mean(mfcc_features, axis=0))
    feature_names = [
        "rms", "mfcc1", "mfcc2", "mfcc5", "mfcc7",
        "mfcc13", "mfcc14", "mfcc16", "mfcc17", "mfcc19", "mfcc20"
    ]
    return pd.DataFrame([features], columns=feature_names)


def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y:start_y + min_dim, start_x:start_x + min_dim]


def load_video_frames(path, max_frames=20, resize=(224, 224)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if max_frames and len(frames) == max_frames:
                break
    finally:
        cap.release()

    return np.array(frames)


def prepare_single_video(frames, feature_extractor, max_seq_length=20, num_features=2048):
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, max_seq_length), dtype="bool")
    frame_features = np.zeros((1, max_seq_length, num_features), dtype="float32")

    video_length = frames.shape[1]
    length = min(max_seq_length, video_length)

    for j in range(length):
        prediction = feature_extractor.predict(frames[:, j, :], verbose=0)
        frame_features[0, j, :] = prediction

    frame_mask[0, :length] = 1
    return frame_features, frame_mask


def predict_video(video_path, video_model, feature_extractor):
    frames = load_video_frames(video_path, max_frames=20)

    if len(frames) == 0:
        raise ValueError("No frames could be read from the video.")

    frame_features, frame_mask = prepare_single_video(frames, feature_extractor)
    prediction = video_model.predict([frame_features, frame_mask], verbose=0)

    score = float(prediction[0][0]) if np.ndim(prediction) > 1 else float(prediction[0])
    label = "FAKE" if score <= 0.5 else "REAL"
    return label, score


# =========================
# UI
# =========================
try:
    model_image, video_model, model_audio = load_all_models()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

st.markdown(
    """
    <div class="hero-box">
        <div class="main-title">🛡️ Deepfake Detection Studio</div>
        <div class="sub-text">
            Analyze audio, images, and video with a cleaner, scrollable interface.
            Upload your media and get a prediction instantly.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        """
        <div class="info-card">
            <h3>🎵 Audio</h3>
            <p>Detect fake or real speech using extracted audio features.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        """
        <div class="info-card">
            <h3>🖼️ Image</h3>
            <p>Classify uploaded images as real or manipulated.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        """
        <div class="info-card">
            <h3>🎬 Video</h3>
            <p>Analyze video frames and sequence patterns for deepfakes.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🎵 Audio Detection", "🖼️ Image Detection", "🎬 Video Detection"])


with tab1:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("Audio Deepfake Detection")
    st.write("Upload a WAV or MP3 file and classify it as REAL or FAKE.")

    audio_file = st.file_uploader(
        "Choose an audio file",
        type=["wav", "mp3"],
        key="audio_uploader",
    )

    if audio_file is not None:
        st.audio(audio_file)

        if st.button("Predict Audio", use_container_width=True):
            with st.spinner("Analyzing audio..."):
                features = extract_audio_features(audio_file)
                prediction = model_audio.predict(features)[0]
                label = "REAL" if prediction == 1 else "FAKE"
                show_result(label)
    st.markdown("</div>", unsafe_allow_html=True)


with tab2:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("Image Deepfake Detection")
    st.write("Upload a JPG, JPEG, or PNG image and classify it as REAL or FAKE.")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        key="image_uploader",
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns([1.2, 1])

        with col1:
            st.image(image, caption="Uploaded image", use_container_width=True)

        with col2:
            st.write("Press the button below to analyze the image.")
            if st.button("Predict Image", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    processed = preprocess_image(image)
                    prediction = model_image.predict(processed, verbose=0)
                    score = float(prediction[0][0])
                    label = "FAKE" if score > 0.5 else "REAL"
                    show_result(label)
                    st.progress(min(max(score, 0.0), 1.0))
                    st.caption(f"Model score: {score:.4f}")
    st.markdown("</div>", unsafe_allow_html=True)


with tab3:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("Video Deepfake Detection")
    st.write("Upload an MP4 or AVI file. Video analysis may take longer.")

    uploaded_video = st.file_uploader(
        "Choose a video",
        type=["mp4", "avi"],
        key="video_uploader",
    )

    if uploaded_video is not None:
        st.video(uploaded_video)

        if st.button("Predict Video", use_container_width=True):
            with st.spinner("Processing video and analyzing frames..."):
                feature_extractor = build_feature_extractor()

                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                    tmp_file.write(uploaded_video.read())
                    temp_video_path = tmp_file.name

                try:
                    label, score = predict_video(temp_video_path, video_model, feature_extractor)
                    show_result(label)
                    st.progress(min(max(score, 0.0), 1.0))
                    st.caption(f"Model score: {score:.4f}")
                except Exception as e:
                    st.error(f"Video processing failed: {e}")
                finally:
                    if os.path.exists(temp_video_path):
                        os.remove(temp_video_path)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.caption("Built with Streamlit • Deepfake Detection Project")