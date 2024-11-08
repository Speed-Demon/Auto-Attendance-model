import streamlit as st
import numpy as np
from PIL import Image
from joblib import load
from keras_facenet import FaceNet
from mtcnn.mtcnn import MTCNN
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches
# Load the models
detector = MTCNN()
# Load the models
svm_model = load("svm_face_recognition_model.joblib")
facenet_model = FaceNet()
facenet_model.load_model('facenet_embedding_model.h5')
label_encoder = load("label_encoder.joblib")

# Define the functions
def extract_faces(filename, required_size=(160, 160)):
    image = Image.open(filename)
    image = image.convert('RGB')
    pixels = np.asarray(image)
    results = detector.detect_faces(pixels)
    faces = []
    bounding_boxes = []
    for result in results:
        confidence = result['confidence']
        if confidence < 0.75:
            continue
        x1, y1, width, height = result['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face = pixels[y1:y2, x1:x2]
        face_image = Image.fromarray(face)
        face_image = face_image.resize(required_size)
        faces.append(np.asarray(face_image))
        bounding_boxes.append((x1, y1, x2, y2, confidence))
    return faces, bounding_boxes

def predict_face(face_pixels):
    face_pixels = np.expand_dims(face_pixels, axis=0)
    embedding = facenet_model.embeddings(face_pixels)[0]
    embedding = np.expand_dims(embedding, axis=0)
    yhat_class = svm_model.predict(embedding)
    yhat_prob = svm_model.predict_proba(embedding)
    class_index = yhat_class[0]
    class_probability = yhat_prob[0, class_index] * 100
    predicted_name = label_encoder.inverse_transform([class_index])[0]
    return predicted_name, class_probability

def predict_faces(image_path):
    faces, bounding_boxes = extract_faces(image_path)
    if not faces:
        st.write("No faces detected with confidence >= 60%")
        return

    image = Image.open(image_path)
    plt.figure(figsize=(15, 15))
    plt.imshow(image)
    ax = plt.gca()

    for face_pixels, (x1, y1, x2, y2, confidence) in zip(faces, bounding_boxes):
        name, prob = predict_face(face_pixels)
        if name is not None and prob is not None and prob >= 60:
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.text(x1, y1 - 10, f'{name}: {prob:.2f}%', color='red', fontsize=12, weight='bold')
    
    plt.axis('off')
    st.pyplot(plt)

# Streamlit app
st.title("Face Recognition App")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image_path = uploaded_file
    predict_faces(image_path)

def mark_attendance(image_path):
    faces, bounding_boxes = extract_faces(image_path)
    attendance = []

    for face_pixels, (x1, y1, x2, y2, confidence) in zip(faces, bounding_boxes):
        name, prob = predict_face(face_pixels)
        if name is not None and prob is not None and prob >= 60:
            attendance.append((name, 'Present'))

    return attendance

attendance_list = mark_attendance(uploaded_file)

attendance_df = pd.DataFrame(attendance_list, columns=['Name', 'Attendance'])

st.write(attendance_df)

if not attendance_df.empty:
    csv = attendance_df.to_csv(index=False)
    st.download_button(
        label="Download attendance as CSV",
        data=csv,
        file_name='attendance.csv',
        mime='text/csv',
    )