import streamlit as st
import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime

# 1. PAGE CONFIG
st.set_page_config(page_title="Smart Attendance", page_icon="📸")
st.title("📸 Smart Attendance System")

# 2. LOAD DATABASE
@st.cache_resource
def load_database():
    path = 'Images'
    if not os.path.exists(path):
        os.makedirs(path)
        return [], []

    images = []
    classNames = []
    myList = os.listdir(path)
    
    for cl in myList:
        if cl.lower().endswith(('.jpg', '.png', '.jpeg')):
            curImg = cv2.imread(f'{path}/{cl}')
            if curImg is not None:
                images.append(cv2.cvtColor(curImg, cv2.COLOR_BGR2RGB))
                classNames.append(os.path.splitext(cl)[0])
    
    encodeListKnown = []
    for img in images:
        encodes = face_recognition.face_encodings(img)
        if encodes:
            encodeListKnown.append(encodes[0])
            
    return encodeListKnown, classNames

with st.spinner('Loading Database...'):
    encodeListKnown, classNames = load_database()

# 3. CAMERA INTERFACE (The Safe Way - No "While True"!)
img_file_buffer = st.camera_input("Click to Mark Attendance")

if img_file_buffer is not None:
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    imgS = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    
    if not facesCurFrame:
        st.warning("No face detected.")
    else:
        for encodeFace in encodesCurFrame:
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                st.success(f"✅ MARKED: {name}")
                
                # Save to CSV
                with open('Attendance.csv', 'a') as f:
                    now = datetime.now()
                    f.write(f'\n{name},{now.strftime("%H:%M:%S")},{now.strftime("%d/%m/%Y")}')
            else:
                st.error("Unknown Face.")

# 4. SHOW DATA
if os.path.exists('Attendance.csv'):
    st.write("---")
    st.header("Log")
    with open('Attendance.csv', 'r') as f:
        st.text(f.read())