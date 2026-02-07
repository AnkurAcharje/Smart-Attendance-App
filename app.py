import gradio as gr
import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime
from PIL import Image

# --- 1. Load Database ---
def load_encodings():
    path = "Images"
    encoded_faces = []
    names = []
    
    if not os.path.exists(path):
        os.makedirs(path)
        return [], []
    
    print("Loading database...")
    for file in os.listdir(path):
        if file.endswith(('.jpg', '.jpeg', '.png')):
            try:
                img_path = os.path.join(path, file)
                pil_img = Image.open(img_path).convert("RGB")
                img = np.array(pil_img)
                encs = face_recognition.face_encodings(img)
                if encs:
                    encoded_faces.append(encs[0])
                    names.append(os.path.splitext(file)[0])
                    print(f"Loaded: {file}")
            except Exception as e:
                print(f"Skipped {file}: {e}")
    return encoded_faces, names

known_encodings, known_names = load_encodings()

# --- 2. The Logic ---
def scan_face(pil_image):
    if pil_image is None:
        return None, "⚠️ Error: Please click the small 'Snap' circle inside the camera first!"

    try:
        # Convert PIL to Numpy
        image_array = np.array(pil_image.convert("RGB"))
        
        # Find faces
        face_locs = face_recognition.face_locations(image_array)
        face_encs = face_recognition.face_encodings(image_array, face_locs)
        
        log_text = "❓ No face detected."
        
        # Prepare image for drawing (RGB -> BGR)
        paint_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        for (top, right, bottom, left), face_enc in zip(face_locs, face_encs):
            matches = face_recognition.compare_faces(known_encodings, face_enc, tolerance=0.5)
            name = "Unknown"
            color = (0, 0, 255) # Red

            if True in matches:
                first_match_index = matches.index(True)
                name = known_names[first_match_index].upper()
                color = (0, 255, 0) # Green
                time_now = datetime.now().strftime('%H:%M:%S')
                log_text = f"✅ ATTENDANCE MARKED: {name} at {time_now}"
            else:
                log_text = "❌ Face detected, but not recognized."

            cv2.rectangle(paint_image, (left, top), (right, bottom), color, 3)
            cv2.putText(paint_image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        final_image = cv2.cvtColor(paint_image, cv2.COLOR_BGR2RGB)
        return final_image, log_text

    except Exception as e:
        return None, f"⚠️ ERROR: {str(e)}"

# --- 3. Interface (Classic Style) ---
with gr.Blocks(title="Smart Attendance") as demo:
    gr.Markdown("# 📸 Smart Attendance System")
    
    with gr.Row():
        inp = gr.Image(source="webcam", label="1. Snap Photo Here", type="pil")
        out_img = gr.Image(label="3. Result")
    
    status = gr.Textbox(label="System Status", value="Ready.")
    btn = gr.Button("🔴 2. MARK ATTENDANCE", variant="primary")
    
    btn.click(fn=scan_face, inputs=inp, outputs=[out_img, status])

demo.launch(server_name="0.0.0.0", server_port=7860)
