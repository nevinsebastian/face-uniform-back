import face_recognition
import cv2
import os
from fastapi import FastAPI, HTTPException

app = FastAPI()

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
known_faces = []
known_names = []

# Load and encode known faces
known_faces.append(face_recognition.load_image_file(os.path.join(current_dir, "images/binn.jpg")))
known_faces.append(face_recognition.load_image_file(os.path.join(current_dir, "images/Nevin.jpg")))
known_faces.append(face_recognition.load_image_file(os.path.join(current_dir, "images/sreejith.jpg")))

known_names = ["Nevin","sreejith","binn"]

# Encode known faces
known_encodings = []
for img in known_faces:
    face_encoding = face_recognition.face_encodings(img)
    if len(face_encoding) > 0:
        known_encodings.append(face_encoding[0])
    else:
        print("No face found in one of the images.")

@app.get("/")
async def root():
    return {"message": "Welcome to face recognition API!"}

@app.get("/mark-attendance")
async def mark_attendance():
    # Initialize webcam
    video_capture = cv2.VideoCapture(0)
    attendance_list = []

    # Capture a single frame
    ret, frame = video_capture.read()

    # Find all the faces and face encodings in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Loop through each face found in the frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face matches any of the known faces
        matches = face_recognition.compare_faces(known_encodings, face_encoding)

        name = "Unknown"

        # If a match is found, use the name of the matched known face
        if True in matches:    
            first_match_index = matches.index(True)
            name = known_names[first_match_index]
            attendance_list.append(name)

    # Release the webcam
    video_capture.release()

    if not attendance_list:
        raise HTTPException(status_code=404, detail="No faces found for attendance.")

    return {"message": "Attendance marked successfully!", "attendance": attendance_list}
