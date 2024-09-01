import cv2
import shutil

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)

# Define the destination path for the classifier XML file
destination_path = '../models/haarcascade_frontalface_default.xml'

# Copy the XML file to the new location
shutil.copy(face_cascade_path, destination_path)
