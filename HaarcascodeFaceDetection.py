import cv2
import os

#Dataset Creation
# Specify the path to the dataset folder and the person's name
dataset = r"D:\AI Projects\dataset"
name = "champ"

# Create the directory for the person if it doesn't exist
path = os.path.join(dataset, name)
if not os.path.isdir(path):
    os.mkdir(path)

# Set the dimensions for the captured face images
(width, height) = (130, 100)

# Load the Haar cascade for face detection
alg = cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"
haar_cascade = cv2.CascadeClassifier(alg)

# Check if the cascade classifier was loaded successfully
if haar_cascade.empty():
    print("Error: Cascade classifier not loaded.")
else:
    print("Cascade classifier loaded successfully.")

# Initialize the camera
cam = cv2.VideoCapture(0)

# Initialize the count for capturing multiple images
count = 1

while count <= 30:  # Capture 30 images
    print(count)
    
    # Read a frame from the camera
    _, img = cam.read()

    # Convert the frame to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = haar_cascade.detectMultiScale(gray_img, scaleFactor=1.3, minNeighbors=4)

    if len(faces) > 0:
        print("Person detected")
    else:
        print("No Person Detected")

    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Extract the region of interest (face) from the grayscale image
        face_only = gray_img[y:y+h, x:x+w]

        # Resize the face image to the specified dimensions
        resize_img = cv2.resize(face_only, (width, height))

        # Save the captured face image to the dataset folder
        cv2.imwrite(os.path.join(path, f"{count}.jpg"), resize_img)

    count += 1

    # Display the frame with face detection
    cv2.imshow("FaceDetection", img)

    # Wait for a key press to exit the loop
    key = cv2.waitKey(10)
    if key == 27:  # Press 'Esc' to exit
        break

# Release the camera and close all OpenCV windows
print("Image capture completed successfully.")
cam.release()
cv2.destroyAllWindows()
