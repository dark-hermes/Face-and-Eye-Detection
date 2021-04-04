import cv2

# Read HAAR face and eye cascades from system
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Capture video from webcam
cap = cv2.VideoCapture(0)

while True:

    # Read video frame by frame and convert the color into gray scale
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect face in multi scale
    # reduce image size by 30%
    # 3 ~ 6 is a good value for minimum neighbors
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Iterate faces for x and y coordinates
    # w and h for weight and height
    for x, y, w, h in faces:

        # Draw rectangles around detected faces
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)

        # Take Region of Interest of frame in gray scale and rgb
        roi_gray = gray[y:y + h, x:x + w]
        roi_rgb = frame[y:y + h, x:x + w]

        # Detect eye in multi scale form region of interest
        # with the same parameters as faces
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)

        # Iterates eyes for it's features
        # ex and ey for each detected eyes X and Y coordinates
        # ew and eh for each eyes weights and heights coordinates
        for ex, ey, ew, eh in eyes:
            # draw rectangles around the eyes
            cv2.rectangle(roi_rgb, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 3)

    # Show the video capture
    cv2.imshow('Frame', frame)

    # Press 'q' on keyboard will close the program
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and close the windows
cap.release()
cv2.destroyAllWindows()
