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

    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)

        roi_gray = gray[y:y + h, x:x + w]
        roi_bgr = frame[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)

        for ex, ey, ew, eh in eyes:
            cv2.rectangle(roi_bgr, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 3)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
