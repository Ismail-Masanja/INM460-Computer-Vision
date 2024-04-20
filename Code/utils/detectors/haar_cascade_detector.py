import cv2


__all__ = ['haar_cascade_detector']


def haar_cascade_detector(frame):
    """ Uses cv2 haarcascades to detect bounding boxes of faces in a frame"""
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    return [(x, y, x + w, y + h) for x, y, w, h in faces]
