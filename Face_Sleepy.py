import cv2
import dlib as d
from scipy.spatial import distance

def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear_aspect_ratio = (A+B) / (2.0 * C)
    return ear_aspect_ratio

cap = cv2.VideoCapture(0)
cap.set(2, 1080)
cap.set(3, 720)
hog_face = d.get_frontal_face_detector()
dlib_faces = d.shape_predictor('shape_predictor_68_face_landmarks.dat')

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = hog_face(gray)

    for face in faces:
        face_landmarks = dlib_faces(gray, face)
        left_eye = []
        right_eye = []

        # Draw the lines around the eye 
        for n in range(36, 42):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            left_eye.append((x, y))
            next_point = n + 1
            if n == 41:
                next_point = 36
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)
        for n in range(42, 48):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            right_eye.append((x, y))
            next_point = n + 1
            if n == 47:
                next_point = 42
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

        left_ear = calculate_EAR(left_eye)
        right_ear = calculate_EAR(right_eye)

        EAR = round((left_ear + right_ear) / 2, 2)
        # U can use EAR < 0.26
        if EAR < 0.28:
            cv2.putText(frame, 'Are You Sleepy?', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)

    cv2.imshow('ff', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

