import cv2
import cvzone

def generate_frames(face_shape, num):
    cap = cv2.VideoCapture(0)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        success, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if not success:
            break

        gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray_scale)

        overlay = cv2.imread(
            f'/home/hprad/Projects/capstone/VirtualTryOn/Glasses/{face_shape}/glass ({num}).png',
            cv2.IMREAD_UNCHANGED
        )

        if overlay is not None:
            for (x, y, w, h) in faces:
                overlay_resize = cv2.resize(overlay, (w, int(h * 0.8)))
                frame = cvzone.overlayPNG(frame, overlay_resize, [x, y])

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')