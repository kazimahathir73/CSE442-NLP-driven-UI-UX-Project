from flask import Flask, render_template, Response
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
from time import time
from selenium import webdriver

app = Flask(__name__)

face_classifier = cv2.CascadeClassifier(r"D:\CSE BRACU\CSE442-NLP-driven-UI-UX\Project\haarcascade_frontalface_default.xml")
emotion_classifier = load_model(r"D:\CSE BRACU\CSE442-NLP-driven-UI-UX\Project\rafdb_best_model.h5")
emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

current_emotion = None
emotion_start_time = None
playlist_opened = False
driver = None

def handle_emotion_change(new_emotion):
    global current_emotion, emotion_start_time, playlist_opened
    if new_emotion != current_emotion:
        current_emotion = new_emotion
        emotion_start_time = time()
        playlist_opened = False
    elif time() - emotion_start_time >= 3 and not playlist_opened:
        if current_emotion != 'Neutral':
            open_playlist(current_emotion)
            playlist_opened = True

def open_playlist(emotion):
    global driver
    playlists = {
        'Angry': 'https://www.youtube.com/playlist?list=PL4p8vRTxHvsOgYdPDPf78404-2RvrkQlY',
        'Disgust': 'https://www.youtube.com/playlist?list=PL4p8vRTxHvsNnSH4OFht0T6im85QUcZfC',
        'Fear': 'https://www.youtube.com/playlist?list=PL4p8vRTxHvsPg2P73rcu9DP_eAZNdSX84',
        'Happy': 'https://www.youtube.com/playlist?list=PL4p8vRTxHvsMPy0Vvq_NVmKw2fVBOwv0m',
        'Sad': 'https://www.youtube.com/playlist?list=PL4p8vRTxHvsPRmxCJEojilHJpVULhkyem',
        'Surprise': 'https://www.youtube.com/playlist?list=PL4p8vRTxHvsP3ajS0xSanWYWes3VV1fb6'
    }

    if emotion in playlists:
        driver = webdriver.Chrome()
        driver.get(playlists[emotion])
    else:
        print("No playlist found for the detected emotion.")

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                prediction = emotion_classifier.predict(roi)[0]
                label = emotion_labels[prediction.argmax()]
                label_position = (x, y)
                cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                handle_emotion_change(label)
            else:
                cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
