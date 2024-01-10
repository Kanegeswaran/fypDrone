from flask import Flask, render_template, Response
import cv2
import os
import torch
from time import time
import numpy as np
import multiprocessing as mp

app = Flask(__name__)
app.secret_key = "Plastic Detection Using Drone"
# camera = cv2.VideoCapture(0)
camera = cv2.VideoCapture("rtmp://10.164.38.22:1935/live")

yolov5_path = os.path.join(os.getcwd(), 'yolov5')
model_path = os.path.join(os.getcwd(), 'yolov5', 'trained_model', 'best.pt')
model = torch.hub.load(yolov5_path, 'custom', path=model_path, source='local')



def process_frame(frame):
    start_time = time()
    results = model(frame)
    labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    for i in range(n):
        row = cord[i]
        if row[4] >= 0.5:
            x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)
            bgr = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
            cv2.putText(frame, model.names[int(labels[i])] + ' : ' + str(round(row.cpu().numpy()[4] * 100, 2)),
                        (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    end_time = time()
    fps = 1 / np.round(end_time - start_time, 2)
    cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
    ret, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes()


def generate_frames(queue):
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        else:
            frame = cv2.resize(frame, (640, 640))
            start_time = time()
            queue.put(process_frame(frame.copy()))


@app.route('/')
def index():
    return render_template('index.html')


def frame_generator(queue):
    while True:
        yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + queue.get() + b'\r\n'


@app.route('/video')
def video():
    frame_queue = mp.Queue()
    process = mp.Process(target=generate_frames, args=(frame_queue,))
    process.daemon = True
    process.start()
    return Response(frame_generator(frame_queue), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/report')
def report():
    # show the form, it wasn't submitted
    return render_template('report.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0")
