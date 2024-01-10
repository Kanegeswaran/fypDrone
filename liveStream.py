import cv2   #include opencv library functions in python
import torch
import numpy as np
from time import time
import os


class PlasticDetection:

    def __init__(self, frame, model_name):
       
        self.frame = frame
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        

    def load_model(self, model_name):
        yolov5_path = os.path.join(os.getcwd(), 'yolov5')

        if model_name:
            model_path = os.path.join(os.getcwd(), 'yolov5', 'trained_model', model_name)
            model = torch.hub.load(yolov5_path, 'custom', path=model_path, source='local')            
        else:
            model_path = os.path.join(os.getcwd(), 'yolov5', 'trained_model', 'yolov5s.pt')
            model = torch.hub.load(yolov5_path, path=model_path, source='local', pretrained=True)
        return model


    def score_frame(self, frame):
       
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]     
        return labels, cord
    
    def class_to_label(self, x):
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.5:
                x1,  y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1,y1), (x2,y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        return frame
        
    
    def __call__(self):

       
        frame = cv2.resize(self.frame, (640,640))

        start_time = time()
        results = self.score_frame(frame)
        frame = self.plot_boxes(results, frame)
        
        end_time = time()
        fps = 1/np.round(end_time - start_time, 2)
            
        return frame    
        
    
        #     frame = cv2.resize(frame, (640,640))
        #     results = model(frame)
        #     labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]     
        #     n = len(labels)
        #     x_shape, y_shape = frame.shape[1], frame.shape[0]
        #     for i in range(n):
        #         row = cord[i]
        #         if row[4] >= 0.5:
        #             x1,  y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
        #             bgr = (0, 255, 0)
        #             cv2.rectangle(frame, (x1,y1), (x2,y2), bgr, 2)
        #             cv2.putText(frame, model.names[int(labels[i])], (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        #     ret,buffer=cv2.imencode('.jpg',frame)
        #     frame=buffer.tobytes()

        # yield(b'--frame\r\n'
        #            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

