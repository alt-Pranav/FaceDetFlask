# ref: https://github.com/DharmarajPi/Opencv-face-detection-deployment-using-flask-API/blob/main/app.py
# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np

# Flask constructor takes the name of
# current module (__name__) as argument.
app = Flask(__name__)


def capture_by_frames(): 
    global camera
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection()
    camera = cv2.VideoCapture(0)
    # initial combo 1280 X 720
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    while True:
	
        success, frame = camera.read()  # read the camera frame
        #print(frame.shape) # default is 480 h and 640 w

        # Convert the frame to RGB (BlazeFace requires RGB input)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces using BlazeFace
        results = face_detection.process(frame_rgb)
        if results.detections:
            for detection in results.detections:
                # Extract the bounding box coordinates
                box = detection.location_data.relative_bounding_box
                #print(detection)
                conf = detection.score
                # Convert relative coordinates to absolute coordinates
                h, w, _ = frame.shape
                x = int(box.xmin * w)
                y = int(box.ymin * h)
                width = int(box.width * w)
                height = int(box.height * h)
                # Draw the bounding box on the frame
                # note to self: y + n pushes down by n
                # MAIN BOUNDING BOX: cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
		

                face_acc = 'FACE: {0}'.format(str(conf)[1:5])
                face_acc_size, _ = cv2.getTextSize(face_acc, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                
                if conf and len(str(conf)) > 5:
                    if float(str(conf[0])[:5]) >= 0.9:
                        seenText = 'I SEE YOU'
                        seenText_size, _ = cv2.getTextSize(seenText, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        overlay = frame.copy()
                        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255), 2)
                        cv2.rectangle(overlay, (x, y), (x+width, y+height), (0, 0, 255), -1)  
    
                        alpha = 0.4  # Transparency factor.
                        
                        # Following line overlays transparent rectangle
                        # over the image
                        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
                        cv2.rectangle(frame, (x, y), (x+seenText_size[0], y-seenText_size[1]), (0, 0, 255), -1)
                        cv2.putText(frame, seenText, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                    else:
                        unseenText = 'ARE YOU THERE?'
                        unseenText_size, _ = cv2.getTextSize(unseenText, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        cv2.rectangle(frame, (x, y), (x+unseenText_size[0], y-unseenText_size[1]), (255, 0, 0), -1)
                        cv2.putText(frame, unseenText, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Draw the filled background rectangle
                # MAIN BG RECT: cv2.rectangle(frame, (x, y), (x+face_acc_size[0], y-face_acc_size[1]), (0, 255, 0), -1)

                #cv2.rectangle(frame, (x+width, y), (x +width + 100, y - 30), (0, 255, 0), -1)# black box bg
                #cv2.putText(frame, conf, (x+width, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0,0), 10) # for black outline on accuracy text
                #cv2.putText(frame, conf, (x+width, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)
                # MAIN FACE ACC: cv2.putText(frame, face_acc, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)                


        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



# The route() function of the Flask class is a decorator,
# which tells the application which URL should call
# the associated function.
@app.route('/')
# ‘/’ URL is bound with hello_world() function.
def index():
	return render_template('mystart.html')

@app.route('/start', methods=['POST'])
def start():
	return render_template('mystart.html')

@app.route('/stop', methods=['POST'])
def stop():
	if camera.isOpened():
		camera.release()
	return render_template('mystop.html')

@app.route('/video_capture')
def video_capture():
	return Response(capture_by_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# main driver function
if __name__ == '__main__':

	# run() method of Flask class runs the application
	# on the local development server.
    # deug = True enables hot reload
	app.run(debug=True, port=5000)
