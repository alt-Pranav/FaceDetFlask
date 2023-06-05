# ref: https://github.com/DharmarajPi/Opencv-face-detection-deployment-using-flask-API/blob/main/app.py
# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from flask import Flask, render_template, Response
import cv2
import mediapipe as mp

# Flask constructor takes the name of
# current module (__name__) as argument.
app = Flask(__name__)


def capture_by_frames(): 
    global camera
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection()
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()  # read the camera frame
        # Convert the frame to RGB (BlazeFace requires RGB input)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces using BlazeFace
        results = face_detection.process(frame_rgb)
        if results.detections:
            for detection in results.detections:
                # Extract the bounding box coordinates
                box = detection.location_data.relative_bounding_box
                #print(detection)
                conf = str(detection.score)[1:5]
                # Convert relative coordinates to absolute coordinates
                h, w, _ = frame.shape
                x = int(box.xmin * w)
                y = int(box.ymin * h)
                width = int(box.width * w)
                height = int(box.height * h)
                # Draw the bounding box on the frame
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                cv2.putText(frame, conf, (x+width, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)


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
	return render_template('index.html')

@app.route('/start', methods=['POST'])
def start():
	return render_template('index.html')

@app.route('/stop', methods=['POST'])
def stop():
	if camera.isOpened():
		camera.release()
	return render_template('stop.html')

@app.route('/video_capture')
def video_capture():
	return Response(capture_by_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# main driver function
if __name__ == '__main__':

	# run() method of Flask class runs the application
	# on the local development server.
	app.run(port=5000)
