# ref: https://github.com/DharmarajPi/Opencv-face-detection-deployment-using-flask-API/blob/main/app.py
# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from flask import Flask, render_template, Response
import cv2
import mtcnn

# Flask constructor takes the name of
# current module (__name__) as argument.
app = Flask(__name__)

def detect_bounding_box(frame):
    # detect faces in the image
    faces = detector.detect_faces(frame)
    """ for face in faces:
        print(face) """
    
    for face in faces:
        x, y, width, height = face['box']
        conf = face['confidence']
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 4)
        cv2.putText(frame, str(conf)[:4], (x+width, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    return faces

def capture_by_frames(): 
    global camera
    global detector
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()  # read the camera frame
        detector=mtcnn.MTCNN()
        faces=detect_bounding_box(frame)

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