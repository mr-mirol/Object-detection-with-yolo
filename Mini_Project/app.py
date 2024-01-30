from flask import Flask, render_template, request
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# Initialize YOLO model
model = YOLO("yolov8n.pt")

# Function to run the object detection script
def run_object_detection():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error opening video file!")
        exit(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Uncomment to display the video frame by frame
        results = model(frame, conf=0.4, show=True)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Video reading complete!")

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_detection', methods=['POST'])
def run_detection():
    run_object_detection()
    return render_template('index.html', message='Object detection complete!')

if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')
