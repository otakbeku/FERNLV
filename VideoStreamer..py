from flask import Flask, render_template, Response
from Camera import VideoCamera


app = Flask(__name__)


def camera_handler(camera: VideoCamera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n'
               + frame + b'\r\n\r\n')


@app.route('/')
def hello_world():
    # return 'Hello World!'
    return render_template('index.html')


@app.route('/video_streamer')
def video_streamer():
    # return Response(camera_handler(VideoCamera('F:\FSR\FERNLV\gtkom.mp4')),
    return Response(camera_handler(VideoCamera('AFW1.mp4')),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
