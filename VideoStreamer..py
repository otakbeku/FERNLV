from flask import Flask, render_template, Response
from Camera import VideoCamera

app = Flask(__name__)

Detected_face = []
VID = VideoCamera(1)


def camera_handler(camera: VideoCamera):
    while True:
        frame, face = camera.get_frame()
        # frame, faces = camera.get_frames()
        # if hasattr(face, 'shape'):
        # if hasattr(faces, 'append'):
        #     [Detected_face.append(face) for face in faces]
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n'
               + frame + b'\r\n\r\n')




def display_face():
    while True:
        for image in Detected_face:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n'
                   + image + b'\r\n\r\n')


@app.route('/')
def hello_world():
    # return 'Hello World!'
    return render_template('index.html')


@app.route('/video_streamer')
def video_streamer():
    return Response(camera_handler(VideoCamera('F:\FSR\FERNLV\\AFW1.mp44')),
    # return Response(camera_handler(VideoCamera(1)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/detected_face')
def detected_face():
    return Response(display_face(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
