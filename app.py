from flask import Flask, render_template, Response
from detector import DetectorFace
app = Flask(__name__)


def gen(detector_face):
    while True:
        frame = detector_face.get_frame()

        predicted_img = detector_face.predict(frame)[1]
        if predicted_img is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + predicted_img + b'\r\n\r\n')


@app.route('/streaming/<int:cam_id>')
def video_feed(cam_id):
    return Response(gen(DetectorFace()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
