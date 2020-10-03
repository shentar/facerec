import hashlib
import json
import pickle

import cv2
import dlib
import numpy as np
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")


def detect_one_file(data):
    data = np.frombuffer(data, np.uint8)
    if data is None:
        raise Exception('image is required.')

    zoom_ratio = 1
    # if data.size > 6 * 1024 * 1024:
    #     img = cv2.imdecode(data, cv2.IMREAD_REDUCED_COLOR_4)
    #     zoom_ratio = 4
    # elif data.size > 4 * 1024 * 124:
    #     img = cv2.imdecode(data, cv2.IMREAD_REDUCED_COLOR_2)
    #     zoom_ratio = 2
    # else:
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)

    faces = []
    dets = detector(img, 1)

    for d in dets:
        if d.width() * zoom_ratio < 100:
            continue
        shape = sp(img, d)
        des_buffer = pickle.dumps(facerec.compute_face_descriptor(img, shape))
        h = hashlib.sha256()
        h.update(des_buffer)
        id = h.hexdigest()

        f = {
            'id': id,
            'des_buffer': des_buffer
        }
        res = requests.put('http://127.0.0.1:5001/sqlite/insert_face', data=pickle.dumps(f))
        if not res.ok:
            raise Exception("insert one face to the sqlite failed.")

        result = json.loads(res.content)
        if result is None or result['success'] is None or not result['success']:
            raise Exception("insert one face to the sqlite failed.")

        face = {
            'token': id,
            'rectangle': {
                'width': d.width() * zoom_ratio,
                'height': d.height() * zoom_ratio,
                'left': d.left() * zoom_ratio,
                'top': d.top() * zoom_ratio,
            },
            'age': 0,
            'quality': 0,
            'gender': 0,
        }
        faces.append(face)
    return faces


@app.route("/api/detect", methods=['PUT'])
def face_detect():
    try:
        faces = detect_one_file(request.get_data())
        response = {
            'success': True,
            'count': len(faces),
            'faces': faces,
        }
        resp = jsonify(response)
        resp.status_code = 200
    except Exception as e:
        response = {
            'success': False,
            'count': 0,
            'faces': [],
        }
        resp = jsonify(response)
        resp.status_code = 400
    return resp


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )
