import pickle
import sqlite3
from datetime import datetime, timezone

import numpy as np
from flask import Flask, request
from flask.json import loads, jsonify

from dao import SQLiteUtil
DBPATH = "face_recon.db"
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

sqlite = SQLiteUtil(DBPATH)


@app.route("/sqlite/insert_face", methods=['PUT'])
def insert_face():
    data = request.get_data()
    face = None
    if data is not None:
        face = pickle.loads(data)
    if face is not None:
        sqlite.execute("INSERT INTO faces values(?,?,?)",
                       [face['id'], sqlite3.Binary(face['des_buffer']), datetime.now(timezone.utc)])
        res = {
            'success': True,
            'message': 'successfully insert the face: ' + face['id']
        }
    else:
        res = {
            'success': False,
            'message': 'the input faceid is empty.'
        }

    response = jsonify(res)
    response.status_code = 200
    return response


@app.route("/api/search", methods=['PUT'])
def face_search():
    face_id_in = request.args.get('facetoken')
    if face_id_in is None:
        response = {
            'success': False,
            'messages': "the input face id is empty.",
        }
        resp = jsonify(response)
        resp.status_code = 400
        return resp

    max_distance = request.args.get('max-distance')
    if max_distance is None:
        max_distance = 0.3
    else:
        max_distance = float(max_distance)

    res = sqlite.execute_query("SELECT facetoken from faces where id=?;", [face_id_in])
    if len(res) != 1:
        response = {
            'success': False,
            'messages': "the input face id is not found in the database.",
        }
        resp = jsonify(response)
        resp.status_code = 400
        return resp
    face_token_in = res[0]['facetoken']
    if face_token_in is None:
        response = {
            'success': False,
            'messages': "the input face id has no face token found.",
        }
        resp = jsonify(response)
        resp.status_code = 400
        return resp

    face_in = pickle.loads(face_token_in)
    face_in = np.array(face_in)
    faces = []
    result = sqlite.execute_query("SELECT id,facetoken FROM faces where id != ?;", [face_id_in])
    for record in result:
        face_token = record['facetoken']
        face = pickle.loads(face_token)
        face = np.array(face)
        res = np.linalg.norm([face] - face_in, axis=1)
        if res[0] <= max_distance:
            face = {
                'token_id': record['id'],
                'distance': res[0],
            }
            faces.append(face)

    response = {
        'success': True,
        'count': len(faces),
        'faces': faces,
    }

    resp = jsonify(response)
    resp.status_code = 200
    return resp


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5001,
        debug=True
    )
