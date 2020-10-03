import queue

import dlib


def singleton(cls):
    instances = {}

    def _singleton(*args, **kw):
        if cls not in instances:
            instances[cls] = cls(*args, **kw)
        return instances[cls]

    return _singleton


@singleton
class Detectors(object):
    __queue = queue.Queue(maxsize=4)

    def __init__(self):
        for i in range(4):
            detector = dlib.get_frontal_face_detector()
            sp = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
            facerec = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")
            detector = {
                'detector': detector,
                'sp': sp,
                'facerec': facerec
            }
            self.__queue.put(detector)

    def close(self, detector):
        if detector is not None:
            self.__queue.put(detector)

    def open(self):
        return self.__queue.get()
