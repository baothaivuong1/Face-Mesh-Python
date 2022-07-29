#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2 as cv
import numpy as np
import mediapipe as mp
from math import sqrt


class FaceMesh(object):
    def __init__(
        self,
        max_num_faces=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    ):
        mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def __call__(
        self,
        image,
    ):

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = self._face_mesh.process(image)

        face_mesh_results = []
        if results.multi_face_landmarks is not None:
            for face_landmarks in results.multi_face_landmarks:
                face_mesh_results.append(
                    self._calc_landmarks(image, face_landmarks.landmark))
        return face_mesh_results

    def _calc_landmarks(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]

        landmark_list = []
        for _, landmark in enumerate(landmarks):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)

            landmark_list.append((landmark_x, landmark_y, landmark.z,
                                  landmark.visibility, landmark.presence))
        return landmark_list


    def _calc_bounding_rect(self, landmarks):
        landmark_array = np.empty((0, 2), int)

        for _, landmark in enumerate(landmarks):
            landmark_x = int(landmark[0])
            landmark_y = int(landmark[1])

            landmark_point = [np.array((landmark_x, landmark_y))]
            landmark_array = np.append(landmark_array, landmark_point, axis=0)

        x, y, w, h = cv.boundingRect(landmark_array)

        return [x, y, x + w, y + h]

    def get_eye_landmarks(self, landmarks):

        left_eye_landmarks = []
        right_eye_landmarks = []

        if len(landmarks) > 0:
            # https://github.com/tensorflow/tfjs-models/blob/master/facemesh/mesh_map.jpg

            left_eye_landmarks.append((landmarks[133][0], landmarks[133][1]))
            left_eye_landmarks.append((landmarks[173][0], landmarks[173][1]))
            left_eye_landmarks.append((landmarks[157][0], landmarks[157][1]))
            left_eye_landmarks.append((landmarks[158][0], landmarks[158][1]))
            left_eye_landmarks.append((landmarks[159][0], landmarks[159][1]))
            left_eye_landmarks.append((landmarks[160][0], landmarks[160][1]))
            left_eye_landmarks.append((landmarks[161][0], landmarks[161][1]))
            left_eye_landmarks.append((landmarks[246][0], landmarks[246][1]))
            left_eye_landmarks.append((landmarks[163][0], landmarks[163][1]))
            left_eye_landmarks.append((landmarks[144][0], landmarks[144][1]))
            left_eye_landmarks.append((landmarks[145][0], landmarks[145][1]))
            left_eye_landmarks.append((landmarks[153][0], landmarks[153][1]))
            left_eye_landmarks.append((landmarks[154][0], landmarks[154][1]))
            left_eye_landmarks.append((landmarks[155][0], landmarks[155][1]))

            # 右目
            right_eye_landmarks.append((landmarks[362][0], landmarks[362][1]))
            right_eye_landmarks.append((landmarks[398][0], landmarks[398][1]))
            right_eye_landmarks.append((landmarks[384][0], landmarks[384][1]))
            right_eye_landmarks.append((landmarks[385][0], landmarks[385][1]))
            right_eye_landmarks.append((landmarks[386][0], landmarks[386][1]))
            right_eye_landmarks.append((landmarks[387][0], landmarks[387][1]))
            right_eye_landmarks.append((landmarks[388][0], landmarks[388][1]))
            right_eye_landmarks.append((landmarks[466][0], landmarks[466][1]))
            right_eye_landmarks.append((landmarks[390][0], landmarks[390][1]))
            right_eye_landmarks.append((landmarks[373][0], landmarks[373][1]))
            right_eye_landmarks.append((landmarks[374][0], landmarks[374][1]))
            right_eye_landmarks.append((landmarks[380][0], landmarks[380][1]))
            right_eye_landmarks.append((landmarks[381][0], landmarks[381][1]))
            right_eye_landmarks.append((landmarks[382][0], landmarks[382][1]))

        return left_eye_landmarks, right_eye_landmarks

    #Tính 3 đường thẳng:
    def get_temple_landmarks(self, image, landmarks):
        if len(landmarks) > 0:
            left_temple = landmarks[46][0:3]
            right_temple = landmarks[276][0:3]

        temple_distance = sqrt((left_temple[0]-right_temple[0])**2 + (left_temple[1]-right_temple[1])**2)

        return left_temple, right_temple, temple_distance

    def get_inner_eyetail_landmarks(self, image, landmarks):
        if len(landmarks) > 0:
            left_inner_eyetail = landmarks[243][0:3]
            right_inner_eyetail = landmarks[463][0:3]

        inner_eyetail_distance = sqrt((left_inner_eyetail[0]-right_inner_eyetail[0])**2 + (left_inner_eyetail[1]-right_inner_eyetail[1])**2)

        return left_inner_eyetail, right_inner_eyetail, inner_eyetail_distance

    def get_outer_eyetail_landmarks(self, image, landmarks):
        if len(landmarks) > 0:
            left_outer_eyetail = landmarks[130][0:3]
            right_outer_eyetail = landmarks[359][0:3]

        outer_eyetail_distance = sqrt((left_outer_eyetail[0]-right_outer_eyetail[0])**2 + (left_outer_eyetail[1]-right_outer_eyetail[1])**2)
        return left_outer_eyetail, right_outer_eyetail, outer_eyetail_distance


    ###### lấy landmark để đo shape mặt
    def get_forehead_landmarks(self, image, landmarks):
        if len(landmarks) > 0:
            left_forehead = landmarks[70][0:3]
            right_forehead = landmarks[300][0:3]

        forehead_distance = sqrt((left_forehead[0]-right_forehead[0])**2 + (left_forehead[1]-right_forehead[1])**2)
        return left_forehead, right_forehead, forehead_distance

    def get_cheekbone_landmarks(self, image, landmarks):
        if len(landmarks) > 0:
            left_cheekbone = landmarks[111][0:3]
            right_cheekbone = landmarks[340][0:3]

        cheekbone_distance = sqrt((left_cheekbone[0]-right_cheekbone[0])**2 + (left_cheekbone[1]-right_cheekbone[1])**2)
        return left_cheekbone, right_cheekbone, cheekbone_distance

    def get_jawline_landmarks(self, image, landmarks):
        if len(landmarks) > 0:
            #left_jawline will include middle point
            left_jawline = [landmarks[172][0:3], landmarks[136][0:3], landmarks[150][0:3], landmarks[149][0:3],
                            landmarks[176][0:3], landmarks[148][0:3], landmarks[152][0:3]]
            right_jawline = [landmarks[377][0:3], landmarks[400][0:3], landmarks[378][0:3], landmarks[379][0:3],
                             landmarks[365][0:3], landmarks[397][0:3]]


        return left_jawline, right_jawline

    def get_facelength_landmarks(self, image, landmarks):
        if len(landmarks) > 0:
            down_facelength = landmarks[152][0:3]
            up_facelength = landmarks[10][0:3]

        facelength_distance = sqrt((down_facelength[0]-up_facelength[0])**2 + (down_facelength[1]-up_facelength[1])**2) / 0.87
        return down_facelength, up_facelength, facelength_distance

    def get_headdepth_landmarks(self, image, landmarks):
        if len(landmarks) > 0:
            left_headdepth = landmarks[263][0:3]
            right_headdepth = landmarks[356][0:3]

        headdepth_distance = sqrt((left_headdepth[0]-right_headdepth[0])**2 + (left_headdepth[1]-right_headdepth[1])**2) / 0.7
        return headdepth_distance
    ##################

    def calc_eye_bbox(self, landmarks):


        left_eye_lm, right_eye_lm = self.get_eye_landmarks(landmarks)

        left_eye_bbox = self._calc_bounding_rect(left_eye_lm)
        right_eye_bbox = self._calc_bounding_rect(right_eye_lm)

        return left_eye_bbox, right_eye_bbox

    def calc_around_eye_bbox(self, landmarks, around_ratio=0.5):


        left_eye_bbox, right_eye_bbox = self.calc_eye_bbox(landmarks)

        left_eye_bbox = self._calc_around_eye(left_eye_bbox, around_ratio)
        right_eye_bbox = self._calc_around_eye(right_eye_bbox, around_ratio)

        return left_eye_bbox, right_eye_bbox

    def _calc_around_eye(self, bbox, around_ratio=0.5):
        x1, y1, x2, y2 = bbox
        x = x1
        y = y1
        w = x2 - x1
        h = y2 - y1

        cx = int(x + (w / 2))
        cy = int(y + (h / 2))
        square_length = max(w, h)
        x = int(cx - (square_length / 2))
        y = int(cy - (square_length / 2))
        w = square_length
        h = square_length

        around_ratio = 0.5
        x = int(x - (square_length * around_ratio))
        y = int(y - (square_length * around_ratio))
        w = int(square_length * (1 + (around_ratio * 2)))
        h = int(square_length * (1 + (around_ratio * 2)))

        return [x, y, x + w, y + h]
