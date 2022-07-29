#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import argparse

import cv2 as cv
import numpy as np

from face_mesh.face_mesh import FaceMesh
from iris_landmark.iris_landmark import IrisLandmark

from math import sqrt

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960*2)
    parser.add_argument("--height", help='cap height', type=int, default=540*2)

    parser.add_argument("--max_num_faces", type=int, default=1)
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.7)

    args = parser.parse_args()

    return args


def main():

    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    max_num_faces = args.max_num_faces
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    #############################################################
    face_mesh = FaceMesh(
        max_num_faces,
        min_detection_confidence,
        min_tracking_confidence,
    )
    iris_detector = IrisLandmark()

    while True:

        #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)

        #############################################################
        # Face Mesh
        face_results = face_mesh(image)

        for face_result in face_results:

            ######
            left_eye, right_eye = face_mesh.calc_around_eye_bbox(face_result)

            left_temple, right_temple, temple_distance = face_mesh.get_temple_landmarks(image, face_result)

            left_inner_eyetail, right_inner_eyetail, inner_eyetail_distance = \
                face_mesh.get_inner_eyetail_landmarks(image, face_result)

            left_outer_eyetail, right_outer_eyetail, outer_eyetail_distance =\
                face_mesh.get_outer_eyetail_landmarks(image, face_result)


            ######
            left_iris, right_iris = detect_iris(image, iris_detector, left_eye,
                                                right_eye)

            ######
            left_center, left_radius = calc_min_enc_losingCircle(left_iris)
            right_center, right_radius = calc_min_enc_losingCircle(right_iris)

            ######
            ratio = 5.85 / left_radius
            temple_distance_mm = ratio*temple_distance
            pd_distance_mm = ratio * sqrt((left_center[0]-right_center[0])**2 + (left_center[1]-right_center[1])**2)
            inner_eyetail_distance_mm = ratio*inner_eyetail_distance
            outer_eyetail_distance_mm = ratio*outer_eyetail_distance

            ######
            debug_image = draw_debug_image(
                debug_image,
                left_center,
                left_radius,
                right_center,
                right_radius,
                left_temple,
                right_temple,
                temple_distance_mm,
                pd_distance_mm,
                left_inner_eyetail,
                right_inner_eyetail,
                inner_eyetail_distance_mm,
                left_outer_eyetail,
                right_outer_eyetail,
                outer_eyetail_distance_mm
            )


        #################################################
        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

        #############################################################
        cv.imshow('Meaure Distance Demo', debug_image)

    cap.release()
    cv.destroyAllWindows()

    return


def detect_iris(image, iris_detector, left_eye, right_eye):
    image_width, image_height = image.shape[1], image.shape[0]
    input_shape = iris_detector.get_input_shape()

    ############################
    left_eye_x1 = max(left_eye[0], 0)
    left_eye_y1 = max(left_eye[1], 0)
    left_eye_x2 = min(left_eye[2], image_width)
    left_eye_y2 = min(left_eye[3], image_height)
    left_eye_image = copy.deepcopy(image[left_eye_y1:left_eye_y2,
                                         left_eye_x1:left_eye_x2])
    #
    eye_contour, iris = iris_detector(left_eye_image)

    #
    left_iris = calc_iris_point(left_eye, eye_contour, iris, input_shape)


    right_eye_x1 = max(right_eye[0], 0)
    right_eye_y1 = max(right_eye[1], 0)
    right_eye_x2 = min(right_eye[2], image_width)
    right_eye_y2 = min(right_eye[3], image_height)
    right_eye_image = copy.deepcopy(image[right_eye_y1:right_eye_y2,
                                          right_eye_x1:right_eye_x2])

    eye_contour, iris = iris_detector(right_eye_image)

    right_iris = calc_iris_point(right_eye, eye_contour, iris, input_shape)

    return left_iris, right_iris


def calc_iris_point(eye_bbox, eye_contour, iris, input_shape):
    iris_list = []
    for index in range(5):
        point_x = int(iris[index * 3] *
                      ((eye_bbox[2] - eye_bbox[0]) / input_shape[0]))
        point_y = int(iris[index * 3 + 1] *
                      ((eye_bbox[3] - eye_bbox[1]) / input_shape[1]))
        point_x += eye_bbox[0]
        point_y += eye_bbox[1]

        iris_list.append((point_x, point_y))

    return iris_list


def calc_min_enc_losingCircle(landmark_list):
    center, radius = cv.minEnclosingCircle(np.array(landmark_list))
    center = (int(center[0]), int(center[1]))
    radius = radius

    return center, radius


def draw_debug_image(
    debug_image,
    left_center,
    left_radius,
    right_center,
    right_radius,
    left_temple,
    right_temple,
    temple_distance_mm,
    pd_distance_mm,
    left_inner_eyetail,
    right_inner_eyetail,
    inner_eyetail_distance_mm,
    left_outer_eyetail,
    right_outer_eyetail,
    outer_eyetail_distance_mm
):



    cv.line(debug_image,left_temple[0:2], right_temple[0:2], (0,0,0), 1)
    cv.line(debug_image, left_center, right_center, (255, 0, 0), 1)
    cv.line(debug_image,left_inner_eyetail[0:2], right_inner_eyetail[0:2], (0,255,0), 1)
    cv.line(debug_image,left_outer_eyetail[0:2], right_outer_eyetail[0:2], (0,0,255), 1)


    ####################
    cv.putText(debug_image, f"Temple width: {temple_distance_mm:.2f} mm", (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
    cv.putText(debug_image, f"Actual PD: {pd_distance_mm:.2f} mm", (10, 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv.LINE_AA)
    cv.putText(debug_image, f"Inner Eyetail distance: {inner_eyetail_distance_mm:.2f} mm", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)
    cv.putText(debug_image, f"Outer eyetail distance: {outer_eyetail_distance_mm:.2f} mm", (10, 80), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)

    ####################

    return debug_image


if __name__ == '__main__':
    main()
