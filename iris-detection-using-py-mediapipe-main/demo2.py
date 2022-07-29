#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import argparse

import mediapipe as mp
import cv2 as cv
import numpy as np

from face_mesh.face_mesh import FaceMesh
from iris_landmark.iris_landmark import IrisLandmark

from math import sqrt
from statistics import mean

from PIL import ImageFont, ImageDraw, Image

temple_distance_lst = []
inner_distance_lst = []
outer_distance_lst = []
pd_distance_lst = []
headdepth_distance_lst = []

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

        ################### Bỏ nền và thay ảnh bv
    #     image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    #     image.flags.writeable = False
    #     results = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1).process(image)
    #
    #     image.flags.writeable = True
    #     image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    #
    # #
    #     condition = np.stack(
    #         (results.segmentation_mask,) * 3, axis=-1) > 0.2
    # #
    #     bg_image = cv.imread('CaoThang.png')
    #     bg_image = cv.resize(bg_image, (1280, 720), interpolation=cv.INTER_AREA)
    #     bg_image = cv.GaussianBlur(bg_image, (55, 55), 0)
    # #
    #     debug_image = np.where(condition, image, bg_image)

        #############################################################

        debug_image = copy.deepcopy(image)

        # Face Mesh
        face_results = face_mesh(image)

        for face_result in face_results:

            ###### list the landmark
            left_eye, right_eye = face_mesh.calc_around_eye_bbox(face_result)

            left_temple, right_temple, temple_distance = face_mesh.get_temple_landmarks(image, face_result)

            left_inner_eyetail, right_inner_eyetail, inner_eyetail_distance = \
                face_mesh.get_inner_eyetail_landmarks(image, face_result)

            left_outer_eyetail, right_outer_eyetail, outer_eyetail_distance =\
                face_mesh.get_outer_eyetail_landmarks(image, face_result)

            headdepth_distance = face_mesh.get_headdepth_landmarks(image, face_result)

            left_forehead, right_forehead,forehead_distance = face_mesh.get_forehead_landmarks(image, face_result)

            left_cheekbone, right_cheekbone, cheekbone_distance = face_mesh.get_cheekbone_landmarks(image, face_result)

            down_facelength, up_facelength, facelength_distance = face_mesh.get_facelength_landmarks(image, face_result)

            left_jawline, right_jawline = face_mesh.get_jawline_landmarks(image, face_result)
            jawline_distance = distance_multi(left_jawline, right_jawline)

            ######
            left_iris, right_iris = detect_iris(image, iris_detector, left_eye,
                                                right_eye)

            ######
            left_center, left_radius = calc_min_enc_losingCircle(left_iris)
            right_center, right_radius = calc_min_enc_losingCircle(right_iris)

            ###### caculate distance in mm
            ratio = 5.5 / left_radius
            temple_distance_mm = ratio*temple_distance
            pd_distance_mm = ratio * sqrt((left_center[0]-right_center[0])**2 + (left_center[1]-right_center[1])**2)
            inner_eyetail_distance_mm = ratio * inner_eyetail_distance
            outer_eyetail_distance_mm = ratio * outer_eyetail_distance
            headdepth_distance_mm = ratio * headdepth_distance

            ########   remove jitter
            #remove jitter temple
            if len(temple_distance_lst) == 200:
                temple_distance_lst.pop(0)
            temple_distance_lst.append(temple_distance_mm)
            temple_distance_show = int(mean(temple_distance_lst))

            #remove jitter inner
            if len(inner_distance_lst) == 200:
                inner_distance_lst.pop(0)
            inner_distance_lst.append(inner_eyetail_distance_mm)
            inner_distance_show = int(mean(inner_distance_lst))

            # remove jitter outer
            if len(outer_distance_lst) == 200:
                outer_distance_lst.pop(0)
            outer_distance_lst.append(outer_eyetail_distance_mm)
            outer_distance_show = int(mean(outer_distance_lst))

            # remove jitter pd
            if len(pd_distance_lst) == 200:
                pd_distance_lst.pop(0)
            pd_distance_lst.append(pd_distance_mm)
            pd_distance_show = int(mean(pd_distance_lst))

            # remove jitter headdepth
            if len(headdepth_distance_lst) == 200:
                headdepth_distance_lst.pop(0)
            headdepth_distance_lst.append(headdepth_distance_mm)
            headdepth_distance_show = int(mean(headdepth_distance_lst))

            ###### caculate len width, bridge width, arm length
            lens_width_show = int((outer_distance_show - inner_distance_show) / 2 - 4)
            bridge_width_show = inner_distance_show -8
            arm_length_show = headdepth_distance_show

            ##### Define face_shape:
            face_shape_show = face_shape(forehead_distance, cheekbone_distance, jawline_distance, facelength_distance)
            line1_show = line1(face_shape_show)
            line2_show = line2(face_shape_show)

            ######
            debug_image = draw_debug_image(
                debug_image,
                left_center,
                left_radius,
                right_center,
                right_radius,
                left_temple,
                right_temple,
                temple_distance_show,
                pd_distance_show,
                left_inner_eyetail,
                right_inner_eyetail,
                inner_distance_show,
                left_outer_eyetail,
                right_outer_eyetail,
                outer_distance_show,
                lens_width_show,
                bridge_width_show,
                arm_length_show,
                face_shape_show,
                line1_show,
                line2_show
            )

            font = ImageFont.truetype('ClearSans-Medium.ttf', 15)
            im = Image.fromarray(debug_image)
            draw = ImageDraw.Draw(im)
            draw.text((10, 210), f'{line1_show}', font=font, fill=(0, 0, 0, 0))
            draw.text((10, 230), f'{line2_show}', font=font, fill=(0, 0, 0, 0))
            debug_image = np.array(im)

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
    temple_distance_show,
    pd_distance_show,
    left_inner_eyetail,
    right_inner_eyetail,
    inner_distance_show,
    left_outer_eyetail,
    right_outer_eyetail,
    outer_distance_show,
    lens_width_show,
    bridge_width_show,
    arm_length_show,
    face_shape_show,
    line1_show,
    line2_show
):


    ### Vẽ 4 đường gồm: temple, pd, inner, outer
    # cv.line(debug_image,left_temple[0:2], right_temple[0:2], (0,0,0), 1)
    # cv.line(debug_image, left_center, right_center, (255, 0, 0), 1)
    # cv.line(debug_image,left_inner_eyetail[0:2], right_inner_eyetail[0:2], (0,255,0), 1)
    # cv.line(debug_image,left_outer_eyetail[0:2], right_outer_eyetail[0:2], (0,0,255), 1)




    ####################
    cv.putText(debug_image, f"Temple width: {temple_distance_show} mm", (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv.LINE_AA)
    cv.putText(debug_image, f"Actual PD: {pd_distance_show} mm", (10, 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv.LINE_AA)
    cv.putText(debug_image, f"Inner Eyetail distance: {inner_distance_show} mm", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv.LINE_AA)
    cv.putText(debug_image, f"Outer eyetail distance: {outer_distance_show} mm", (10, 80), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv.LINE_AA)
    cv.putText(debug_image, "----------", (10, 100), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv.LINE_AA)

    cv.putText(debug_image, f"Lens width: {lens_width_show} mm", (10, 120), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)
    cv.putText(debug_image, f"Bridge width: {bridge_width_show} mm", (10, 140), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)
    cv.putText(debug_image, f"Arm length: {arm_length_show} mm", (10, 160), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)
    cv.putText(debug_image, "----------", (10, 180), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)

    cv.putText(debug_image, f"Face Shape: {face_shape_show}", (10, 200), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
    # cv.putText(debug_image, f"{line1_show}", (10, 220), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
    # cv.putText(debug_image, f"{line2_show}", (10, 240), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)

    return debug_image
    ####################

def distance(a,b):
    return sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def distance_multi(a,b):
    s = 0
    for i in range(len(a) - 1):
        s += distance(a[i], a[i+1])
    s += distance(a[len(a)-1], b[0])
    for i in range(len(b)-1):
        s += distance(b[i], b[i+1])
    return s

def face_shape(a,b,c,d):
    if c > b > a:
        return 'Triangle'
    elif a > b:
        return 'Heart'
    elif abs(a-b) <= 20 and abs(b-c) <= 20 and d > a and d > b and d > c:
        return 'Oblong'
    elif abs(b-d) <= 20 and b > a and b > c:
        return 'Round'
    elif d > b > a > c:
        return 'Diamond'
    elif d > b and a > c:
        return 'Oval'
    else:
        return "Undifined"

def line1(a):
    if a == "Triangle":
        return "Chiều dài cằm bạn lớn hơn khoảng cách 2 gò má"
    elif a == "Heart":
        return "Chiều ngang trán lớn hơn khoảng cách 2 gò má"
    elif a == "Oblong":
        return "Chiều dài cằm, trán và gò má gần bằng nhau"
    elif a == "Round":
        return "Khoảng cách 2 gò má bằng chiều dài mặt"
    elif a == "Diamond":
        return "Chiều dài mặt lớn nhất, rồi đến khoảng cách 2 gò má"
    elif a == "Oval":
        return "Chiều dài khuôn mặt bạn lớn hơn khoảng cách 2 gò má"
    else:
        return ""

def line2(a):
    if a == "Triangle":
        return "Khoảng cách 2 gò má lớn hơn chiều ngang trán"
    elif a == "Heart":
        return "Cằm của bạn có xu hướng sang 2 bên"
    elif a == "Oblong":
        return "Chiều dai khuôn mặt bạn là lớn nhất"
    elif a == "Round":
        return "Khoảng cách này lớn hơn chiều dài cằm và bề rộng trán"
    elif a == "Diamond":
        return "Rồi đến bề rộng trán, rồi đến chiều dài cằm"
    elif a == "Oval":
        return "Bề rộng trán lớn hơn chiều dài cằm, cằm tròn"
    else:
        return ""
#########################


if __name__ == '__main__':
    main()
