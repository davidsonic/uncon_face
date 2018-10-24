from __future__ import division
import dlib
import glob
import cv2
import os
import numpy as np


def draw_point(img, p, color):
    cv2.circle(img, p, 2, color, -1)


def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def reshape_for_polyline(array):
    return np.array(array, np.int32).reshape((-1, 1, 2))


def init():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    DOWNSAMPLE_RATIO = 4.0
    return detector, predictor, DOWNSAMPLE_RATIO


def get_land_image(landmarks,width, height):
    # here is the bug
    black_image = np.zeros((height, width), np.uint8)
    jaw = reshape_for_polyline(landmarks[0:17])
    left_eyebrow = reshape_for_polyline(landmarks[22:27])
    right_eyebrow = reshape_for_polyline(landmarks[17:22])
    nose_bridge = reshape_for_polyline(landmarks[27:31])
    lower_nose = reshape_for_polyline(landmarks[30:35])
    left_eye = reshape_for_polyline(landmarks[42:48])
    right_eye = reshape_for_polyline(landmarks[36:42])
    outer_lip = reshape_for_polyline(landmarks[48:60])
    inner_lip = reshape_for_polyline(landmarks[60:68])

    color = (255, 255, 255)
    thickness = 3

    cv2.polylines(black_image, [jaw], False, color, thickness)
    cv2.polylines(black_image, [left_eyebrow], False, color, thickness)
    cv2.polylines(black_image, [right_eyebrow], False, color, thickness)
    cv2.polylines(black_image, [nose_bridge], False, color, thickness)
    cv2.polylines(black_image, [lower_nose], True, color, thickness)
    cv2.polylines(black_image, [left_eye], True, color, thickness)
    cv2.polylines(black_image, [right_eye], True, color, thickness)
    cv2.polylines(black_image, [outer_lip], True, color, thickness)
    cv2.polylines(black_image, [inner_lip], True, color, thickness)

    return black_image


def interpolate_land_image(land1,land2):
    assert(len(land1)==len(land2))
    black_image = np.zeros((512,512), np.uint8)
    interp_land=np.zeros((68,2))

    for i,p in enumerate(zip(land1,land2)):
        ori1=p[0]
        ori2=p[1]
        tmp_x=(ori1[0]+ori2[0])/2.0
        tmp_y=(ori1[1]+ori2[1])/2.0
        interp_land[i]=[tmp_x, tmp_y]
        
    jaw = reshape_for_polyline(interp_land[0:17])
    left_eyebrow = reshape_for_polyline(interp_land[22:27])
    right_eyebrow = reshape_for_polyline(interp_land[17:22])
    nose_bridge = reshape_for_polyline(interp_land[27:31])
    lower_nose = reshape_for_polyline(interp_land[30:35])
    left_eye = reshape_for_polyline(interp_land[42:48])
    right_eye = reshape_for_polyline(interp_land[36:42])
    outer_lip = reshape_for_polyline(interp_land[48:60])
    inner_lip = reshape_for_polyline(interp_land[60:68])

    color = (255, 255, 255)
    thickness = 3

    cv2.polylines(black_image, [jaw], False, color, thickness)
    cv2.polylines(black_image, [left_eyebrow], False, color, thickness)
    cv2.polylines(black_image, [right_eyebrow], False, color, thickness)
    cv2.polylines(black_image, [nose_bridge], False, color, thickness)
    cv2.polylines(black_image, [lower_nose], True, color, thickness)
    cv2.polylines(black_image, [left_eye], True, color, thickness)
    cv2.polylines(black_image, [right_eye], True, color, thickness)
    cv2.polylines(black_image, [outer_lip], True, color, thickness)
    cv2.polylines(black_image, [inner_lip], True, color, thickness)
    return black_image

def gen_landmark(file, detector, predictor, DOWNSAMPLE_RATIO):
    frame = cv2.imread(file)
    frame_resize = cv2.resize(frame, None, fx=1 / DOWNSAMPLE_RATIO, fy=1 / DOWNSAMPLE_RATIO)
    gray = cv2.cvtColor(frame_resize, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    black_image = np.zeros(frame.shape, np.uint8)
    landmarks=None
    if len(faces) == 1:
        for face in faces:
            detected_landmarks = predictor(gray, face).parts()
            landmarks = [[p.x * DOWNSAMPLE_RATIO, p.y * DOWNSAMPLE_RATIO] for p in detected_landmarks]

            jaw = reshape_for_polyline(landmarks[0:17])
            left_eyebrow = reshape_for_polyline(landmarks[22:27])
            right_eyebrow = reshape_for_polyline(landmarks[17:22])
            nose_bridge = reshape_for_polyline(landmarks[27:31])
            lower_nose = reshape_for_polyline(landmarks[30:35])
            left_eye = reshape_for_polyline(landmarks[42:48])
            right_eye = reshape_for_polyline(landmarks[36:42])
            outer_lip = reshape_for_polyline(landmarks[48:60])
            inner_lip = reshape_for_polyline(landmarks[60:68])

            color = (255, 255, 255)
            thickness = 3

            cv2.polylines(black_image, [jaw], False, color, thickness)
            cv2.polylines(black_image, [left_eyebrow], False, color, thickness)
            cv2.polylines(black_image, [right_eyebrow], False, color, thickness)
            cv2.polylines(black_image, [nose_bridge], False, color, thickness)
            cv2.polylines(black_image, [lower_nose], True, color, thickness)
            cv2.polylines(black_image, [left_eye], True, color, thickness)
            cv2.polylines(black_image, [right_eye], True, color, thickness)
            cv2.polylines(black_image, [outer_lip], True, color, thickness)
            cv2.polylines(black_image, [inner_lip], True, color, thickness)

    return black_image, landmarks


if __name__ == '__main__':

    # params
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    DOWNSAMPLE_RATIO = 4.0

    # get images
    # root = '/var/www/jiali/static/test/web/original_face'
    # land_root = '/var/www/jiali/static/test/web/landmarks_face'
    # files = os.listdir(root)
    # for file in files:
    #     path = os.path.join(root, file)
    #     black_image,landmarks = gen_landmark(path, detector, predictor, DOWNSAMPLE_RATIO)
    #     if file=='1.png':
    #         print '1.png: {}'.format(landmarks)
    #     cv2.imwrite(os.path.join(land_root, file), black_image)

    filename='comp2_trump.png'
    landname='comp2_trump_land.png'
    black_iamge, landmarks=gen_landmark(filename, detector, predictor, DOWNSAMPLE_RATIO)
    cv2.imwrite(landname, black_iamge)
