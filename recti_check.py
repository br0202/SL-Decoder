import os
import cv2
import numpy as np

IMG_WIDTH = 1280
IMG_HEIGHT = 1024
GET_DISP_MAPS = True
GET_DEPTH_MAPS = False


def rectify(m1, d1, m2, d2, width, height, r, t):
    R1, R2, P1, P2, Q, _roi1, _roi2 = \
    cv2.stereoRectify(cameraMatrix1=m1,
                      distCoeffs1=d1,
                      cameraMatrix2=m2,
                      distCoeffs2=d2,
                      imageSize=(width, height),
                      R=r,
                      T=t,
                      # flags=0,
                      flags=cv2.CALIB_ZERO_DISPARITY + cv2.CALIB_USE_INTRINSIC_GUESS,
                      # flags = cv2.CALIB_ZERO_DISPARITY,
                      alpha=0.0
                      )

    map1_x, map1_y = cv2.initUndistortRectifyMap(
        cameraMatrix=m1,
        distCoeffs=d1,
        R=R1,
        newCameraMatrix=P1,
        size=(width, height),
        m1type=cv2.CV_32FC1)

    map2_x, map2_y = cv2.initUndistortRectifyMap(
        cameraMatrix=m2,
        distCoeffs=d2,
        R=R2,
        newCameraMatrix=P2,
        size=(width, height),
        m1type=cv2.CV_32FC1)

    f = Q[2, 3]
    baseline = 1./Q[3, 2]

    return map1_x, map1_y, map2_x, map2_y, f, baseline, Q


def check_image_sizes(im1, im2, expected_im_width, expected_im_height):
    assert(im1.shape[0] == expected_im_height)
    assert(im1.shape[1] == expected_im_width)
    assert(im2.shape[0] == expected_im_height)
    assert(im2.shape[1] == expected_im_width)


if __name__ == '__main__':
    R = np.array([[0.999994904322419, 0.00184516301646638, -0.00260513006163661],
                  [-0.00183493829621461, 0.999990626236343, 0.00392179052174489],
                  [0.00261234198459220, -0.00391699028464812, 0.999988916366809]]).T
    T = np.array([[-4.85478525514112],
                  [0.293836727184584],
                  [-1.34102340683622]])
    camera1_matrix = np.array([[5822.16202625669, 0, 3251.80127801624],
                               [0, 5860.41440826788, 652.801586792536],
                               [0, 0, 1]])
    camera2_matrix = np.array([[5772.80032182431, 0, 3272.05639167104],
                               [0, 5801.81548414867, 647.280682286567],
                               [0, 0, 1]])
    camera1_distortion = np.array([-0.138789074741545, 0.468445738528183,
                                   -0.0115318825555549, -0.00327806072565552, 1.25080456554691])
    camera2_distortion = np.array([-0.302560781263738, 1.11787883198903,
                                   -0.000827412125481400, -0.00369570311131464, -0.780806685051512])
    height = 3680
    width = 4896
    map1_x, map1_y, map2_x, map2_y, f, baseline, Q = rectify(camera1_matrix, camera1_distortion, camera2_matrix,
                                                             camera2_distortion, width, height, R, T)
    






