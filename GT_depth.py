import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
from recti_check import rectify
import math
import os
np.set_printoptions(threshold=sys.maxsize)


class GT_depth_from_SL():
    def __init__(self, data_path, round):
        self.data_path = data_path
        self.round = round
        self.images_l = np.zeros((20, 256, 306))
        self.images_r = np.zeros((20, 256, 306))
        self.three_phase = np.zeros((3, 256, 306))
        self.patterns_l = np.ones((10, 256, 306))*0.5
        self.patterns_r = np.ones((10, 256, 306))*0.5
       
        self.R = np.array([[0.9991810103, 0.0056766523, 0.0400635027],
                          [-0.0048003309, 0.9997478588, -0.0219357141],
                          [-0.0401779225, 0.0217254309, 0.9989563255]])
        self.T = np.array([[-1.0065751052], [-0.0018898823], [0.1112990166]])
        self.camera1_matrix = np.array([[4681.3952099281, 0.0000000000, 1501.8525293985],
                                        [0.0000000000, 4723.9576386860, 936.1865248361],
                                        [0.0000000000, 0.0000000000, 1.0000000000]])
        self.camera2_matrix = np.array([[4767.1447471692, 0.0000000000, 1241.0644991558],
                                        [0.0000000000, 4819.5994314859, 1053.2608889591],
                                        [0.0000000000, 0.0000000000, 1.0000000000]])
        self.camera1_distortion = np.array([0.1238203917, -2.1992422424, 0.0000000000, 0.0000000000, 0.0000000000])
        self.camera2_distortion = np.array([-0.0749538579, 0.5059215310, 0.0000000000, 0.0000000000, 0.0000000000])
        self.height = 1024
        self.width = 1224
        self.map1_x, self.map1_y, self.map2_x, self.map2_y, self.f, self.baseline, self.Q = rectify(self.camera1_matrix,
                                                                                                    self.camera1_distortion,
                                                                                                    self.camera2_matrix,
                                                                                                    self.camera2_distortion,
                                                                                                    self.width, self.height,
                                                                                                    self.R, self.T)

    def three_phase_modulation(self):
        self.data_path = self.data_path + '/'
        modulation = {}
        map_x = {'L': self.map1_x, 'R': self.map2_x}
        map_y = {'L': self.map1_y, 'R': self.map2_y}
        for image_side in ['L', 'R']:
            phase_1 = '/00015.jpg'
            im_1 = cv2.imread(self.data_path + image_side + '/' + self.round + phase_1, 0)
            im_1 = cv2.remap(im_1, map_x[image_side], map_y[image_side], cv2.INTER_LINEAR)
            print(self.data_path + image_side + '/' + self.round + phase_1)
            #im_1 = cv2.flip(im_1, 1)
            im_1 = cv2.pyrDown(im_1).astype(float)
            self.three_phase[0, :, :] = cv2.pyrDown(im_1).astype(float)
            phase_2 = '/00017.jpg'
            im_2 = cv2.imread(self.data_path + image_side + '/' + self.round + phase_2, 0)
            im_2 = cv2.remap(im_2, map_x[image_side], map_y[image_side], cv2.INTER_LINEAR)
            #im_2 = cv2.flip(im_2, 1)
            im_2 = cv2.pyrDown(im_2).astype(float)
            self.three_phase[1, :, :] = cv2.pyrDown(im_2).astype(float)
            phase_3 = '/00018.jpg'
            im_3 = cv2.imread(self.data_path + image_side + '/' + self.round + phase_3,0)
            im_3 = cv2.remap(im_3, map_x[image_side], map_y[image_side], cv2.INTER_LINEAR)
            #im_3 = cv2.flip(im_3, 1)
            im_3 = cv2.pyrDown(im_3).astype(float)
            self.three_phase[2, :, :] = cv2.pyrDown(im_3).astype(float)
            modulation[image_side] = (2*np.sqrt(2)/3)*np.sqrt((self.three_phase[0, :, :]-self.three_phase[1, :, :])**2 +
                                                              (self.three_phase[0, :, :]-self.three_phase[2, :, :])**2 +
                                                              (self.three_phase[1, :, :]-self.three_phase[2, :, :])**2)

        return modulation

    def detect_pattern_dg_seperate(self, b=0.75, m=1, plot=True):
        data_path = self.data_path + '/'
        Images = {'L': self.images_l, 'R': self.images_r}
        Patterns = {'L': self.patterns_l, 'R': self.patterns_r}
        map_x = {'L': self.map1_x, 'R': self.map2_x}
        map_y = {'L': self.map1_y, 'R': self.map2_y}
        mask_l = np.zeros((920, 1224))
        mask_r = np.zeros((920, 1224))
        M = {'L': mask_l, 'R': mask_r}
        for image_side in ['L', 'R']:
            for id in range(1, 11, 1):
                pattern_id = '/{:05d}.jpg'.format(2*id-1)
                pattern_inverse_id = '/{:05d}.jpg'.format(2*id)
                SL_image = cv2.imread(data_path + image_side + '/' + self.round + pattern_id, 0)
                SL_image = cv2.remap(SL_image, map_x[image_side], map_y[image_side], cv2.INTER_LINEAR)
                SL_image = cv2.pyrDown(SL_image).astype(float)
                SL_image = cv2.pyrDown(SL_image).astype(float)
                Images[image_side][2*id-2, :, :] = SL_image
                SL_image_inv = cv2.imread(data_path + image_side + '/' + self.round + pattern_inverse_id, 0)
                SL_image_inv = cv2.remap(SL_image_inv, map_x[image_side], map_y[image_side], cv2.INTER_LINEAR)
                SL_image_inv = cv2.pyrDown(SL_image_inv).astype(float)
                SL_image_inv = cv2.pyrDown(SL_image_inv).astype(float)
                Images[image_side][2*id-1, :, :] = SL_image_inv

            I_max = np.max(Images[image_side])
            I_min = np.min(Images[image_side])
            Images[image_side] = (Images[image_side] - I_min)/(I_max - I_min)
            for i in range(0, 20, 2):
                thres = (Images[image_side][i, :, :] + Images[image_side][i+1, :, :])/2
                temp = Images[image_side][i, :, :] - thres
                Patterns[image_side][int(i/2), :, :] = (temp > 0)*1
        modulation = self.three_phase_modulation()
        mask_uncer_l = (modulation['L'] > 20)*1
        mask_uncer_r = (modulation['R'] > 20)*1

        return Patterns, mask_uncer_l, mask_uncer_r

    def decode_gray_code(self, Patterns):
        P_l = Patterns['L'].astype(int)
        P_r = Patterns['R'].astype(int)
        for i in range(1, 10):
            P_l[i, :, :] = np.bitwise_xor(Patterns['L'][i, :, :].astype(int), P_l[i-1, :, :])
            P_r[i, :, :] = np.bitwise_xor(Patterns['R'][i, :, :].astype(int), P_r[i-1, :, :])
        code_l = np.sum((P_l * 2**(np.arange(P_l.shape[0])[::-1]).reshape(10, 1, 1)), axis=0)
        code_r = np.sum((P_r * 2**(np.arange(P_r.shape[0])[::-1]).reshape(10, 1, 1)), axis=0)

        mask_l = ((code_l[:, 1:] - code_l[:, :-1]) != 0.0)*1
        mask_l = np.pad(mask_l, ((0, 0), (1, 0)), 'constant', constant_values=1)
        mask_r = ((code_r[:, 1:] - code_r[:, :-1]) != 0.0)*1
        mask_r = np.pad(mask_r, ((0, 0), (1, 0)), 'constant', constant_values=1)

        return code_l, code_r, mask_l, mask_r

    def interpolarion(self, code_l, code_r, mask_l, mask_r):
        height, width = code_l.shape
        for i in range(height):
            length = 0
            for j in range(width):
                if mask_l[i, j] == 0:
                    length += 1
                    if j == 1223:
                        step = np.linspace(1, length, length)
                        code_l[i, j-length+1::] = code_l[i, j-length+1::] + step
                elif mask_l[i, j] != 0 and j != 0:
                    code_l[i, j-length-1:j+1] = np.linspace(code_l[i, j - length-1], code_l[i, j], length+2)
                    length = 0

        for i in range(height):
            length = 0
            for j in range(width):
                if mask_r[i, j] == 0:
                    length += 1
                    if j == 1223:
                        step = np.linspace(1, length, length)
                        code_r[i, j-length+1::] = code_r[i, j-length+1::] + step
                elif mask_r[i, j] != 0 and j != 0:
                    code_r[i, j-length-1:j+1] = np.linspace(code_r[i, j - length-1], code_r[i, j], length+2)
                    length = 0
        return code_l, code_r

    def stereo_matching(self, code_l, code_r, mask_uncert_l, mask_uncert_r, plot=True):
        disp_l = np.zeros((920, 1224))
        disp_r = np.zeros((920, 1224))
        height, width = code_l.shape
        for i in range(height):
            for j in range(width):
                code_hori_r = code_r[i, :j] * mask_uncert_r[i, :j]
                code_l_unique = (code_l[i, :] == code_l[i, j])*1
                num_code = np.sum(code_l_unique)
                if mask_uncert_l[i, j] == 0:
                    disp_l[i, j] = 0
                elif num_code != 1:
                    disp_l[i, j] = 0
                else:
                    code_matched = (code_hori_r == code_l[i, j]) * 1
                    num_matched = np.sum(code_matched)
                    if num_matched == 1:
                        idx_r = np.where(code_matched == 1)[0][0]
                        disp_l[i, j] = np.abs(j - idx_r)

        for i in range(height):
            for j in range(width):
                code_hori_l = code_l[i, j:] * mask_uncert_l[i, j:]
                code_r_unique = (code_r[i, :] == code_r[i, j])*1
                num_code = np.sum(code_r_unique)
                if mask_uncert_r[i, j] == 0:
                    disp_r[i, j] = 0
                elif num_code != 1:
                    disp_r[i, j] = 0
                else:
                    code_matched = (code_hori_l == code_r[i, j]) * 1
                    num_matched = np.sum(code_matched)
                    if num_matched == 1:
                        idx_l = np.where(code_matched == 1)[0][0]
                        disp_r[i, j] = idx_l


        im_in_l = cv2.imread(self.data_path + 'L' + '/' + self.round + '/00000.jpg', cv2.COLOR_BGR2RGB)
        im_in_r = cv2.imread(self.data_path + 'R' + '/' + self.round + '/00000.jpg', cv2.COLOR_BGR2RGB)
        im_in_l = cv2.remap(im_in_l, self.map1_x, self.map1_y, cv2.INTER_LINEAR)
        #im_in_l = cv2.flip(im_in_l, 1)
        im_in_l = cv2.pyrDown(im_in_l)
        im_in_l = cv2.pyrDown(im_in_l)
        plt.imsave(self.data_path + 'recimgdepth/' + self.round + 'left_rect.jpg', im_in_l[:, :, [2, 1, 0]])

        im_in_r = cv2.remap(im_in_r, self.map2_x, self.map2_y, cv2.INTER_LINEAR)
        #im_in_r = cv2.flip(im_in_r, 1)
        im_in_r = cv2.pyrDown(im_in_r)
        im_in_r = cv2.pyrDown(im_in_r)
        plt.imsave(self.data_path + 'recimgdepth/' + self.round + 'right_rect.jpg', im_in_r[:, :, [2, 1, 0]])

        mask_l = (disp_l == 0)
        mask_d_l = (disp_l != 0)
        points_3d_l = cv2.reprojectImageTo3D(disp_l.astype(np.float32), self.Q)
        depth_l = points_3d_l[:, :, 2]
        # depth = -self.baseline * self.f/(disp*2 + 1e-7)
        depth_l[mask_l] = 0


        plt.subplot(1, 2, 1)
        plt.imshow(depth_l)
        plt.imsave(self.data_path + 'recimgdepth/' + self.round + 'depth_l_GT.jpg', depth_l, vmax=65, vmin=45)
        np.save(self.data_path + 'recimgdepth/' + self.round + 'depth_l_GT.npy', depth_l)
        # plt.imshow(mask_ab, cmap='Reds')
        plt.subplot(1, 2, 2)
        plt.imshow(disp_l)
        #plt.show()
        # plt.imsave(self.data_path + 'depth_abnormal.png', mask_ab, cmap='Reds')
        mask_r = (disp_r == 0)
        mask_d_r = (disp_r != 0)
        points_3d_r = cv2.reprojectImageTo3D(disp_r.astype(np.float32), self.Q)
        depth_r = points_3d_r[:, :, 2]

        # depth = -self.baseline * self.f/(disp*2 + 1e-7)
        depth_r[mask_r] = 0
        plt.subplot(1, 2, 1)
        plt.imshow(depth_r)
        plt.imsave(self.data_path + 'recimgdepth/' + self.round + 'depth_r_GT.jpg', depth_r, vmax=65, vmin=45)
        np.save(self.data_path + 'recimgdepth/' + self.round + 'depth_r_GT.npy', depth_r)
        # plt.imshow(mask_ab, cmap='Reds')
        plt.subplot(1, 2, 2)
        plt.imshow(disp_r)
        #plt.show()
        return depth_l, depth_r, points_3d_l, im_in_r[:, :, [2, 1, 0]], mask_d_l


folder_list = './dataset'    # customer dataset root
for round in sorted(os.listdir(os.path.join(folder_list, 'L'))):
    data_path = folder_list
    depth = GT_depth_from_SL(data_path, round)
    patterns, mask_uncert_l, mask_uncert_r = depth.detect_pattern_dg_seperate()
    print('depth.baseline', depth.baseline)
    print('depth.f', depth.f)
    code_l, code_r, mask_l, mask_r = depth.decode_gray_code(patterns)
    code_l, code_r = depth.interpolarion(code_l.astype(float), code_r.astype(float), mask_l, mask_r)
    d_l, d_r, points, color, mask = depth.stereo_matching(code_l, code_r, mask_uncert_l, mask_uncert_r)

