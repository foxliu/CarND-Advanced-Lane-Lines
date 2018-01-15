#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by LiuSonghui on 2018/1/13
import cv2
import numpy as np


class ImageThresh(object):
    def __init__(self, s_thresh=(170, 255), r_thresh=(200, 255), x_thresh=(20, 100), y_thresh=(20, 100),
                 m_thresh=(30, 100), d_thresh=(0.7, 1.3), h_thresh=(15, 100), l_thresh=(90, 255), v_thresh=(90, 255)):
        self.s_thresh = s_thresh
        self.r_thresh = r_thresh
        self.x_thresh = x_thresh
        self.y_thresh = y_thresh
        self.m_thresh = m_thresh
        self.d_thresh = d_thresh
        self.h_thresh = h_thresh
        self.l_thresh = l_thresh
        self.v_thresh = v_thresh

    @staticmethod
    def binary_output(gray_img, abs_sobel, thresh):
        scale_factor = np.max(abs_sobel) / 255
        scaled_sobel = (abs_sobel / scale_factor).astype(np.uint8)
        binary_output = np.zeros_like(gray_img)
        binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        return binary_output

    def abs_sobel_thresh(self, warped_img, orient='x', sobel_kernel=3):
        gray = cv2.cvtColor(warped_img, cv2.COLOR_RGB2GRAY)
        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
            thresh = self.x_thresh
        elif orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
            thresh = self.y_thresh
        else:
            raise ValueError('orient must be "x" or "y"')
        return self.binary_output(gray, abs_sobel, thresh)

    def mag_thresh(self, warped_img, sobel_kernel=3):
        gray = cv2.cvtColor(warped_img, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        sobelxy = np.sqrt(sobelx ** 2 + sobely ** 2)
        return self.binary_output(gray, sobelxy, self.m_thresh)

    def dir_threshold(self, warped_img, sobel_kernel=3):
        gray = cv2.cvtColor(warped_img, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        arcsobel = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary_output = np.zeros_like(gray)
        binary_output[(arcsobel > self.d_thresh[0]) & (arcsobel < self.d_thresh[1])] = 1
        return binary_output

    def combined_sobel(self, warped_img, sobel_kernel=3):
        gradx = self.abs_sobel_thresh(warped_img, orient='x', sobel_kernel=sobel_kernel)
        grady = self.abs_sobel_thresh(warped_img, orient='y', sobel_kernel=sobel_kernel)
        mag_binary = self.mag_thresh(warped_img, sobel_kernel=sobel_kernel)
        dir_binary = self.dir_threshold(warped_img, sobel_kernel=sobel_kernel)

        combin = np.zeros((warped_img.shape[0], warped_img.shape[1]))
        combin[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
        return combin

    @staticmethod
    def ret_binary(channel, thresh):
        binary = np.zeros_like(channel)
        binary[(channel > thresh[0]) & (channel <= thresh[1])] = 1
        return binary

    def h_threshhold(self, warped_img):
        hls = cv2.cvtColor(warped_img, cv2.COLOR_RGB2HLS)
        h_channel = hls[:, :, 0]
        return self.ret_binary(h_channel, self.h_thresh)

    def l_threshhold(self, warped_img):
        hls = cv2.cvtColor(warped_img, cv2.COLOR_RGB2HLS)
        l_channel = hls[:, :, 1]
        return self.ret_binary(l_channel, self.l_thresh)

    def s_threshhold(self, warped_img):
        hls = cv2.cvtColor(warped_img, cv2.COLOR_RGB2HLS)
        s_channel = hls[:, :, 2]
        return self.ret_binary(s_channel, self.s_thresh)

    def v_threshhold(self, warped_img):
        hsv = cv2.cvtColor(warped_img, cv2.COLOR_RGB2HSV)
        v_channel = hsv[:, :, 2]
        return self.ret_binary(v_channel, self.v_thresh)

    def r_threshhold(self, warped_img):
        r_channel = warped_img[:, :, 0]
        return self.ret_binary(r_channel, self.r_thresh)

    def x_threshhold(self, warped_image):
        return self.abs_sobel_thresh(warped_image, 'x')

    def s_dir_threshhold(self, warped_img):
        s_binary = self.s_threshhold(warped_img)
        d_binary = self.dir_threshold(warped_img)
        binary = np.zeros((warped_img.shape[0], warped_img.shape[1]))
        binary[(s_binary == 1) | (d_binary == 1)] = 1
        return binary

    def s_x_threshshold(self, warped_img):
        x_binary = self.abs_sobel_thresh(warped_img, 'x')
        s_binary = self.s_threshhold(warped_img)
        binary = np.zeros((warped_img.shape[0], warped_img.shape[1]))
        binary[(x_binary == 1) | (s_binary == 1)] = 1
        return binary

    def s_r_threshshold(self, warped_img):
        s_binary = self.s_threshhold(warped_img)
        r_binary = self.r_threshhold(warped_img)
        binary = np.zeros((warped_img.shape[0], warped_img.shape[1]))
        binary[(s_binary == 1) | (r_binary == 1)] = 1
        return binary

    def s_x_r_threshshold(self, warped_img):
        x_binary = self.abs_sobel_thresh(warped_img, 'x')
        s_binary = self.s_threshhold(warped_img)
        r_binary = self.r_threshhold(warped_img)
        binary = np.zeros((warped_img.shape[0], warped_img.shape[1]))
        binary[(x_binary == 1) | (s_binary == 1) | (r_binary == 1)] = 1
        return binary

    def auto_choice_r_s_x(self, warped_img):
        """
        use threshsholds are: s_r_threshshold, s_threshshold, r_threshshold, abs_sobel_thresh_x
        """

        def _count(histogram):
            num = len(histogram[histogram >= 20])
            return num

        def _remove_noise(binary):
            h, w = binary.shape
            batch_size_h = 90
            for i in range(0, h, batch_size_h):
                start = i
                end = start + batch_size_h
                histogram = np.sum(binary[start:end, :], axis=1)
                binary[start:end, :][histogram > 600, :] = 0
            return cv2.blur(binary, (5, 5))

        count_numbers = []

        sr_binary = self.s_r_threshshold(warped_img)
        sr_binary = _remove_noise(sr_binary)
        sr_histogram = np.sum(sr_binary, axis=1)
        sr_number = _count(sr_histogram)
        count_numbers.append((sr_number, sr_binary))
        if sr_number < 300:
            r_binary = self.r_threshhold(warped_img)
            r_binary = _remove_noise(r_binary)
            r_histogram = np.sum(r_binary, axis=1)
            r_number = _count(r_histogram)
            count_numbers.append((r_number, r_binary))
            if r_number < 300:
                s_binary = self.s_threshhold(warped_img)
                s_binary = _remove_noise(s_binary)
                s_histogram = np.sum(s_binary, axis=1)
                s_number = _count(s_histogram)
                count_numbers.append((s_number, s_binary))
                if s_number < 300:
                    x_binary = self.abs_sobel_thresh(warped_img, 'x')
                    x_binary = _remove_noise(x_binary)
                    x_histogram = np.sum(x_binary, axis=1)
                    x_number = _count(x_histogram)
                    count_numbers.append((x_number, x_binary))
                    if x_number < 300:
                        count_numbers.sort(key=lambda x: x[0])
                        return count_numbers[-1][1]
                    else:
                        return x_binary
                else:
                    return s_binary
            else:
                return r_binary
        else:
            return sr_binary


if __name__ == '__main__':
    pass
