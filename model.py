#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by LiuSonghui on 2018/1/13
import pickle
import cv2
from preprocess import PreProcess
from line import Line
from thresh import ImageThresh
from moviepy.editor import VideoFileClip
import numpy as np
import traceback
import os

camera_wide_dist_file = "camera_cal/wide_dist_pickle.p"


class PipleLine(object):
    def __init__(self, undistort_pickle_file):
        self.undistort_pickle_file = undistort_pickle_file
        self.mtx, self.dist, self.M, self.Minv = self.load_undistort_pickle(self.undistort_pickle_file)

    @staticmethod
    def load_undistort_pickle(undistort_pickle_file):
        # Load coefficient of Correcting for Distortion and coefficient of Perspective Transform
        data = pickle.load(open(undistort_pickle_file, 'rb'))
        return data['mtx'], data['dist'], data['M'], data['Minv']

    def undistort_img(self, img):
        # Correcting for Distortion
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

    def warpperspectiv_img(self, img):
        # Perspective Transform
        undistort = self.undistort_img(img)
        return cv2.warpPerspective(undistort, self.M, undistort.shape[1::-1])

    def unwarpperspectiv_img(self, warped_img):
        return cv2.warpPerspective(warped_img, self.Minv, warped_img.shape[1::-1])


pipline = PipleLine(camera_wide_dist_file)
img_thresh = ImageThresh(r_thresh=(170, 255), s_thresh=(90, 255))
fun_names = ['s_r_threshhold','r_threshshold', 's_threshhold']
pre = PreProcess(img_thresh, pipline.Minv, fun_names)

def process_image(image):
    undis_image = pipline.undistort_img(image)
    warped_image = pipline.warpperspectiv_img(image)
    try:
        result = pre.draw_data(undis_image, warped_image)
        return result
    except Exception as e:
        cv2.imwrite('not_found_line.jpg', image)
        cv2.imwrite('not_found_warp.jpg', warped_image)
        print(traceback.format_exc())


project_output = 'harder_challenge_video_output.mp4'
clip1 = VideoFileClip("harder_challenge_video.mp4")
project_clip = clip1.fl_image(process_image)
project_clip.write_videofile(project_output, audio=False)


if __name__ == '__main__':
    pass
