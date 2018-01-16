#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by LiuSonghui on 2018/1/13
import numpy as np
import cv2
from line import Line
import traceback
import matplotlib.pyplot as plt


class PreProcess(object):
    def __init__(self, image_thresh, minv, fun_name_list):
        """
        PreProcess line, find the lines and draw picture
        """
        self.image_thresh = image_thresh
        self.Minv = minv
        self.l_line = Line()
        self.r_line = Line()
        self.i = 0
        self.fun_names = fun_name_list

        self.ym_per_pix = 30 / 720  # meters per pixel in y dimension
        self.xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    @staticmethod
    def find_base(binary_image):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_image[binary_image.shape[0] // 2:, :], axis=0)
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        return leftx_base, rightx_base

    def get_binary_image(self, warped_img, fun_name):
        """
        type is the image_thresh function's name
        """
        return getattr(self.image_thresh, fun_name)(warped_img)

    def sliding_windows(self, binary_image, nwindows=9, margin=100, minpix=50):
        """
        Fit a second order polynomial to each
        """
        # Set height of windows
        window_height = np.int(binary_image.shape[0] / nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be update for each window
        leftx_current, rightx_current = self.find_base(binary_image)
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_image.shape[0] - (window + 1) * window_height
            win_y_high = binary_image.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) &
                              (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) &
                               (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(lefty * self.ym_per_pix, leftx * self.xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty * self.ym_per_pix, rightx * self.xm_per_pix, 2)

        return locals()

    def skip_sliding_windows(self, binary_image, margin=100):
        """
        Skip the sliding windows step once you know where the lines are
        """
        nonzero = binary_image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_fit_x = self.l_line.best_fit[0] * (nonzeroy ** 2) + self.l_line.best_fit[1] * nonzeroy + \
                     self.l_line.best_fit[2]
        right_fit_x = self.r_line.best_fit[0] * (nonzeroy ** 2) + self.r_line.best_fit[1] * nonzeroy + \
                      self.r_line.best_fit[2]

        left_lane_inds = ((nonzerox > left_fit_x - margin) & (nonzerox < left_fit_x + margin))
        right_lane_inds = ((nonzerox > right_fit_x - margin) & (nonzerox < right_fit_x + margin))

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each Again
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        left_fit_cr = np.polyfit(lefty * self.ym_per_pix, leftx * self.xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty * self.ym_per_pix, rightx * self.xm_per_pix, 2)

        return locals()

    def add_to_line(self, warped_img, fun_name, margin=100):
        try:
            self.binary_image = self.get_binary_image(warped_img, fun_name)
            if not self.l_line.detected or not self.r_line.detected:
                line_info = self.sliding_windows(self.binary_image, margin=margin)
            else:
                line_info = self.skip_sliding_windows(self.binary_image, margin=margin)

            left_fit = line_info.get('left_fit')
            right_fit = line_info.get('right_fit')
            left_lane_inds = line_info.get('left_lane_inds')
            right_lane_inds = line_info.get('right_lane_inds')
            left_fit_cr = line_info.get('left_fit_cr')
            right_fit_cr = line_info.get('right_fit_cr')

            if left_fit is None or right_fit is None:
                return False, ValueError('not found fit ...')
            self.l_line.add(left_fit, left_lane_inds)
            self.r_line.add(right_fit, right_lane_inds)

            self.l_line.fit_cr = left_fit_cr
            self.r_line.fit_cr = right_fit_cr
            return True, None
        except Exception as ex:
            # print(ex.args)
            return False, ex

    def calc_curv_rad_and_center_dist(self, warped_img, margin=100):
        """
        Method to determine radius of curvature and distance from lane center
        """
        # Todo: 可以在这里添加一个过滤，如果没有找到合适的路线，那就在已经找到中使用最合适的，或是使用上一下图像的
        for fun_name in self.fun_names:
            result, ex = self.add_to_line(warped_img, fun_name, margin=margin)
            if result:
                break
        else:
            print('notfound fun is: ', ex.args)
            # cv2.imwrite('output_images/test_{}.jpg'.format(self.i), warped_img)
            self.i += 1
            self.l_line.current_fit = []
            self.r_line.current_fit = []
            # raise ex

        l_best_fit = self.l_line.best_fit
        r_best_fit = self.r_line.best_fit
        left_fit_cr = self.l_line.fit_cr
        right_fit_cr = self.r_line.fit_cr
        ploty = np.linspace(0, self.binary_image.shape[0] - 1, self.binary_image.shape[0])
        left_fitx = l_best_fit[0] * ploty ** 2 + l_best_fit[1] * ploty + l_best_fit[2]
        right_fitx = r_best_fit[0] * ploty ** 2 + r_best_fit[1] * ploty + r_best_fit[2]

        # Calculate the new radii of curvature
        y_eval = np.max(ploty)
        left_curverad = (
                                (1 + (2 * left_fit_cr[0] * y_eval * self.ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5
                        ) / np.absolute(2 * left_fit_cr[0])
        right_curverad = (
                                 (1 + (2 * right_fit_cr[0] * y_eval * self.ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5
                         ) / np.absolute(2 * right_fit_cr[0])
        # Now our radius of curvature is in meters
        self.l_line.radius_of_curvature = (left_curverad + right_curverad) / 2
        self.r_line.radius_of_curvature = (left_curverad + right_curverad) / 2

        # Calculating the distance between the vehicle distance and the center of the road， > 0 is right else left
        pix_center = self.binary_image.shape[1] / 2 - (left_fitx[-1] + right_fitx[-1]) / 2
        m_center = pix_center * self.xm_per_pix

        self.l_line.line_base_pos = m_center
        self.r_line.line_base_pos = m_center

    def draw_lane(self, src_image):
        new_img = np.copy(src_image)
        l_fit = self.l_line.best_fit
        r_fit = self.r_line.best_fit

        if l_fit is None or r_fit is None:
            return src_image
        h, w = src_image.shape[0], src_image.shape[1]
        # Create an image to draw the lines on
        warp_zero = np.zeros((h, w), dtype=np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        ploty = np.linspace(0, h - 1, num=h)  # to cover same y-range as image
        left_fitx = l_fit[0] * ploty ** 2 + l_fit[1] * ploty + l_fit[2]
        right_fitx = r_fit[0] * ploty ** 2 + r_fit[1] * ploty + r_fit[2]

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 180, 0))
        cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255, 0, 0), thickness=15)
        cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0, 0, 255), thickness=15)

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, self.Minv, (w, h))
        # Combine the result with the original image
        result = cv2.addWeighted(new_img, 1, newwarp, 1, 0)
        return result

    def draw_small(self, src_image, warped_img):
        small_shape = (640, 300)
        small_binary = cv2.resize(self.binary_image, small_shape)
        small_binary_image = np.dstack((small_binary, small_binary, small_binary)) * 255
        small_warp_image = cv2.resize(warped_img, small_shape)
        hmerge = np.hstack((small_warp_image, small_binary_image))
        background = self.draw_lane(src_image)
        background[:300, :, :] = hmerge * 0.8
        return background

    def draw_data(self, src_image, warped_img):
        self.calc_curv_rad_and_center_dist(warped_img)
        new_img = np.copy(self.draw_small(src_image, warped_img))
        font = cv2.FONT_HERSHEY_DUPLEX
        text = 'Curve radius: ' + '{:04.2f}.'.format(self.l_line.radius_of_curvature) + 'm'
        cv2.putText(new_img, text, (40, 350), font, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
        if self.l_line.line_base_pos > 0:
            direction = 'right'
        else:
            direction = 'left'
        abs_center_dist = abs(self.l_line.line_base_pos)
        text = direction + ' of center ' + '{:04.3f}'.format(abs_center_dist) + 'm'
        cv2.putText(new_img, text, (40, 400), font, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
        return new_img


if __name__ == '__main__':
    pass
