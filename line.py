#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by LiuSonghui on 2018/1/13
# Define a class to receive the characteristics of each line detection
import numpy as np


class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = []
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        self.fit_cr = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # number of detected pixels
        self.px_count = None

    def add(self, fit, inds):
        # add a found fit to the line, up to n
        if fit is not None:
            if self.best_fit is not None:
                # if we have a best fit, see how this new fit compares
                self.diffs = abs(fit - self.best_fit)
            if (self.diffs[0] > 0.001 or self.diffs[1] > 1.0 or self.diffs[2] > 100.) and len(self.current_fit) > 0:
                # bad fit
                self.detected = False
                raise ValueError('error fit', self.diffs)
            else:
                self.detected = True
                self.px_count = np.count_nonzero(inds)
                self.current_fit.append(fit)
                self.recent_xfitted.append(inds)
                if len(self.current_fit) > 5:
                    # throw out old fits, keep newest n
                    self.current_fit = self.current_fit[-5:]
                    self.recent_xfitted = self.recent_xfitted[-5:]
                # self.best_fit = np.average(self.current_fit, axis=0)
                self.best_fit = fit
        # or remove one from the history, if not found
        else:
            self.detected = False
            if len(self.current_fit) > 0:
                # throw out oldest fit
                self.current_fit = self.current_fit[:-1]
                self.recent_xfitted = self.recent_xfitted[:-1]
            if len(self.current_fit) > 0:
                # if there are still any fits in the queue, best_fit is their average
                self.best_fit = np.average(self.current_fit, axis=0)

    def skip_error(self, fit):
        """
        when to much error skip it
        """
        self.current_fit.append(fit)
        self.current_fit = self.current_fit[1:]
        self.best_fit = np.average(self.current_fit, axis=0)

if __name__ == '__main__':
    pass
