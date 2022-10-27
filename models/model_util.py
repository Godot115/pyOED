# -*- coding: utf-8 -*-
# @Time    : 10/6/22 03:02
# @Author  : godot
# @FileName: model_util.py
# @Project : MAC
# @Software: PyCharm
import numpy as np


class ModelUtil():

    def __init__(self, model):
        self.model = model
        self.min_sup_points = model.min_sup_points
        self.inf_mat = np.zeros((self.min_sup_points, self.min_sup_points))
        self.inv_inf_mat = np.zeros((self.min_sup_points, self.min_sup_points))

    def set_model(self, model):
        self.model = model
        self.min_sup_points = model.min_sup_points
        self.inf_mat = np.zeros((self.min_sup_points, self.min_sup_points))
        self.inv_inf_mat = np.zeros((self.min_sup_points, self.min_sup_points))


    def get_inf_mat(self):
        return self.inf_mat

    def get_inv_inf_mat(self):
        return self.inv_inf_mat

    def get_det_fim(self):
        return np.linalg.det(self.inf_mat)


    def cal_inf_mat_w(self, design_points):
        """
        :param designPoints: design points
        :return: information matrix
        """
        self.inf_mat = np.zeros((self.min_sup_points, self.min_sup_points))
        for i in range(len(design_points)):
            self.inf_mat += self.model.par_deriv_vec(design_points[i][0]) * \
                            self.model.par_deriv_vec(design_points[i][0]).T * \
                            design_points[i][1]
        self.inv_inf_mat = np.linalg.inv(self.inf_mat)
        return np.array(self.inf_mat)

    def cal_variance(self, x):
        left = np.matmul(self.model.par_deriv_vec(x).T, self.inv_inf_mat)
        result = np.matmul(left, self.model.par_deriv_vec(x))
        return result[0][0]

    def cal_combined_variance(self, x_i, x_j):
        return np.matmul(np.matmul(self.model.par_deriv_vec(x_i).T, self.inv_inf_mat),
                         self.model.par_deriv_vec(x_j))[0][0]

    def cal_delta(self, x_i, x_j):
        return self.cal_variance(x_j) - \
               (self.cal_variance(x_i) * self.cal_variance(x_j) -
                self.cal_combined_variance(x_i, x_j) *
                self.cal_combined_variance(x_i, x_j)) - \
               self.cal_variance(x_i)

    def cal_observ_mass(self, x):
        return self.model.par_deriv_vec(x) * \
               self.model.par_deriv_vec(x).T

    def fim_add_mass(self, mass, step):
        self.inf_mat = (1 - step) * self.inf_mat + step * mass
        self.inv_inf_mat = np.linalg.inv(self.inf_mat)
