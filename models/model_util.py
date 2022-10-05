# -*- coding: utf-8 -*-
# @Time    : 10/6/22 03:02
# @Author  : godot
# @FileName: model_util.py
# @Project : MAC
# @Software: PyCharm
import numpy as np


class ModelUtil():

    def __init__(self, model, design_space, po_ne, *args):
        self.model = model
        self.design_space = design_space
        self.min_sup_points = model.min_sup_points
        self.inf_mat = np.zeros((self.min_sup_points, self.min_sup_points))
        self.inv_inf_mat = np.zeros((self.min_sup_points, self.min_sup_points))
        self.po_ne = po_ne
        self.args = args


    def set_model(self, model):
        self.model = model
        self.min_sup_points = model.min_sup_points
        self.inf_mat = np.zeros((self.min_sup_points, self.min_sup_points))
        self.inv_inf_mat = np.zeros((self.min_sup_points, self.min_sup_points))

    def set_design_space(self, design_space):
        self.design_space = design_space

    def get_inf_mat(self):
        return self.inf_mat

    def set_parameters(self, parameters):
        self.args = parameters

    def get_inv_inf_mat(self):
        return self.inv_inf_mat

    def get_det_fim(self):
        return np.linalg.det(self.inf_mat)

    def cal_inf_mat(self, designPoints):
        """
        :param designPoints: design points
        :return: information matrix
        """
        self.inf_mat = np.zeros((self.min_sup_points, self.min_sup_points))
        weights = [1 / len(designPoints) for i in range(len(designPoints))]
        for i in range(len(designPoints)):
            self.inf_mat += self.model.par_deriv_vec(designPoints[i], self.po_ne, *self.args) * \
                            self.model.par_deriv_vec(designPoints[i], self.po_ne, *self.args).T * \
                            weights[i]

        self.inv_inf_mat = np.linalg.inv(self.inf_mat)
        return np.array(self.inf_mat)

    def cal_inf_mat_w(self, designPoints):
        """
        :param designPoints: design points
        :return: information matrix
        """
        self.inf_mat = np.zeros((self.min_sup_points, self.min_sup_points))
        for i in range(len(designPoints)):
            self.inf_mat += self.model.par_deriv_vec(designPoints[i][0], self.po_ne, *self.args) * \
                            self.model.par_deriv_vec(designPoints[i][0], self.po_ne, *self.args).T * \
                            designPoints[i][1]
        self.inv_inf_mat = np.linalg.inv(self.inf_mat)
        return np.array(self.inf_mat)

    def cal_variance(self, x):
        left = np.matmul(self.model.par_deriv_vec(x, self.po_ne, *self.args).T, self.inv_inf_mat)
        result = np.matmul(left, self.model.par_deriv_vec(x, self.po_ne, *self.args))
        return result[0][0]

    def cal_combined_variance(self, x_i, x_j):
        return np.matmul(np.matmul(self.model.par_deriv_vec(x_i, self.po_ne, *self.args).T, self.inv_inf_mat),
                         self.model.par_deriv_vec(x_j, self.po_ne, *self.args).T)[0][0]

    def cal_delta(self, x_i, x_j):
        return self.cal_variance(x_j) - \
               (self.cal_variance(x_i) * self.cal_variance(x_j) -
                self.cal_combined_variance(x_i, x_j) *
                self.cal_combined_variance(x_i, x_j)) - \
               self.cal_variance(x_i)

    def cal_observ_mass(self, x):
        return self.model.par_deriv_vec(x, self.po_ne, *self.args) * \
               self.model.par_deriv_vec(x, self.po_ne, *self.args).T

    def fim_add_mass(self, mass, step):
        self.inf_mat = (1 - step) * self.inf_mat + step * mass
        self.inv_inf_mat = np.linalg.inv(self.inf_mat)
        return self.inf_mat
