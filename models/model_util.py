# -*- coding: utf-8 -*-
# @Time    : 10/6/22 03:02
# @Author  : godot
# @FileName: model_util.py
# @Project : pyOED
# @Software: PyCharm
import math

import numpy as np


class ModelUtil():

    def __init__(self, model, criterion, restrictions, grid_size):
        self.model = model
        self.min_sup_points = model.min_sup_points
        self.inf_mat = np.zeros((self.min_sup_points, self.min_sup_points))
        self.inv_inf_mat = np.zeros((self.min_sup_points, self.min_sup_points))
        self.restrictions = restrictions
        self.grid_size = grid_size
        self.candidate_set = self.generate_candidate_set()
        self.criterion = criterion
        if criterion == "D-optimal":
            self.gradient_func = self.cal_variance
            self.cal_ve_alpha_s = self.cal_ve_alpha_s_d
            self.gradient_target = self.get_min_sup_points
            self.get_criterion_val = self.get_det_fim
        elif criterion == 'A-optimal':
            self.gradient_func = self.cal_a
            self.cal_ve_alpha_s = self.cal_ve_alpha_s_a
            self.gradient_target = self.get_tr_inv_fim
            self.get_criterion_val = self.get_tr_inv_fim

    def generate_candidate_set(self):
        candidate_set = []
        factor_grid_sets = []
        low_high_restrictions = self.restrictions
        grid_size = self.grid_size
        for restriction in low_high_restrictions:
            factor_grid_sets.append(np.linspace(restriction[0], restriction[1], grid_size))
        for point in factor_grid_sets[0]:
            candidate_set.append((point,))
        if len(factor_grid_sets) == 1:
            return candidate_set
        for idx in range(1, len(factor_grid_sets)):
            candidate_set = [point + (next_point,) for point in candidate_set for next_point in factor_grid_sets[idx]]
        return candidate_set

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

    def cal_inf_mat(self, design_points):
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


    def cal_inf_mat_w(self, design_points,weights):
        """
        :param designPoints: design points
        :return: information matrix
        """
        self.inf_mat = np.zeros((self.min_sup_points, self.min_sup_points))
        for i in range(len(design_points)):
            self.inf_mat += self.model.par_deriv_vec(design_points[i]) * \
                            self.model.par_deriv_vec(design_points[i]).T * \
                            weights[design_points[i]]
        self.inv_inf_mat = np.linalg.inv(self.inf_mat)
        return np.array(self.inf_mat)
    def cal_variance(self, x):
        left = np.matmul(self.model.par_deriv_vec(x).T, self.inv_inf_mat)
        result = np.matmul(left, self.model.par_deriv_vec(x))
        return result[0][0]

    def get_min_sup_points(self):
        return self.min_sup_points

    def cal_a(self, x):
        # calculate the f(x).T * inv(fim) * inv(fim) * f(x)
        left = np.matmul(self.model.par_deriv_vec(x).T, np.matmul(self.inv_inf_mat, self.inv_inf_mat))
        result = np.matmul(left, self.model.par_deriv_vec(x))
        return result[0][0]

    def cal_ve_alpha_s_d(self, x_add, x_del, w_add, w_del):
        var_add = self.cal_variance(x_add)
        var_del = self.cal_variance(x_del)
        combined_var = self.cal_combined_variance(x_add, x_del)
        res = min(max((var_del - var_add) / (2 * (combined_var ** 2 - var_add * var_del)), -w_add), w_del)
        if np.isnan(res):
            print("nan")
            print(x_add,x_del,w_add,w_del)
            print()
        return res

    def cal_ve_alpha_s_a(self, x_add, x_del, w_add, w_del):
        threshold = 1e-14
        a_add = self.cal_a(x_add)
        a_del = self.cal_a(x_del)
        combined_a = self.cal_combined_a(x_add, x_del)
        d_add = self.cal_variance(x_add)
        d_del = self.cal_variance(x_del)
        combined_d = self.cal_combined_variance(x_add, x_del)
        A = a_add - a_del
        B = 2 * combined_a * combined_d - d_add * a_del - d_del * a_add
        C = d_add - d_del
        D = d_add * d_del - combined_d ** 2
        G = A * D + B * C
        if abs(G) <= threshold and abs(B) > threshold:
            r = -A / (2 * B)
            if r >= -w_add and r <= w_del:
                return r
        if abs(G) > 0:
            r = -(B + math.sqrt(B ** 2 - A * G)) / G
            if r >= -w_add and r <= w_del:
                return r
        if A > threshold:
            return w_del
        if A < -threshold:
            return -w_add
        return 0

    def get_tr_inv_fim(self):
        return np.trace(self.inv_inf_mat)

    def get_d_optimal_delta(self):
        return self.min_sup_points

    def cal_combined_a(self, x_i, x_j):
        return np.matmul(np.matmul(self.model.par_deriv_vec(x_i).T, np.matmul(self.inv_inf_mat, self.inv_inf_mat)),
                         self.model.par_deriv_vec(x_j))[0][0]

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
