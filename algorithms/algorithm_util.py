# -*- coding: utf-8 -*-
# @Time    : 10/6/22 03:41
# @Author  : godot
# @FileName: algorithm_util.py
# @Project : MAC
# @Software: PyCharm
import sys
import time

import numpy as np

from models.custom_model import CustomModel
from models.model2 import Model2
from models.model2_test import Model2Test
from models.model3_negative import Model3Negative
from models.model3_positive import Model3Positive
from models.model4 import Model4
from models.model5 import Model5
from models.model_util import ModelUtil


# model, design_space, plus_minus_sign, *args
class AlgorithmUtil():
    def __init__(self, model, low_high_restrictions, grid_size):
        self.model = model
        self.low_high_restrictions = low_high_restrictions
        self.grid_size = grid_size
        self.candidate_set = self.generate_candidate_set(low_high_restrictions, grid_size)
        self.model_util = ModelUtil(model)
        self.min_sup_points = self.model_util.min_sup_points
        self.criterion_val = 0

    def generate_candidate_set(self, low_high_restrictions, grid_size):
        candidate_set = []
        factor_grid_sets = []
        for restriction in low_high_restrictions:
            factor_grid_sets.append(np.linspace(restriction[0], restriction[1], grid_size))
        for point in factor_grid_sets[0]:
            candidate_set.append([point])
        if len(factor_grid_sets) == 1:
            return candidate_set
        for idx in range(len(factor_grid_sets) - 1):
            candidate_set = [point + [next_point] for point in candidate_set for next_point in factor_grid_sets[idx]]
        return candidate_set

    def mac(self):
        algorithm = 'MAC'
        min_sup_points = self.min_sup_points
        model_util = self.model_util

        design_points = [
            [self.candidate_set[int(i * (len(self.candidate_set) - 1) / (min_sup_points - 1))], 1 / min_sup_points] for
            i in
            range(min_sup_points)]
        candidate_set = self.candidate_set
        model_util.cal_inf_mat_w(design_points)
        threshold = 1e-6
        max_variance = float('inf')
        while max_variance - min_sup_points > threshold:
            max_variance = float('-inf')
            max_variance_point = sys.maxsize
            for j in range(len(candidate_set)):
                variance = model_util.cal_variance(candidate_set[j])
                if variance > max_variance:
                    max_variance = variance
                    max_variance_point = candidate_set[j]

            if max_variance - min_sup_points < threshold:
                break
            min_variance = float('inf')
            min_variance_point = sys.maxsize
            for point in design_points:
                variance = model_util.cal_variance(point[0])
                if variance < min_variance:
                    min_variance = variance
                    min_variance_point = point

            delta = (min_variance - max_variance) / (2 * (
                    model_util.cal_combined_variance(max_variance_point,
                                                     min_variance_point[0]) ** 2 - max_variance * min_variance))
            # print(delta, min_variance_point[1], )
            alpha_s = min(delta, min_variance_point[1])
            inserted = False
            for idx in range(len(design_points)):
                if design_points[idx][0] == max_variance_point:
                    design_points[idx][1] += alpha_s
                    inserted = True
                    break
            if not inserted:
                design_points.append([max_variance_point, alpha_s])

            for idx in range(len(design_points)):
                if design_points[idx][0] == min_variance_point[0]:
                    design_points[idx][1] -= alpha_s
                    if design_points[idx][1] <= 0:
                        design_points.pop(idx)
                    break
            #
            design_add = model_util.cal_observ_mass(max_variance_point)
            design_del = model_util.cal_observ_mass(min_variance_point[0])
            model_util.fim_add_mass(design_add, alpha_s)
            model_util.fim_add_mass(design_del, -alpha_s)

            idx = 0
            design_points = sorted(design_points, key=lambda x: x[0], reverse=False)
            while idx < len(design_points) - 1:
                point_1 = design_points[idx]
                point_2 = design_points[idx + 1]
                point_1_var = model_util.cal_variance(point_1[0])
                point_2_var = model_util.cal_variance(point_2[0])
                combined_var = model_util.cal_combined_variance(point_1[0], point_2[0])
                delta = (point_1_var - point_2_var) / (2 * (combined_var ** 2 - point_2_var * point_1_var))
                alpha_s = min(max(delta, -point_2[1]), point_1[1])
                design_points[idx][1] -= alpha_s
                if design_points[idx][1] <= 0:
                    design_points.pop(idx)
                    idx -= 1
                design_points[idx + 1][1] += alpha_s
                if design_points[idx + 1][1] <= 0:
                    design_points.pop(idx + 1)
                    idx -= 1
                design_add = model_util.cal_observ_mass(point_2[0])
                design_del = model_util.cal_observ_mass(point_1[0])
                model_util.fim_add_mass(design_add, alpha_s)
                model_util.fim_add_mass(design_del, -alpha_s)
                idx += 1

            variances = [0] * len(design_points)
            for idx in range(len(design_points)):
                variances[idx] = model_util.cal_variance(design_points[idx][0])
                max_variance = max(variances[idx], max_variance)
            if max_variance - min_sup_points < threshold:
                break
            for idx in range(len(design_points)):
                design_points[idx][1] *= variances[idx] / min_sup_points
            model_util.cal_inf_mat_w(design_points)

        self.design_points = design_points
        self.criterion_val = model_util.get_det_fim()

    def cocktail_algorithm(self):
        algorithm = 'combined_algorithm'
        min_sup_points = self.min_sup_points
        model_util = self.model_util

        design_points = [
            [self.candidate_set[int(i * (len(self.candidate_set) - 1) / (min_sup_points - 1))], 1 / min_sup_points] for
            i in
            range(min_sup_points)]
        candidate_set = self.candidate_set
        min_sup_points = self.min_sup_points
        model_util.cal_inf_mat_w(design_points)
        threshold = 1e-6
        max_variance = float('inf')
        while max_variance - min_sup_points > threshold:
            # VDM step begin
            max_variance = float('-inf')
            max_variance_point = sys.maxsize
            for j in range(len(candidate_set)):
                variance = model_util.cal_variance(candidate_set[j])
                if variance > max_variance:
                    max_variance = variance
                    max_variance_point = candidate_set[j]
            if max_variance - min_sup_points < threshold:
                break
            alpha_s = (max_variance / min_sup_points - 1) / (max_variance - 1)
            design_points = [[point[0], point[1] * (1 - alpha_s)] for point in design_points]
            inserted = False
            for idx in range(len(design_points)):
                if design_points[idx][0] == max_variance_point:
                    design_points[idx][1] += alpha_s
                    inserted = True
                    break
            if not inserted:
                design_points.append([max_variance_point, alpha_s])
            design_add = model_util.cal_observ_mass(max_variance_point)
            model_util.fim_add_mass(design_add, alpha_s)
            # VDM step end

            # NNE step begin
            idx = 0
            design_points = sorted(design_points, key=lambda x: x[0], reverse=False)
            while idx < len(design_points) - 1:
                point_1 = design_points[idx]
                point_2 = design_points[idx + 1]
                point_1_var = model_util.cal_variance(point_1[0])
                point_2_var = model_util.cal_variance(point_2[0])
                combined_var = model_util.cal_combined_variance(point_1[0], point_2[0])
                delta = (point_1_var - point_2_var) / (2 * (combined_var ** 2 - point_2_var * point_1_var))
                alpha_s = min(max(delta, -point_2[1]), point_1[1])
                design_points[idx][1] -= alpha_s
                if design_points[idx][1] <= 0:
                    design_points.pop(idx)
                    idx -= 1
                design_points[idx + 1][1] += alpha_s
                if design_points[idx + 1][1] <= 0:
                    design_points.pop(idx + 1)
                    idx -= 1
                design_add = model_util.cal_observ_mass(point_2[0])
                design_del = model_util.cal_observ_mass(point_1[0])
                model_util.fim_add_mass(design_add, alpha_s)
                model_util.fim_add_mass(design_del, -alpha_s)
                idx += 1
            # NNE step end

            # MA step begin
            denominator = 0
            variances = [0] * len(design_points)
            for idx in range(len(design_points)):
                variances[idx] = model_util.cal_variance(design_points[idx][0])
                denominator += design_points[idx][1] * variances[idx]
                max_variance = max(variances[idx], max_variance)
            if max_variance - min_sup_points < threshold:
                break
            for idx in range(len(design_points)):
                design_points[idx][1] *= variances[idx] / denominator
            model_util.cal_inf_mat_w(design_points)
            # MA step end

        self.design_points = design_points
        self.criterion_val = model_util.get_det_fim()


if __name__ == '__main__':
    parameters = (349.0268, 1067.0434, 0.7633, 2.6055)
    model = CustomModel("a*e**(x/b)", ["x"], ["a", "b"], [349.0268, 1067.0434])
    restrictions = [[0.01, 2500]]
    grid_size = 100
    au = AlgorithmUtil(model, restrictions, grid_size)
    ########################################
    start = time.time()
    au.cocktail_algorithm()
    end = time.time()
    print(end - start)
    # print(math.log(np.linalg.det(au.model_util.inv_inf_mat)))
    print("criterion_val: ", au.criterion_val)
    restrictions = [[0.01, 1250], [0.01, 1250]]
    grid_size = 100
    model = CustomModel("a*e**((x+z)/b)", ["x", "z"], ["a", "b"], [349.0268, 1067.0434])
    au = AlgorithmUtil(model, restrictions, grid_size)
    start = time.time()
    au.cocktail_algorithm()
    end = time.time()
    print(end - start)
    # print(math.log(np.linalg.det(au.model_util.inv_inf_mat)))
    print("criterion_val: ", au.criterion_val)
    model = Model2(parameters=[349.0268, 1067.0434])
    restrictions = [[0.01, 2500]]
    au = AlgorithmUtil(model, restrictions, grid_size)
    au.cocktail_algorithm()
    print("criterion_val: ", au.criterion_val)