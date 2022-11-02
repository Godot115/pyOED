# -*- coding: utf-8 -*-
# @Time    : 10/6/22 03:41
# @Author  : godot
# @FileName: algorithm_util.py
# @Project : pyOED
# @Software: PyCharm
import heapq
import sys
from random import shuffle
import time
from collections import defaultdict

import numpy as np

from models.custom_model import CustomModel
from models.model2 import Model2
from models.model3_negative import Model3Negative
from models.model3_positive import Model3Positive
from models.model4 import Model4
from models.model5 import Model5
from models.model_util import ModelUtil


# model, design_space, plus_minus_sign, *args
class AlgorithmUtil():
    def __init__(self, model_util, algorithm, eff_threshold):
        self.model_util = model_util
        self.candidate_set = model_util.candidate_set
        self.min_sup_points = model_util.min_sup_points
        self.criterion_val = 0
        self.eff = 0
        self.algorithm = algorithm
        self.eff_threshold = eff_threshold
        if algorithm == "rex":
            self.selected_algorithm = self.rex
        elif algorithm == "cocktail":
            self.selected_algorithm = self.cocktail_algorithm

    def start(self):
        self.selected_algorithm()

    def cocktail_algorithm(self):
        candidate_set = self.candidate_set
        min_sup_points = self.min_sup_points
        model_util = self.model_util
        gradient_func = model_util.gradient_func
        gradient_target = model_util.gradient_target
        weights = defaultdict(lambda: 0)
        design_points = [candidate_set[int(i * (len(candidate_set) - 1) / (min_sup_points - 1))] for i in
                         range(min_sup_points)]
        for point in design_points:
            weights[point] += (1 / len(design_points))
        model_util.cal_inf_mat_w(design_points, weights)
        threshold = self.eff_threshold
        # m_dfv = max_direction_function_val
        max_gradient = float('inf')
        iter = min_sup_points
        while 1 - (gradient_target() / max_gradient) > threshold:
            iter += 1
            # VEM step begin
            max_gradient = float('-inf')
            min_gradient = float('inf')
            max_g_point = sys.maxsize
            min_g_point = sys.maxsize
            for j in range(len(candidate_set)):
                gradient = gradient_func(candidate_set[j])
                if gradient > max_gradient:
                    max_gradient = gradient
                    max_g_point = candidate_set[j]
            for j in range(len(design_points)):
                gradient = gradient_func(design_points[j])
                if gradient < min_gradient:
                    min_gradient = gradient
                    min_g_point = design_points[j]
            if 1 - (gradient_target() / max_gradient) < threshold:
                break
            w_add = weights[max_g_point]
            w_del = weights[min_g_point]
            alpha_s = model_util.cal_ve_alpha_s(max_g_point, min_g_point, w_add, w_del)
            weights[max_g_point] += alpha_s
            weights[min_g_point] -= alpha_s
            if weights[min_g_point] <= 0:
                if max_g_point in design_points:
                    design_points.pop(design_points.index(max_g_point))
                weights.pop(min_g_point)
            if max_g_point not in design_points:
                design_points.append(max_g_point)
            design_add = model_util.cal_observ_mass(max_g_point)
            model_util.fim_add_mass(design_add, alpha_s)
            design_del = model_util.cal_observ_mass(min_g_point)
            model_util.fim_add_mass(design_del, -alpha_s)
            # VEM step end
            # NNE step begin
            idx = 0
            design_points = sorted(design_points, key=lambda x: x, reverse=False)
            while idx < len(design_points) - 1:
                point_1 = design_points[idx]
                point_2 = design_points[idx + 1]
                weight_1 = weights[point_1]
                weight_2 = weights[point_2]
                alpha_s = model_util.cal_ve_alpha_s(point_2, point_1, weight_1, weight_2)
                alpha_s = min(max(alpha_s, -weight_2), weight_1)
                weights[point_1] -= alpha_s
                if weights[point_1] <= 0:
                    if point_1 in design_points:
                        design_points.pop(idx)
                    weights.pop(point_1)
                    idx -= 1
                weights[point_2] += alpha_s
                if weights[point_2] <= 0:
                    if point_2 in design_points:
                        design_points.pop(idx + 1)
                    weights.pop(point_2)
                    idx -= 1
                design_add = model_util.cal_observ_mass(point_2)
                design_del = model_util.cal_observ_mass(point_1)
                model_util.fim_add_mass(design_add, alpha_s)
                model_util.fim_add_mass(design_del, -alpha_s)
                idx += 1
            # NNE step end

            # MA step begin
            denominator = 0
            gradients = [0] * len(design_points)
            for idx in range(len(design_points)):
                gradients[idx] = gradient_func(design_points[idx])
                max_gradient = max(gradients[idx], max_gradient)
                denominator += weights[design_points[idx]] * gradients[idx]
            if 1 - (gradient_target() / max_gradient) < threshold:
                break
            for idx in range(len(design_points)):
                weights[design_points[idx]] *= gradients[idx] / denominator
            model_util.cal_inf_mat_w(design_points, weights)
            # MA step end
        self.design_points = [(point, weights[point]) for point in design_points]
        self.criterion_val = model_util.get_criterion_val()
        self.eff = gradient_target() / max_gradient

    def rex(self):
        candidate_set = self.candidate_set
        min_sup_points = self.min_sup_points
        model_util = self.model_util
        gradient_func = model_util.gradient_func
        gradient_target = model_util.gradient_target
        weights = defaultdict(lambda: 0)
        design_points = [candidate_set[int(i * (len(candidate_set) - 1) / (min_sup_points - 1))] for i in
                         range(min_sup_points)]
        for point in design_points:
            weights[point] += (1 / len(design_points))
        model_util.cal_inf_mat_w(design_points, weights)
        threshold = self.eff_threshold
        del_threshold = 1e-14
        gamma = 5 / min_sup_points

        max_gradient = float('-inf')
        min_gradient = float('inf')
        max_g_point = sys.maxsize
        min_g_point = sys.maxsize
        gradients = defaultdict(lambda: float('inf'))
        heap = []
        for point in candidate_set:
            gradient = gradient_func(point)
            if len(heap) < gamma * min_sup_points:
                heapq.heappush(heap, (gradient, point))
            else:
                if gradient > gradients[heap[0][1]]:
                    heapq.heappop(heap)
                    heapq.heappush(heap, (gradient, point))
            if gradient > max_gradient:
                max_gradient = gradient
                max_g_point = point
            if gradient < min_gradient and point in design_points:
                min_gradient = gradient
                min_g_point = point
            gradients[point] = gradient

        while 1 - (gradient_target() / max_gradient) > threshold:
            w_add = weights[max_g_point]
            w_del = weights[min_g_point]
            alpha_s = model_util.cal_ve_alpha_s(max_g_point, min_g_point, w_add, w_del)
            weights[max_g_point] += alpha_s
            weights[min_g_point] -= alpha_s
            if weights[min_g_point] <= del_threshold:
                if max_g_point in design_points:
                    design_points.pop(design_points.index(max_g_point))
                weights.pop(min_g_point)
            if max_g_point not in design_points:
                design_points.append(max_g_point)
            design_add = model_util.cal_observ_mass(max_g_point)
            model_util.fim_add_mass(design_add, alpha_s)
            design_del = model_util.cal_observ_mass(min_g_point)
            model_util.fim_add_mass(design_del, -alpha_s)
            L = [point[1] for point in heap]
            shuffle(L)
            shuffle(design_points)

            if weights[min_g_point] < del_threshold:
                for l in L:
                    for k in design_points:
                        if k == l:
                            continue
                        alpha = model_util.cal_ve_alpha_s(l, k, weights[l], weights[k])
                        if weights[k] - alpha < del_threshold or weights[l] + alpha < del_threshold:
                            weights[k] -= alpha
                            weights[l] += alpha
                            model_util.fim_add_mass(model_util.cal_observ_mass(l), alpha)
                            model_util.fim_add_mass(model_util.cal_observ_mass(k), -alpha)
            else:
                for l in L:
                    for k in design_points:
                        if k == l:
                            continue
                        alpha = model_util.cal_ve_alpha_s(l, k, weights[l], weights[k])
                        weights[k] -= alpha
                        weights[l] += alpha
                        model_util.fim_add_mass(model_util.cal_observ_mass(l), alpha)
                        model_util.fim_add_mass(model_util.cal_observ_mass(k), -alpha)

            design_points.clear()
            for point in weights.keys():
                if weights[point] > del_threshold:
                    design_points.append(point)
            model_util.cal_inf_mat_w(design_points, weights)

            max_gradient = float('-inf')
            min_gradient = float('inf')
            max_g_point = sys.maxsize
            min_g_point = sys.maxsize
            gradients = defaultdict(lambda: float('inf'))
            heap = []
            for point in candidate_set:
                gradient = gradient_func(point)
                if len(heap) < gamma * min_sup_points:
                    heapq.heappush(heap, (gradient, point))
                else:
                    if gradient > gradients[heap[0][1]]:
                        heapq.heappop(heap)
                        heapq.heappush(heap, (gradient, point))
                if gradient > max_gradient:
                    max_gradient = gradient
                    max_g_point = point
                if gradient < min_gradient and point in design_points:
                    min_gradient = gradient
                    min_g_point = point
                gradients[point] = gradient
        self.design_points = [(point, weights[point]) for point in design_points]
        self.criterion_val = model_util.get_criterion_val()
        self.eff = gradient_target() / max_gradient
