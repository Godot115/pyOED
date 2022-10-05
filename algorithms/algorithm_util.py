# -*- coding: utf-8 -*-
# @Time    : 10/6/22 03:41
# @Author  : godot
# @FileName: algorithm_util.py
# @Project : MAC
# @Software: PyCharm
import math
import sys
import time

import numpy as np

from cluster.dbscan_util import dbscan
from models.model2 import Model2
from models.model3 import Model3
from models.model4 import Model4
from models.model5 import Model5
from models.model_util import ModelUtil


# model, design_space, plus_minus_sign, *args
class AlgorithmUtil():
    def __init__(self, model, design_space, grid_size, po_ne, *args):
        self.model = model
        self.design_space = design_space
        self.grid_size = grid_size
        self.model_util = ModelUtil(model, self.design_space, po_ne, *args)
        self.min_sup_points = self.model_util.min_sup_points

    def mac(self):
        algorithm = 'MAC'
        lo_bound = self.design_space[0]
        up_bound = self.design_space[1]
        min_sup_points = self.min_sup_points
        grid_size = self.grid_size
        space_len = up_bound - lo_bound
        model_util = self.model_util

        design_points = [[lo_bound + space_len / (min_sup_points + 1) * (i + 1), 1 / min_sup_points] for i
                         in
                         range(min_sup_points)]
        design_space = np.linspace(lo_bound, up_bound, num=math.ceil(math.sqrt(grid_size)))
        model_util.cal_inf_mat_w(design_points)
        det_fim = model_util.get_det_fim()

        iteration = 0
        start = time.time()
        time_recorder = []
        det_fim_recorder = []
        threshold = 1e-6
        det_fim_gain = float('inf')
        while det_fim_gain / det_fim > threshold:
            iteration += 1
            max_variance = float('-inf')
            max_variance_point = sys.maxsize
            for j in range(len(design_space)):
                dVariance = model_util.cal_variance(design_space[j])
                if dVariance > max_variance:
                    max_variance = dVariance
                    max_variance_point = design_space[j]

            new_design_space = np.linspace(
                max_variance_point - space_len / (math.ceil(math.sqrt(grid_size)) / 2)
                if max_variance_point - space_len / (math.ceil(math.sqrt(grid_size)) / 2) > lo_bound else lo_bound,
                max_variance_point + space_len / (math.ceil(math.sqrt(grid_size)) / 2)
                if max_variance_point + space_len / (math.ceil(math.sqrt(grid_size)) / 2) < up_bound else up_bound,
                num=math.ceil(math.sqrt(grid_size)))

            for k in new_design_space:
                dVariance = model_util.cal_variance(k)
                if dVariance > max_variance:
                    max_variance = dVariance
                    max_variance_point = k

            A_point = model_util.cal_observ_mass(max_variance_point)
            alpha_s = ((max_variance / min_sup_points) - 1) / (max_variance - 1)
            design_points = [[point[0], point[1] * (1 - alpha_s)] for point in design_points]

            if alpha_s > 0:
                design_points.append([max_variance_point, alpha_s])
            elif alpha_s < 0:
                for idx in range(len(design_points)):
                    # design_points[idx][1] *= (1 - alpha_s)
                    if design_points[idx][0] == max_variance_point:
                        design_points[idx][1] += alpha_s
                        break
            model_util.fim_add_mass(A_point, alpha_s)

            if len(design_points) % (min_sup_points * 10) == 0:
                clusters = dbscan(design_points, 1000, lo_bound, up_bound)
                if len(clusters) >= min_sup_points:
                    design_points = []
                    for group in clusters:
                        group_max_variance = float('-inf')
                        group_max_variance_point = sys.maxsize
                        weight = group[1]
                        for point in group[0]:
                            variance = model_util.cal_variance(point)
                            if variance > group_max_variance:
                                group_max_variance = variance
                                group_max_variance_point = point
                        design_points.append([group_max_variance_point, weight])
                    model_util.cal_inf_mat_w(design_points)

                    denominator = 0
                    variances = [0] * len(design_points)
                    for idx in range(len(design_points)):
                        variances[idx] = model_util.cal_variance(design_points[idx][0])
                        denominator += design_points[idx][1] * variances[idx]
                    for idx in range(len(design_points)):
                        design_points[idx][1] *= variances[idx] / denominator
                    model_util.cal_inf_mat_w(design_points)
            det_fim_gain = model_util.get_det_fim() - det_fim

            det_fim = model_util.get_det_fim()

            time_recorder.append(time.time() - start)
            det_fim_recorder.append(det_fim)

        # matplotlib.pyplot.scatter(time_recorder, det_fim_recorder)
        # matplotlib.pyplot.show()
        # print(max_variance)
        # sql = "INSERT INTO iteration(algorithm, \
        #        model, space_points_num, parameters,lowbound,highbound,det_record,time_record,iteration_times,computer) \
        #        VALUES ('%s',  %s,  %s, '%s',%s,%s,'%s','%s',%s,'%s')" % \
        #       (algorithm, model_name, grid, str(args), lo_bound, up_bound,
        #        det_fim_recorder, time_recorder, iteration, COMPUTER_NAME)

        # cursor.execute(sql)
        # db.commit()
        #
        end = time.time()
        # print("*********  modifiedFWA  *************")
        print(det_fim, end - start)
        return [det_fim, end - start]


if __name__ == '__main__':
    model = Model5()
    design_space = [0.01, 2500.0]
    grid_size = 10000
    po_ne = "neg"
    args = (349.02687, 1067.04343, 0.76332, 2.60551)
    au = AlgorithmUtil(model, [0.01, 2500.0], 1000,po_ne, *args)
    start = time.time()
    au.mac()
