# -*- coding: utf-8 -*-
# @Time    : 10/29/22 00:03
# @Author  : godot
# @FileName: main.py
# @Project : pyOED
# @Software: PyCharm
import time

import numpy as np

from algorithms.algorithm_util import AlgorithmUtil
from models.custom_model import CustomModel
from models.model5 import Model5
from models.model_util import ModelUtil

if __name__ == '__main__':
    parameters = (349.0268, 1067.0434, 0.7633, 2.6055)
    model = CustomModel("t_1*1+t_2*x_1+t_3*x_2+t_4*x_1**2+t_5*x_2**2+t_6*x_1**3+t_7*x_2**3", ["x_1", "x_2"],
                        ["t_1", "t_2", "t_3", "t_4", "t_5", "t_6", "t_7"],
                        [1, 1, 1, 1, 1, 1, 1])
    # model = Model5([349.0268, 1067.0434, 0.7633, 2.6055])
    restrictions = [[0.01, 20], [0.01, 500]]
    grid_size = 100
    model_util = ModelUtil(model, "D-optimal", restrictions, grid_size)
    au = AlgorithmUtil(model_util, "rex", 1e-6)
    start = time.time()
    au.start()
    print(au.algorithm, time.time() - start)
    print("c_val: ", au.criterion_val)
    print("eff: ", au.eff)
    # print(au.design_points)
    ########################################
