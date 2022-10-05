# -*- coding: utf-8 -*-
# @Time    : 10/6/22 03:14
# @Author  : godot
# @FileName: model2.py
# @Project : MAC
# @Software: PyCharm
import math

import numpy as np


class Model2():

    def __init__(self):
        self.model_name = "model2"
        self.min_sup_points = 2

    def par_a(self, x, *args):
        a = args[0]
        b = args[1]
        return x

    def par_b(self, x, *args):
        a = args[0]
        b = args[1]
        return -a * b * math.log(x) * x / (b * b)

    def par_deriv_vec(self, x, plus_minus_sign, *args):
        """
        :param x: value of the design point
        :return: f(x,Theta).T
        [[∂η(x,Theta) / ∂θ1],
         [∂η(x,Theta) / ∂θ2],
         ..................
         [∂η(x,Theta) / ∂θm]]
        """
        x = math.exp(x / args[1])
        return np.array([[self.par_a(x, *args),
                          self.par_b(x, *args)]]).T
