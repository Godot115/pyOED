# -*- coding: utf-8 -*-
# @Time    : 10/6/22 04:22
# @Author  : godot
# @FileName: model4.py
# @Project : MAC
# @Software: PyCharm


import math

import numpy as np


class Model4():
    def __init__(self):
        self.model_name = "model4"
        self.min_sup_points = 3

    def par_a(self, x, *args):
        a = args[0]
        b = args[1]
        c = args[2]
        return c - (c - 1) * x

    def par_b(self, x, *args):
        a = args[0]
        b = args[1]
        c = args[2]
        return a * x * math.log(x) * (c - 1) / b

    def par_c(self, x, *args):
        a = args[0]
        b = args[1]
        c = args[2]
        return a - a * x

    def par_deriv_vec(self, x, plus_minus_sign, *args):
        """
        :param x: value of the design point
        :return: f(x,Theta).T
        [[∂η(x,Theta) / ∂θ1],
         [∂η(x,Theta) / ∂θ2],
         ..................
         [∂η(x,Theta) / ∂θm]]
        """
        x = math.exp(-x / args[1])
        return np.array([[self.par_a(x, *args),
                          self.par_b(x, *args),
                          self.par_c(x, *args)]]).T
