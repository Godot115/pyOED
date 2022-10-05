# -*- coding: utf-8 -*-
# @Time    : 10/6/22 04:26
# @Author  : godot
# @FileName: model5.py
# @Project : MAC
# @Software: PyCharm


from math import exp
from math import log

import numpy as np


class Model5():
    def __init__(self):
        self.model_name = "model5"
        self.min_sup_points = 4

    def par_a(self, x, *args):
        a = args[0]
        b = args[1]
        c = args[2]
        d = args[3]

        return c - (c - 1) * x

    def par_b(self, x, *args):
        a = args[0]
        b = args[1]
        c = args[2]
        d = args[3]

        return a * d * (-log(x)) * (1 - c) * x / b

    def par_c(self, x, *args):
        a = args[0]
        b = args[1]
        c = args[2]
        d = args[3]

        return a * (1 - x)

    def par_d(self, x, *args):
        a = args[0]
        b = args[1]
        c = args[2]
        d = args[3]
        return -a * (-log(x)) * (1 - c) * x * log((-log(x)) ** (1 / d))

    def par_deriv_vec(self, x, plus_minus_sign, *args):
        """
        :param x: value of the design point
        :return: f(x,Theta).T
        [[∂η(x,Theta) / ∂θ1],
         [∂η(x,Theta) / ∂θ2],
         ..................
         [∂η(x,Theta) / ∂θm]]
        """
        x = exp(-(x / args[1]) ** args[3])
        return np.array([[self.par_a(x, *args),
                          self.par_b(x, *args),
                          self.par_c(x, *args),
                          self.par_d(x, *args)]]).T
