# -*- coding: utf-8 -*-
# @Time    : 10/6/22 03:35
# @Author  : godot
# @FileName: model3.py
# @Project : MAC
# @Software: PyCharm
from math import log, exp

import numpy as np


class Model3():
    def __init__(self):
        self.model_name = "model3"
        self.min_sup_points = 3

    def par_a_po(self, x, *args):
        a = args[0]
        b = args[1]
        c = args[2]
        d = args[3]
        return x

    def par_a_ne(self, x, *args):
        a = args[0]
        b = args[1]
        c = args[2]
        d = args[3]
        return x

    def par_b_po(self, x, *args):
        a = args[0]
        b = args[1]
        c = args[2]
        d = args[3]
        return (-a * d * log(x) * x) / b

    def par_b_ne(self, x, *args):
        a = args[0]
        b = args[1]
        c = args[2]
        d = args[3]
        return (-a * d * log(x) * x) / b

    def par_d_po(self, x, *args):
        a = args[0]
        b = args[1]
        c = args[2]
        d = args[3]
        return a * log(x) * x * log(log(x) ** (-d))

    def par_d_ne(self, x, *args):
        a = args[0]
        b = args[1]
        c = args[2]
        d = args[3]
        return a * log(x) * x * log((-log(x)) ** (-d))

    def par_deriv_vec(self, x, po_ne="positive", *args):
        """
        :param x: value of the design point
        :return: f(x,Theta).T
        [[∂η(x,Theta) / ∂θ1],
         [∂η(x,Theta) / ∂θ2],
         ..................
         [∂η(x,Theta) / ∂θm]]
         [∂η(x,Theta) / ∂θm]]
        """
        if po_ne == "positive":
            x = exp((x / args[1]) ** args[3])
            return np.array([[self.par_a_po(x, *args),
                              self.par_b_po(x, *args),
                              self.par_d_po(x, *args)]]).T
        else:
            x = exp(-((x / args[1]) ** args[3]))
            return np.array([[self.par_a_ne(x, *args),
                              self.par_b_ne(x, *args),
                              self.par_d_ne(x, *args)]]).T
