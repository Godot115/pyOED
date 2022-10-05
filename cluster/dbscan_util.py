# -*- coding: utf-8 -*-
# @Time    : 10/6/22 03:00
# @Author  : godot
# @FileName: dbscan_util.py
# @Project : MAC
# @Software: PyCharm

from typing import List

import numpy as np
from sklearn.cluster import DBSCAN




def dbscan(designPoints: List[List[float]], lambda_1: int, lowerBound: float, upperBound: float) -> List[float]:
    space_len = upperBound - lowerBound
    keys = list(set([i[0] for i in designPoints]))
    p_w_dict = dict()
    for key in keys:
        p_w_dict[key] = 0
    for point in designPoints:
        p_w_dict[point[0]] += point[1]
    points = []
    for item in p_w_dict.items():
        points += [item[0]] * int(item[1] * 1000)
    points = np.array(points)
    points = points.reshape(len(points), 1)
    db = DBSCAN(eps=(space_len / 100), min_samples=(lambda_1 / 10)).fit(points)
    labels = db.labels_
    l_p_dict = dict()
    for key in set(labels):
        l_p_dict[key] = []

    for idx in range(len(labels)):
        l_p_dict[labels[idx]].append(points[idx][0])

    res = []
    weight_sum = 0
    for key in l_p_dict.keys():
        if not key == -1:
            l_p_dict[key] = list(set(l_p_dict[key]))
            weight = 0
            for point in l_p_dict[key]:
                weight += p_w_dict[point]
            weight_sum += weight
            res.append([l_p_dict[key], weight])
    res = [[i[0], i[1] / weight_sum] for i in res]
    return res

