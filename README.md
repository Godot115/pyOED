# Multi-layer Algorithm Clustering

## Overview

Multi-layer Algorithm Clustering(MAC) is an algorithm for constructing optimal experimental designs. It can construct optimal design for experimental with large grid size instantly.

## Usage

Import presetting models.

```python
from models.model2 import Model2
from models.model3 import Model3
from models.model4 import Model4
from models.model5 import Model5
```

$$
\begin{align*}
    y=& \theta_1&(model 1)\\
    y=& \theta_1e^{\frac {x}{\theta_2}}&(model 2)\\
    y=& \theta_1e^{\pm(x/\theta_2)^{\theta_4}}&(model 3)\\
    y=& \theta_1[\theta_3-(\theta_3-1)e^{(x/\theta_2)}]&(model 4)\\
    y=& \theta_1[\theta_3-(\theta_3-1)e^{(x/\theta_2)^{\theta_4}}]&(model 5).
\end{align*}
$$

Constructing D-optimal design requires users input `design space`, `grid size` and  `initial parameters guesses`. `po_ne` denotes the plus or minus sign for model 3.

```python
from algorithms.algorithm_util import AlgorithmUtil

model = Model5()
design_space = [0.01, 2500.0]
grid_size = 10000
po_ne = "neg"
args = (349.02687, 1067.04343, 0.76332, 2.60551)
au = AlgorithmUtil(model, [0.01, 2500.0], 1000, po_ne, *args)
au.mac()
print(au.design_points)
# [[2500.0, 0.2499644988426227], [1285.287116935484, 0.24996449884262303], [710.6926411290322, 0.2499644988426224], [0.01, 0.24996449884262267], [1295.3677217741936, 0.00014200462950923057]]

print(au.criterion_val)
# 283.5403685634794
```

## Things To Do

* [x] Converting jumbled code to oop code
* [ ] Adding the feature of user-defined models
* [ ] Extend MAC to other optimality criteria.

## License

This software is released under the GNU Lesser Public Liscence V3





[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)