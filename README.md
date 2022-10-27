# Multi-layer Algorithm Clustering

## Overview

Multi-layer Algorithm Clustering(MAC) is an algorithm for constructing optimal experimental designs. It can construct optimal design for experimental with large grid size instantly.

## Usage

Import presetting models.

```python
from models.model2 import Model2
from models.model3_positive import Model3
from models.model4 import Model4
from models.model5 import Model5
```

$$
\begin{align*}
    y=& \theta_1&(model 1)\\
    y=& \theta_1e^{\frac {x}{\theta_2}}&(model 2)\\
    y=& \theta_1e^{\pm(x/\theta_2)^{\theta_4}}&(model 3)\\
    y=& \theta_1[\theta_3-(\theta_3-1)e^{(x/\theta_2)}]&(model 4)\\
    y=& \theta_1[\theta_3-(\theta_3-1)e^{(x/\theta_2)^{\theta_4}}]&(model 5)
\end{align*}
$$

Constructing D-optimal design requires users input `design space`, `grid size` and  `initial parameters guesses`. `po_ne` denotes the plus or minus sign for model 3.

```python
from models.model_util import ModelUtil
from models.custom_model import CustomModel
from algorithms.algorithm_util import AlgorithmUtil

model = CustomModel("a*e**(x/b)", ["x"], ["a", "b"], [349.0268, 1067.0434])
restrictions = [[0.01, 2500]]
grid_size = 100
au = AlgorithmUtil(model, restrictions, grid_size)
########################################
au.cocktail_algorithm()
print("criterion_val: ", au.criterion_val)
# criterion_val:  42.53650523036459
########################################
model = Model2(parameters=[349.0268, 1067.0434])
au = AlgorithmUtil(model, restrictions, grid_size)
au.cocktail_algorithm()
print("criterion_val: ", au.criterion_val)
# criterion_val:  42.53650523036449
########################################
restrictions = [[0.01, 1250], [0.01, 1250]]
model = CustomModel("a*e**((x+z)/b)", ["x", "z"], ["a", "b"], [349.0268, 1067.0434])
au = AlgorithmUtil(model, restrictions, grid_size)
au.cocktail_algorithm()
print("criterion_val: ", au.criterion_val)
# criterion_val:  42.536779108194594
```

## Things To Do

* [x] Converting jumbled code to oop code
* [ ] Adding the feature of user-defined models
* [ ] Extend MAC to other optimality criteria.

## License

This software is released under the GNU Lesser Public Liscence V3





[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)