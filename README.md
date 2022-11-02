# pyOED

## Overview

pyOED is a python optimal experimental design package
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
model_util = ModelUtil(model, "D-optimal", restrictions, grid_size)
au = AlgorithmUtil(model_util, "rex", 1e-6)
au.start()
print("c_val: ", au.criterion_val)
print("eff: ", au.eff)
# print(au.design_points)
########################################
```

## Things To Do

* [x] Converting jumbled code to oop code
* [x] Adding the feature of user-defined models
* [x] Extend MAC to other optimality criteria.

## License

This software is released under the GNU Lesser Public Liscence V3





[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)