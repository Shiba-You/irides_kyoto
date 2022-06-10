#%%
import numpy as np
import pandas as pd
import graphviz
import lingam
from lingam.utils import make_dot


np.set_printoptions(precision=3, suppress=True)
np.random.seed(100)

#%%

x0 = np.random.uniform(size=1000)
x1 = 3.0*x0 + np.random.uniform(size=1000)
x2 = 6.0*x0 + np.random.uniform(size=1000)
x3 = 3.0*x1 + 2.0*x2 + np.random.uniform(size=1000)
x4 = 4.0*x1 + np.random.uniform(size=1000)
x5 = 8.0*x1 - 1.0*x2 + np.random.uniform(size=1000)
X = pd.DataFrame(np.array([x0, x1, x2, x3, x4, x5]).T ,columns=['x0', 'x1', 'x2', 'x3', 'x4', 'x5'])
X.head()

#%%
m = np.array([
  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  [3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  [6.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  [0.0, 3.0, 2.0, 0.0, 0.0, 0.0],
  [0.0, 4.0, 0.0, 0.0, 0.0, 0.0],
  [0.0, 8.0,-1.0, 0.0, 0.0, 0.0]
])

dot = make_dot(m)

# # Save pdf
# dot.render('dag')

# # Save png
# dot.format = 'png'
# dot.render('dag')

dot
#%%
#! LiNGAM で学習開始

model = lingam.DirectLiNGAM()
model.fit(X)

#%%
#! LiNGAM の学習によって発見した  因果関係の順番  を羅列

model.causal_order_
# %%
#! LiNGAM の学習によって発見した  因果の変数間の結合強度  を羅列

model.adjacency_matrix_
# %%
#! 図示
make_dot(model.adjacency_matrix_)
# %%
#! p値によって出現性
p_values = model.get_error_independence_p_values(X)
print(p_values)
# %%
