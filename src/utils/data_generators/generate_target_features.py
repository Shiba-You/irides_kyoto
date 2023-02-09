import numpy as np
import pandas as pd

'''
コンポーネントの生成
input :   target_components          ::  <DataFrame>
          target_components_columns  ::  <string[]>         加味する成分の名前の配列
          feature                    ::  <numpy.ndarray>    (n_samples = 31, n_components) : 各サンプルがそれぞれの主成分をどれだけ有しているかを分布する
          X_transformed              ::  <numpy.ndarray>    (n_samples = 31, n_components) : 各サンプルがそれぞれの主成分をどれだけ有しているかを分布する
          group_and_gender           ::  <DataFrame>        (n_samples = 31, 群 + 性別)  
'''
def generate(target_components, target_components_columns, feature, X_transformed, group_and_gender):
  _target_components = []
  for key, val in target_components.items():
    if key == "PC":
      for i in val:
        _target_components.append(feature[:,i].tolist())
    if key == "IC":
      for i in val:
        _target_components.append(X_transformed[:,i].tolist())
  _target_components = np.array(_target_components).T
  target_df = pd.DataFrame(_target_components, columns=target_components_columns)
  target_df = pd.concat([target_df, group_and_gender], axis=1)
  return target_df
