import numpy as np
import pandas as pd

'''
コンポーネントの生成
input :   target_components          ::  <DataFrame>
          target_components_columns  ::  <string[]>         加味する成分の名前の配列
          ica                        ::  <Custom>           FastICA
          pca                        ::  <Custom>           PCA
'''
def generate(target_components, target_components_columns, pca, ica):
  _target_c = []
  for key, val in target_components.items():
    if key == "PC":
      for i in val:
        _target_c.append(pca.components_[i].tolist())
    if key == "IC":
      for i in val:
        _target_c.append(ica.components_[i].tolist())
  _target_c = np.array(_target_c).T
  target_df = pd.DataFrame(_target_c, columns=target_components_columns)
  return target_df
