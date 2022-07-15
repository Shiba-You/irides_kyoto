import numpy as np
import pandas as pd

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
