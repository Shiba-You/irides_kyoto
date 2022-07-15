import pandas as pd

'''
A群の平均 > B群の平均 となるように正負を調整
input :   n_components          ::  <string>           加味する components 数
          X_transformed         ::  <numpy.ndarray>    (n_samples = 31, n_components) : 各サンプルがそれぞれの主成分をどれだけ有しているかを分布する
          ica_components        ::  <numpy.ndarray>    (n_components  , 特徴量   = 40) : 各 components がそれぞれの特徴量をどれだけ加味しているか
          feature               ::  <numpy.ndarray>    (n_samples = 31, n_components) : 各サンプルがそれぞれの主成分をどれだけ有しているかを分布する
          pca_components        ::  <numpy.ndarray>    (n_components  , 特徴量   = 40) : 各 components がそれぞれの特徴量をどれだけ加味しているか
          group_and_gender      ::  <pandas.series>    (n_samples = 31, )               :   
output:   X_transformed         ::  <numpy.ndarray>    (n_samples = 31, n_components) : 各サンプルがそれぞれの主成分をどれだけ有しているかを分布する
          ica_components        ::  <numpy.ndarray>    (n_components  , 特徴量   = 40) : 各 components がそれぞれの特徴量をどれだけ加味しているか
          feature               ::  <numpy.ndarray>    (n_samples = 31, n_components) : 各サンプルがそれぞれの主成分をどれだけ有しているかを分布する
          pca_components        ::  <numpy.ndarray>    (n_components  , 特徴量   = 40) : 各 components がそれぞれの特徴量をどれだけ加味しているか
'''
def convert_A_positive(n_components, df, X_transformed, ica_components, feature, pca_components, group_and_gender):
  for i in range(n_components):
    # print("========================================================")
    # print(f"Components: {i+1}")
    A_pca_mean, A_ica_mean, B_pca_mean, B_ica_mean = _count_mean_as_group(i, df, X_transformed, feature, group_and_gender)
    if A_pca_mean < B_pca_mean:
      pca_components[i] *= -1
      feature[:, i] *= -1
    if A_ica_mean < B_ica_mean:
      ica_components[i] *= -1
      X_transformed[:, i] *= -1
  return X_transformed, ica_components, feature, pca_components

def _count_mean_as_group(i, df, X_transformed, feature, group_and_gender):
  A_pca_mean = 0
  A_ica_mean = 0
  B_pca_mean = 0
  B_ica_mean = 0
  for j, group in enumerate(group_and_gender["group"]):
    if group == "A":
      A_pca_mean += feature[j, i]
      A_ica_mean += X_transformed[j, i]
    elif group == "B":
      B_pca_mean += feature[j, i]
      B_ica_mean += X_transformed[j, i]
  A_pca_mean /= group_and_gender["group"].value_counts()["A"]
  A_ica_mean /= group_and_gender["group"].value_counts()["A"]
  B_pca_mean /= group_and_gender["group"].value_counts()["B"]
  B_ica_mean /= group_and_gender["group"].value_counts()["B"]
  # print(f"PCA -- A: {round(A_pca_mean, 8)},  B: {round(B_pca_mean, 8)}, Bigger: {'A' if A_pca_mean > B_pca_mean else 'B'}")
  # print(f"ICA -- A: {round(A_ica_mean, 8)},  B: {round(B_ica_mean, 8)}, Bigger: {'A' if A_ica_mean > B_ica_mean else 'B'}")
  return A_pca_mean, A_ica_mean, B_pca_mean, B_ica_mean

def arange_pca_and_ica(method, feature, X_transformed, gender_and_group):
  if method == "PC":
    df = pd.DataFrame({"component": "PC{}".format(1), "value": feature[:,0], "group": gender_and_group["group"]})
    for i in range(1, 10):
      df_pc = pd.DataFrame({"component": "PC{}".format(i+1), "value": feature[:,i], "group": gender_and_group["group"]})
      df = pd.concat([df, df_pc])
  elif method == "IC":
    df = pd.DataFrame({"component": "IC{}".format(1), "value": X_transformed[:,0], "group": gender_and_group["group"]})
    for i in range(1, 10):
      df_ic = pd.DataFrame({"component": "IC{}".format(i+1), "value": X_transformed[:,i], "group": gender_and_group["group"]})
      df = pd.concat([df, df_ic])
  return df

def _drop_outliers(origin_df, c_f, c_s):
  c_f_q1 = origin_df[c_f].quantile(0.25)                    # 第1四分位数
  c_f_q3 = origin_df[c_f].quantile(0.75)                    # 第3四分位値
  c_f_iqr = c_f_q3 - c_f_q1                                 # 第1四分位値 と 第3四分位値 の範囲
  c_f_lower_limit  = c_f_q1 - 1.5 * c_f_iqr                 # 下限値として、q1 から 1.5 * iqrを引いたもの 
  c_f_upper_limit  = c_f_q3 + 1.5 * c_f_iqr                 # 上限値として、q3 から 1.5 * iqrをたしたもの 
  c_s_q1 = origin_df[c_s].quantile(0.25)                    # 第1四分位数
  c_s_q3 = origin_df[c_s].quantile(0.75)                    # 第3四分位値
  c_s_iqr = c_s_q3 - c_s_q1                                 # 第1四分位値 と 第3四分位値 の範囲
  c_s_lower_limit  = c_s_q1 - 1.5 * c_s_iqr                 # 下限値として、q1 から 1.5 * iqrを引いたもの 
  c_s_upper_limit  = c_s_q3 + 1.5 * c_s_iqr                 # 上限値として、q3 から 1.5 * iqrをたしたもの 
  dropped_df = origin_df.query(f'{c_f_lower_limit} <= {c_f} <= {c_f_upper_limit} & {c_s_lower_limit} <= {c_s} <= {c_s_upper_limit}')
  print("=======================================")
  print(f"c_f: {c_f}, c_s: {c_s}")
  print(origin_df.query(f'not ({c_f_lower_limit} <= {c_f} <= {c_f_upper_limit} & {c_s_lower_limit} <= {c_s} <= {c_s_upper_limit})'))
  dropped_df = pd.DataFrame(dropped_df)
  return dropped_df

def _drop_outlier(origin_df, c):
  c_q1 = origin_df[c].quantile(0.25)                    # 第1四分位数
  c_q3 = origin_df[c].quantile(0.75)                    # 第3四分位値
  c_iqr = c_q3 - c_q1                                 # 第1四分位値 と 第3四分位値 の範囲
  c_lower_limit  = c_q1 - 1.5 * c_iqr                 # 下限値として、q1 から 1.5 * iqrを引いたもの 
  c_upper_limit  = c_q3 + 1.5 * c_iqr                 # 上限値として、q3 から 1.5 * iqrをたしたもの 
  dropped_df = origin_df.query(f'{c_lower_limit} <= {c} <= {c_upper_limit}')
  dropped_df = pd.DataFrame(dropped_df)
  return dropped_df
