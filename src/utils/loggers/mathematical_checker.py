import numpy as np

def checker(feature, pca_components, pca, df, n_components):
  origin_data = df.to_numpy()
  sigma = pca.explained_variance_
  print(f"feature: {feature.shape}")
  print(f"pca_components: {pca_components.shape}")
  print(f"pca_components_inv: {np.linalg.pinv(pca_components).shape}")
  print(f"origin_data: {origin_data.shape}")
  print(f"sigma: {sigma.shape}")
  print("====================================================")
  print(f"sigma: {sigma}")

  S = np.cov(origin_data.T)

  WD = np.dot(pca_components.T, np.diag(sigma))
  WDWinv = np.dot(WD, np.linalg.pinv(pca_components.T))

  print(f"S: {S.shape}, WDWinv: {WDWinv.shape}")
  print(S.astype("float32") == WDWinv.astype("float32"))

  # WSigma = np.dot(pca_components.T, np.diag(sigma))
  # print(WSigma.shape)
  # print(origin_data.astype("float32") == WSigma.T.astype("float32"))

  #! 主成分得点(feature) = 元データ(df(= origin_data)) * 固有ベクトル(pca_components)
  # Y = np.dot(origin_data, pca_components.T)
  # dummy_feature = np.zeros_like(feature)
  # print(Y.astype("float32") == feature.astype("float32"))
  # print()
  # print(np.dot(np.dot(pca_components.T, np.diag(sigma)), pca_components).shape)
  # print(np.dot(np.dot(pca_components, np.diag(accumulation_list)), np.linalg.inv(pca_components)))
