from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA

'''
コンポーネントの生成
input :   n_components          ::  <number>           加味する components 数
          df                    ::  <DataFrame>        元のデータから group, gender を除外して 正規化 & 欠損値処理 した dataFrame
output:   X_transformed         ::  <numpy.ndarray>    (n_samples = 31, n_components) : 各サンプルがそれぞれの主成分をどれだけ有しているかを分布する
          ica_components        ::  <numpy.ndarray>    (n_components  , 特徴量   = 40) : 各 components がそれぞれの特徴量をどれだけ加味しているか
          feature               ::  <numpy.ndarray>    (n_samples = 31, n_components) : 各サンプルがそれぞれの主成分をどれだけ有しているかを分布する
          pca_components        ::  <numpy.ndarray>    (n_components  , 特徴量   = 40) : 各 components がそれぞれの特徴量をどれだけ加味しているか
          ica                   ::  <Custom>           FastICA
          pca                   ::  <Custom>           PCA
'''
def generate(n_components, df):
  ica = FastICA(n_components=n_components, random_state=0)
  # ica = FastICA(random_state=0)
  X_transformed = ica.fit_transform(df)  #? (n_samples, n_features) => (n_samples, n_components): 各サンプルがそれぞれの主成分をどれだけ有しているかを分布する
  ica_components = ica.components_
  pca = PCA(n_components=n_components)
  # pca = PCA()
  pca.fit(df)
  feature = pca.transform(df)            #? (n_samples, n_features) => (n_samples, n_components): 各サンプルがそれぞれの主成分をどれだけ有しているかを分布する
  pca_components = pca.components_
  return X_transformed, ica_components, feature, pca_components, ica, pca