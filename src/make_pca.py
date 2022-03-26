#%%
'''
python @ 3.10.2
'''
import itertools
import datetime
import pandas as pd
from pandas import plotting 
import numpy as np
import os.path
import matplotlib
import japanize_matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import sklearn
from sklearn.decomposition import PCA

class make_pca:
  def __init__(self, input_file_name, output_file_name, output_chart_path):
    self.input_file_name = input_file_name
    self.output_file_name = output_file_name
    self.output_chart_path = output_chart_path
    self.df = pd.DataFrame()
    self.dfs = pd.DataFrame()
    self.feature = []
    self.key = []

  def init_data(self):
    if os.path.exists(self.input_file_name):
      xls = pd.ExcelFile(self.input_file_name)
      sheets = xls.sheet_names
      input_sheet = sheets[4]
      self.df = pd.DataFrame(xls.parse(input_sheet))
      self.key = list(self.df.columns)[3:]
    else:
      print("ファイルが存在しません．")
      exit()
    return
  
  def normalize_data(self):
    self.dfs = self.df.iloc[:, 3:].apply(lambda x: (x-x.mean())/x.std(), axis=0)
    for k in self.key:
      self.dfs[k] = self.dfs[k].fillna(self.dfs[k].mean())
  
  def output_to_sheet(self, df, sheet):
    with pd.ExcelWriter(self.output_file_name, mode='a') as writer:
      df.to_excel(writer, sheet_name=sheet)

  def make_scatter(self, col):
    pdf = PdfPages(self.output_chart_path+"_主成分プロット.pdf")
    i = 0
    group = self.df.iloc[:, 1] == "A"
    feature_A = self.feature[[group]]
    feature_B = self.feature[[np.logical_not(group)]]
    for pc_f, pc_s in itertools.combinations(range(len(self.feature)), 2):
      if i%8==0:
        if i != 0:
          pdf.savefig()
          plt.clf()
        i = 0
        f, axs = plt.subplots(2, 4, figsize=(20, 10))
        plt.subplots_adjust(wspace=0.4, hspace=0.8, bottom=0.17, top=0.93)
      for x, y, idx, gender in zip(self.feature[:, pc_f], self.feature[:, pc_s], self.df.iloc[:, 0], self.df.iloc[:, 2]):
        axs[i//4, i%4].text(x, y, str(idx)+"_"+gender, fontsize=5)
      axs[i//4, i%4].scatter(feature_A[:, pc_f], feature_A[:, pc_s], alpha=0.8, s=5, c="red", label="A")
      axs[i//4, i%4].scatter(feature_B[:, pc_f], feature_B[:, pc_s], alpha=0.8, s=5, c="blue", label="B")
      axs[i//4, i%4].legend()
      axs[i//4, i%4].grid()
      axs[i//4, i%4].set_xlabel("PC" + str(pc_f+1))
      axs[i//4, i%4].set_ylabel("PC" + str(pc_s+1))

      i += 1
      if pc_f == len(self.feature)-2 and pc_s == len(self.feature)-1:
        pdf.savefig()
        plt.clf()
    pdf.close()

  def make_ticker(self, pca):
    plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
    plt.plot([0] + list( np.cumsum(pca.explained_variance_ratio_)), "-o")
    for x, y in zip(self.df.iloc[:, 0], [0]+list( np.cumsum(pca.explained_variance_ratio_))):
      plt.text(x-0.5, y+0.02, x)
    plt.xlabel("Number of principal components")
    plt.ylabel("Cumulative contribution rate")
    plt.grid()
    plt.savefig(self.output_chart_path+"_累積寄与率.png")
    plt.show()
  
  def make_relations(self, pca, feature_len):
    pdf = PdfPages(self.output_chart_path+"_寄与度相関.pdf")
    i = 0
    # for pc_f, pc_s in itertools.combinations(range(feature_len), 2):
    for pc_f, pc_s in itertools.combinations(range(8), 2):
      if i%8==0:
        if i != 0:
          pdf.savefig()
          plt.clf()
        i = 0
        f, axs = plt.subplots(2, 4, figsize=(20, 10))
        plt.subplots_adjust(wspace=0.4, hspace=0.8, bottom=0.17, top=0.93)
      
      for x, y, name in zip(pca.components_[pc_f], pca.components_[pc_s], self.dfs.columns):
        axs[i//4, i%4].text(x, y, name, fontsize=5)
      axs[i//4, i%4].scatter(pca.components_[pc_f], pca.components_[pc_s], alpha=0.8, s=5)
      axs[i//4, i%4].grid()
      axs[i//4, i%4].set_xlabel("PC" + str(pc_f+1))
      axs[i//4, i%4].set_ylabel("PC" + str(pc_s+1))

      i += 1
      if pc_f == feature_len-2 and pc_s == feature_len-1:
        pdf.savefig()
        plt.clf()
    pdf.close()
  
  def make_eigenvector_ticker(self, pca):
    pdf = PdfPages(self.output_chart_path+"_固有ベクトルの累積寄与率.pdf")
    i = 0
    for feature_idx in range(len(pca.components_[0])):
      print(i)
      if i%2==0:
        if i != 0:
          pdf.savefig()
          plt.clf()
        i = 0
        f, axs = plt.subplots(1, 2, figsize=(20, 10))
        plt.subplots_adjust(wspace=0.4, hspace=0.8, bottom=0.17, top=0.93)

      contribution_rate = [0]
      for idx, val in enumerate(pca.components_[:, feature_idx]):
        contribution_rate.append(contribution_rate[idx]+val**2)
      axs[i%2].plot(contribution_rate, "-o")
      for x, y in zip(self.df.iloc[:, 0], contribution_rate):
        axs[i%2].text(x-0.5, y+0.02, x)
      axs[i%2].set_xlabel("Number of principal components")
      axs[i%2].set_ylabel("Cumulative contribution rate")
      axs[i%2].grid()
      i += 1
      if feature_idx == len(pca.components_[0])-1:
        pdf.savefig()
        plt.clf()
    pdf.close()

  def calc_pca(self):
    '''
    n_samples     : 被験体数
    n_features    : 特徴量数
    n_components  : 主成分数
    '''
    pca = PCA()
    pca.fit(self.dfs)
    self.feature = pca.transform(self.dfs)         #? (n_samples, n_features) => (n_samples, n_components): 各サンプルがそれぞれの主成分をどれだけ有しているかを分布する
    cols = ["PC{}".format(x + 1) for x in range(len(self.feature))]

    #! 主成分得点
    # df = pd.DataFrame(feature, columns=cols)
    # self.output_to_sheet(df, sheet="主成分得点")

    #! 散布図
    # self.make_scatter(cols)

    #! 寄与率
    # df = pd.DataFrame(pca.explained_variance_ratio_, index=cols)
    # self.output_to_sheet(df, sheet="寄与率")

    #! 累積寄与率
    # self.make_ticker(pca)

    #! 固有値
    # df = pd.DataFrame(pca.explained_variance_, index=cols)
    # self.output_to_sheet(df, sheet="固有値")

    #! 固有ベクトル
    # df = pd.DataFrame(pca.components_, columns=self.dfs.columns, index=cols)
    # self.output_to_sheet(df, sheet="固有ベクトル")
    
    #! 固有ベクトルの散布図
    # self.make_relations(pca, len(self.feature))

    #! 固有ベクトルの累積寄与率
    self.make_eigenvector_ticker(pca)


  def main(self):
    self.init_data()                    #! data/arange から必要データを DataFrame に整形
    self.normalize_data()
    self.calc_pca()


if __name__ == "__main__":
  today = str(datetime.date.today())
  date_format = today[2:4] + today[5:7] + today[8:10]
  #? >>>> ここは変更する >>>>
  input_file_name = "220325 調査報告書+IDs.xlsx"
  output_file_name = date_format + "_result.xlsx"
  output_chart_name = date_format + "_result"
  dir_names = ["04_pickup_params"]
  this_dir = 0                                    #! {0: 04_pickup_params}
  #? <<<< ここは変更する <<<<
  input_file_path = os.path.join("../data/arange", dir_names[this_dir], input_file_name)
  output_file_path = os.path.join("../results/", dir_names[this_dir], output_file_name)
  output_chart_path = os.path.join("../results/", dir_names[this_dir], output_chart_name)
  mp = make_pca(input_file_path, output_file_path, output_chart_path)
  mp.main()


#%%

