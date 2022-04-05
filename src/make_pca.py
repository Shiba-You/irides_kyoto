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
from adjustText import adjust_text

class make_pca:
  def __init__(self, input_file_name, output_file_name, output_chart_path):
    self.input_file_name = input_file_name
    self.output_file_name = output_file_name
    self.output_chart_path = output_chart_path
    self.df = pd.DataFrame()
    self.dfs = pd.DataFrame()
    self.feature = []
    self.f_len = 0
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
  
  def calc_lim(self, axs):
    min_x, max_x = axs.get_xlim()
    min_y, max_y = axs.get_ylim()
    dx = (max_x - min_x) / 100
    dy = (max_y - min_y) / 100
    return dx, dy

  def make_scatter(self):
    pdf = PdfPages(self.output_chart_path+"_主成分散布図.pdf")
    i = 0
    group = self.df.iloc[:, 1] == "A"
    feature_A = self.feature[[group]]
    feature_B = self.feature[[np.logical_not(group)]]
    for pc_f, pc_s in itertools.combinations(range(self.f_len), 2):
      if i%2==0:
        if i != 0:
          pdf.savefig()
          plt.clf()
        i = 0
        f, axs = plt.subplots(1, 2, figsize=(16, 8))
        plt.subplots_adjust(wspace=0.4, hspace=0.8, bottom=0.17, top=0.93)
        dx, dy = self.calc_lim(axs[i%2])
      texts = []
      for x, y, idx, gender in zip(self.feature[:, pc_f], self.feature[:, pc_s], self.df.iloc[:, 0], self.df.iloc[:, 2]):
        t = axs[i%2].text(x, y, str(idx)+"_"+gender)
        texts.append(t)
      axs[i%2].scatter(feature_A[:, pc_f], feature_A[:, pc_s], alpha=0.8, s=10, c="red", label="A")
      axs[i%2].scatter(feature_B[:, pc_f], feature_B[:, pc_s], alpha=0.8, s=10, c="blue", label="B")
      axs[i%2].legend()
      axs[i%2].grid()
      axs[i%2].set_xlabel("PC" + str(pc_f+1))
      axs[i%2].set_ylabel("PC" + str(pc_s+1))
      axs[i%2].axis('square')
      adjust_text(texts, ax=axs[i%2], arrowprops=dict(arrowstyle='->', color='m'))

      i += 1
      if pc_f == self.f_len-2 and pc_s == self.f_len-1:
        pdf.savefig()
        plt.clf()
    pdf.close()

  def make_ticker(self, pca):
    pdf = PdfPages(self.output_chart_path+"_累積寄与率.pdf")
    plt.figure(figsize=(20,10))
    axes = plt.gca()
    axes.get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
    dx, dy = self.calc_lim(axes)
    plt.plot([0] + list( np.cumsum(pca.explained_variance_ratio_)), "-o")
    for x, y in zip(self.df.iloc[:, 0], list(np.cumsum(pca.explained_variance_ratio_))):
      plt.text(x-dx, y+dy, x)
    plt.xlabel("Number of principal components")
    plt.ylabel("Cumulative contribution rate")
    plt.grid()
    pdf.savefig()
    plt.clf()
    pdf.close()

  
  def make_relations(self, pca):
    pdf = PdfPages(self.output_chart_path+"_寄与度相関.pdf")
    i = 0
    for pc_f, pc_s in itertools.combinations(range(self.f_len), 2):
      if i%2==0:
        if i != 0:
          pdf.savefig()
          plt.clf()
        i = 0
        f, axs = plt.subplots(1, 2, figsize=(20, 10))
        plt.subplots_adjust(wspace=0.4, hspace=0.8, bottom=0.17, top=0.93)
      texts = []
      for x, y, name in zip(pca.components_[pc_f], pca.components_[pc_s], self.dfs.columns):
        t = axs[i%2].text(x, y, name)
        texts.append(t)
      axs[i%2].scatter(pca.components_[pc_f], pca.components_[pc_s], alpha=0.8, s=10, color="red")
      axs[i%2].grid()
      axs[i%2].set_xlabel("PC" + str(pc_f+1))
      axs[i%2].set_ylabel("PC" + str(pc_s+1))
      adjust_text(texts, ax=axs[i%2], arrowprops=dict(arrowstyle='->', color='m'))

      i += 1
      if pc_f == self.f_len-2 and pc_s == self.f_len-1:
        pdf.savefig()
        plt.clf()
    pdf.close()
  
  def make_eigenvector_ticker(self, pca, n_components, lim=False):
    if lim:
      pdf = PdfPages(self.output_chart_path+"_固有ベクトルの累積寄与率（y軸固定）.pdf")
    else:
      pdf = PdfPages(self.output_chart_path+"_固有ベクトルの累積寄与率.pdf")
    i = 0
    for feature_idx in range(len(pca.components_[0])):
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
      contribution_rate.pop(0)
      components_list = np.arange(n_components)+1
      axs[i%2].plot(components_list, contribution_rate, "-o")
      if lim:
        axs[i%2].set_ylim(0, 1)
      dx, dy = self.calc_lim(axs[i%2])
      for x, y in zip(components_list, contribution_rate):
        axs[i%2].text(x-dx, y+dy, x)
      axs[i%2].set_xlabel("Number of principal components")
      axs[i%2].set_ylabel("Cumulative contribution rate of " + self.df.columns[feature_idx])
      axs[i%2].grid()
      i += 1
      if feature_idx == len(pca.components_[0])-1:
        pdf.savefig()
        plt.clf()
    pdf.close()

  def make_histgran(self):
    pdf = PdfPages(self.output_chart_path+"_ヒストグラム.pdf")
    group = self.df.iloc[:, 1] == "A"
    feature_A = self.feature[[group]]
    feature_B = self.feature[[np.logical_not(group)]]
    for i in range(self.f_len):
      if i%2==0:
        if i != 0:
          pdf.savefig()
          plt.clf()
        i = 0
        f, axs = plt.subplots(1, 2, figsize=(16, 8))
        plt.subplots_adjust(wspace=0.4, hspace=0.8, bottom=0.17, top=0.93)
      axs[i%2].hist(feature_A[:, i], alpha=0.5, label="A")
      axs[i%2].hist(feature_B[:, i], alpha=0.5, label="B")
      # x_lim = np.linspace(min(self.feature[:,i]), max(self.feature[:,i]), 10)
      # axs[i%2].hist(feature_A[:, i], x_lim, alpha=0.5, label="A")
      # axs[i%2].hist(feature_B[:, i], x_lim, alpha=0.5, label="B")
      axs[i%2].grid()
      axs[i%2].axis('square')
    pdf.savefig()
    plt.clf()
    pdf.close()

  def calc_pca(self):
    '''
    n_samples     : 被験体数
    n_features    : 特徴量数
    n_components  : 主成分数
    '''
    pca = PCA(n_components=4)
    pca.fit(self.dfs)
    self.feature = pca.transform(self.dfs)         #? (n_samples, n_features) => (n_samples, n_components): 各サンプルがそれぞれの主成分をどれだけ有しているかを分布する
    self.f_len = len(self.feature[0])
    cols = ["PC{}".format(x + 1) for x in range(self.f_len)]

    # # ! 主成分得点
    # df = pd.DataFrame(self.feature, columns=cols)
    # self.output_to_sheet(df, sheet="主成分得点")

    # #! 寄与率
    # df = pd.DataFrame(pca.explained_variance_ratio_, index=cols)
    # self.output_to_sheet(df, sheet="寄与率")

    # #! 固有値
    # df = pd.DataFrame(pca.explained_variance_, index=cols)
    # self.output_to_sheet(df, sheet="固有値")

    # #! 固有ベクトル
    # df = pd.DataFrame(pca.components_, columns=self.dfs.columns, index=cols)
    # self.output_to_sheet(df, sheet="固有ベクトル")

    # # ! 主成分散布図
    # self.make_scatter()

    # #! 各群による pca ヒストグラム
    self.make_histgran()

    # #! 固有ベクトルの寄与相関
    # self.make_relations(pca)

    # n_components = 20
    # pca = PCA(n_components=n_components)
    # pca.fit(self.dfs)

    # #! 累積寄与率
    # self.make_ticker(pca)

    # #! 固有ベクトルの累積寄与率
    # self.make_eigenvector_ticker(pca, n_components)

    # #! 固有ベクトルの累積寄与率（x軸固定）
    # self.make_eigenvector_ticker(pca, n_components, lim=True)


  def main(self):
    self.init_data()                    #! data/arange から必要データを DataFrame に整形
    self.normalize_data()
    self.calc_pca()


if __name__ == "__main__":
  today = str(datetime.date.today())
  date_format = today[2:4] + today[5:7] + today[8:10]
  #? >>>> ここは変更する >>>>
  input_file_name = "220325 調査報告書+IDs.xlsx"
  output_file_name = date_format + "_集計.xlsx"
  output_chart_name = date_format
  dir_names = ["04_pickup_params"]
  this_dir = 0                                    #! {0: 04_pickup_params}
  #? <<<< ここは変更する <<<<
  input_file_path = os.path.join("../data/arange", dir_names[this_dir], input_file_name)
  output_file_path = os.path.join("../results/", dir_names[this_dir], output_file_name)
  output_chart_path = os.path.join("../results/", dir_names[this_dir], output_chart_name)
  mp = make_pca(input_file_path, output_file_path, output_chart_path)
  mp.main()


#%%
