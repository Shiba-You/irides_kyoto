#%%
'''
python @ 3.10.2
'''
import itertools
import datetime
from tokenize import group
import pandas as pd
from pandas import plotting 
import numpy as np
import os.path
import matplotlib
import japanize_matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pyparsing import alphas
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import sklearn
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
import openpyxl
import glob
from adjustText import adjust_text

class make_ica:
  def __init__(self, input_file_name, output_file_name, output_chart_path):
    self.input_file_name = input_file_name
    self.output_file_name = output_file_name
    self.output_chart_path = output_chart_path
    self.df = pd.DataFrame()
    self.dfs = pd.DataFrame()
    self.pca_components = []
    self.ica_components = []
    self.X_transformed = []
    self.feature = []
    self.f_len = 0
    self.key = []
    self.target_df = pd.DataFrame()
    self.target_components = ["PC2", "PC4", "PC6", "PC7", "IC1", "IC4", "IC7"]

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
  
  def calc_lim(self, axs):
    min_x, max_x = axs.get_xlim()
    min_y, max_y = axs.get_ylim()
    dx = (max_x - min_x) / 100 * 2
    dy = (max_y - min_y) / 100 * 2
    return dx, dy
  
  def normalize_data(self):
    self.dfs = self.df.iloc[:, 3:].apply(lambda x: (x-x.mean())/x.std(), axis=0)
    for k in self.key:
      self.dfs[k] = self.dfs[k].fillna(self.dfs[k].mean())
  
  def output_to_sheet(self, df, sheet_name):
    if not os.path.isfile(self.output_file_name):
      wb = openpyxl.Workbook()
      sheet = wb.active
      # sheet.title = '_'
      wb.save(self.output_file_name)
      glob.glob("*.xlsx")
    try:
      with pd.ExcelWriter(self.output_file_name, mode='a') as writer:
        print("df    : ", df.shape)
        print("sheet : ", sheet_name)
        print("writer: ", writer)
        df.to_excel(writer, sheet_name=sheet_name)
    except:
      print("既に作成済みです．")
  
  def make_ticker(self, pca):
    pdf = PdfPages(self.output_chart_path+"_累積寄与率.pdf")
    plt.figure(figsize=(20,10))
    axes = plt.gca()
    axes.get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
    dx, dy = self.calc_lim(axes)
    plt.plot([0] + list( np.cumsum(pca.explained_variance_ratio_)), "-o")
    for x, y in zip(self.df.iloc[:, 0], list(np.cumsum(pca.explained_variance_ratio_))):
      if x <= 10:
        print(f"No. : {x, y}")
        print(f"x: {x-dx}")
        print(f"y: {y-dy}")
      plt.text(x-dx, y+dy, x)
    plt.xlabel("Number of principal components")
    plt.ylabel("Cumulative contribution rate")
    plt.grid()
    plt.hlines(0.9, -1, self.f_len+1, "red", linestyles='dashed')
    plt.xlim(-dx*30, self.f_len+dx*30)
    plt.ylim(-dy*3, 1.+dy*3)
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
  
  def make_box_plot(self):
    pdf = PdfPages(self.output_chart_path+"_独立成分箱ひげ図.pdf")
    sns.set(font='IPAexGothic', font_scale = 1)
    sns.set_style("whitegrid")
    temp_df = pd.DataFrame(data=self.X_transformed)
    temp_df["group"] = self.df["群"]
    color_palette = {"A": "#FF8080", "B": "#8080FF"}
    for i in range(self.f_len):
      if i%2==0:
        if i != 0:
          pdf.savefig()
          plt.clf()
        f, axs = plt.subplots(1, 2, figsize=(16, 8))
        plt.subplots_adjust(wspace=0.4, hspace=0.8, bottom=0.17, top=0.93)
      sns.boxplot(
        data=temp_df,
        x=temp_df["group"],
        y=temp_df.iloc[:, i],
        ax=axs[i%2],
        palette=color_palette
      )
      axs[i%2].set_title("IC{}".format(i+1))
    pdf.savefig()
    plt.clf()
    pdf.close()
  
  def arange_pca_and_ica(self, method, startAt, endAt):
    if method == "PC":
      df = pd.DataFrame({"component": "PC{}".format(startAt+1), "value": self.feature[:,startAt], "group": self.df["群"]})
      for i in range(startAt+1, endAt):
        df_pc = pd.DataFrame({"component": "PC{}".format(i+1), "value": self.feature[:,i], "group": self.df["群"]})
        df = pd.concat([df, df_pc])
    elif method == "IC":
      df = pd.DataFrame({"component": "IC{}".format(startAt+1), "value": self.X_transformed[:,startAt], "group": self.df["群"]})
      for i in range(startAt, endAt):
        df_ic = pd.DataFrame({"component": "IC{}".format(i+1), "value": self.X_transformed[:,i], "group": self.df["群"]})
        df = pd.concat([df, df_ic])
    elif method == "ALL":
      df = pd.DataFrame({"component": "PC{}".format(startAt+1), "value": self.feature[:,startAt], "group": self.df["群"]})
      for i in range(startAt+1, endAt):
        df_pc = pd.DataFrame({"component": "PC{}".format(i+1), "value": self.feature[:,i], "group": self.df["群"]})
        df = pd.concat([df, df_pc])
      for i in range(startAt, endAt):
        df_ic = pd.DataFrame({"component": "IC{}".format(i+1), "value": self.X_transformed[:,i], "group": self.df["群"]})
        df = pd.concat([df, df_ic])
    return df
  
  def make_box_plot_pca_and_ica(self, wide=False):
    if wide:
      pdf = PdfPages(self.output_chart_path+"_主成分+独立成分箱ひげ図（拡大）.pdf")
    else:
      pdf = PdfPages(self.output_chart_path+"_主成分+独立成分箱ひげ図.pdf")
    sns.set(font='IPAexGothic', font_scale = 1)
    sns.set_style("whitegrid")
    color_palette = {"A": "#FF8080", "B": "#8080FF"}
    for i in range(self.f_len//5):
      if i%2==0:
        if i != 0:
          pdf.savefig()
          plt.clf()
        f, axs = plt.subplots(2, 1, figsize=(16, 10))
        plt.subplots_adjust(wspace=0.4, hspace=0.8, bottom=0.17, top=0.93)
      temp_df = self.arange_pca_and_ica("PC" if i == 0 else "IC", 0, 10)
      sns.boxplot(
        x="component",
        y="value",
        hue="group",
        data=temp_df,
        ax=axs[i%2],
        palette=color_palette
      )
      axs[i%2].legend(loc='upper right')
      if wide:
        lim_range = (-5,5) if i == 0 else (-.5,.5)
        axs[i%2].set(ylim=(lim_range))
    pdf.savefig()
    plt.clf()
    pdf.close()
  
  def make_scatter(self):
    pdf = PdfPages(self.output_chart_path+"_主成分散布図.pdf")
    i = 0
    feature_A = self.target_df.query('group == "A"')
    feature_A_M = feature_A.query('gender == "男"')
    feature_A_W = feature_A.query('gender == "女"')
    feature_B = self.target_df.query('group == "B"')
    feature_B_M = feature_B.query('gender == "男"')
    feature_B_W = feature_B.query('gender == "女"')
    for c_f, c_s in itertools.combinations(self.target_components, 2):
      if i%2==0:
        if i != 0:
          pdf.savefig()
          plt.clf()
        i = 0
        f, axs = plt.subplots(1, 2, figsize=(16, 8))
        plt.subplots_adjust(wspace=0.4, hspace=0.8, bottom=0.17, top=0.93)
      c_f_max, c_f_min, c_f_a_ave, c_f_b_ave, dx = self.target_df[c_f].max(), self.target_df[c_f].min(), feature_A[c_f].mean(), feature_B[c_f].mean(), (self.target_df[c_f].max() - self.target_df[c_f].min())/100*5
      c_s_max, c_s_min, c_s_a_ave, c_s_b_ave, dy = self.target_df[c_s].max(), self.target_df[c_s].min(), feature_A[c_s].mean(), feature_B[c_s].mean(), (self.target_df[c_s].max() - self.target_df[c_s].min())/100*5
      axs[i%2].scatter(feature_A_M[c_f].values, feature_A_M[c_s].values, alpha=0.8, s=100, c="red", label="A-male", marker="o")
      axs[i%2].scatter(feature_A_W[c_f].values, feature_A_W[c_s].values, alpha=0.8, s=100, c="red", label="A-female", marker="^")
      axs[i%2].scatter(feature_B_M[c_f].values, feature_B_M[c_s].values, alpha=0.8, s=100, c="blue", label="B-male", marker="o")
      axs[i%2].scatter(feature_B_W[c_f].values, feature_B_W[c_s].values, alpha=0.8, s=100, c="blue", label="B-female", marker="^")
      axs[i%2].vlines(c_f_a_ave, c_s_min, c_s_max, "red", linestyles='dashed', label='A-Average')
      axs[i%2].vlines(c_f_b_ave, c_s_min, c_s_max, "blue", linestyles='dashed', label='B-Average')
      axs[i%2].hlines(c_s_a_ave, c_f_min, c_f_max, "red", linestyles='dashed')
      axs[i%2].hlines(c_s_b_ave, c_f_min, c_f_max, "blue", linestyles='dashed')
      plt.xlim(c_f_min-dx, c_f_max+dx)
      plt.ylim(c_s_min-dy, c_s_max+dy)
      axs[i%2].legend(fontsize="x-large")
      axs[i%2].set_xlabel(c_f)
      axs[i%2].set_ylabel(c_s)

      i += 1
    pdf.savefig()
    plt.clf()
    pdf.close()
  
  def make_histgran(self):
    pdf = PdfPages(self.output_chart_path+"_ヒストグラム.pdf")
    feature_A = self.target_df.query('group == "A"')
    feature_B = self.target_df.query('group == "B"')
    i = 0
    for c in self.target_components:
      if i%2==0:
        if i != 0:
          pdf.savefig()
          plt.clf()
        f, axs = plt.subplots(1, 2, figsize=(16, 8))
        plt.subplots_adjust(wspace=0.4, hspace=0.8, bottom=0.17, top=0.93)
      x_range = (min(self.target_df[c]), max(self.target_df[c]))
      axs[i%2].hist(feature_A[c].values, range=x_range, bins=10, alpha=0.5, color="red", ec="darkred", label="A")
      axs[i%2].hist(feature_B[c].values, range=x_range, bins=10, alpha=0.5, color="blue", ec="darkblue", label="B")
      axs[i%2].grid()
      axs[i%2].set_title(c)
      axs[i%2].legend()
      i += 1
    pdf.savefig()
    plt.clf()
    pdf.close()

  def make_relations(self):
    pdf = PdfPages(self.output_chart_path+"_寄与度相関.pdf")
    i = 0
    for c_f, c_s in itertools.combinations(self.target_components, 2):
      if i%2==0:
        if i != 0:
          pdf.savefig()
          plt.clf()
        i = 0
        f, axs = plt.subplots(1, 2, figsize=(20, 10))
        plt.subplots_adjust(wspace=0.4, hspace=0.8, bottom=0.17, top=0.93)
      texts = []
      for x, y, name in zip(self.target_df[c_f].values, self.target_df[c_s].values, self.dfs.columns):
        t = axs[i%2].text(x, y, name)
        texts.append(t)
      axs[i%2].scatter(self.target_df[c_f].values, self.target_df[c_s].values, alpha=0.8, s=10, color="red")
      axs[i%2].grid()
      axs[i%2].set_xlabel(c_f)
      axs[i%2].set_ylabel(c_s)
      adjust_text(texts, ax=axs[i%2], arrowprops=dict(arrowstyle='->', color='m'))
      i += 1
    pdf.savefig()
    plt.clf()
    pdf.close()

  def calc_pca(self):
    '''
    n_samples     : 被験体数
    n_features    : 特徴量数
    n_components  : 主成分数
    '''
    n_components = 20
    ica = FastICA(n_components=n_components, random_state=0)
    self.X_transformed = ica.fit_transform(self.dfs)  #? (n_samples, n_features) => (n_samples, n_components): 各サンプルがそれぞれの主成分をどれだけ有しているかを分布する
    self.f_len = len(self.X_transformed[0])
    self.ica_components = ica.components_
    pca = PCA(n_components=n_components)
    pca.fit(self.dfs)
    self.feature = pca.transform(self.dfs)            #? (n_samples, n_features) => (n_samples, n_components): 各サンプルがそれぞれの主成分をどれだけ有しているかを分布する
    self.f_len = n_components
    self.pca_components = pca.components_

    #! A群 > B群となるように，固有ベクトルの正負を変更
    for i in range(self.f_len):
      A_pca_mean = 0
      A_ica_mean = 0
      B_pca_mean = 0
      B_ica_mean = 0
      for j, group in enumerate(self.df["群"]):
        if group == "A":
          A_pca_mean += self.feature[j, i]
          A_ica_mean += self.X_transformed[j, i]
        elif group == "B":
          B_pca_mean += self.feature[j, i]
          B_ica_mean += self.X_transformed[j, i]
      if A_pca_mean / self.df["群"].value_counts()["A"] < B_pca_mean / self.df["群"].value_counts()["B"]:
        self.pca_components[i] *= -1
        self.feature[:, i] *= -1
      if A_ica_mean / self.df["群"].value_counts()["A"] < B_ica_mean / self.df["群"].value_counts()["B"]:
        self.ica_components[i] *= -1
        self.X_transformed[:, i] *= -1
      
    
    # #! 累積寄与率
    self.make_ticker(pca)

    # #! 固有ベクトルの累積寄与率（x軸固定）
    # self.make_eigenvector_ticker(pca, n_components, lim=True)

    # #! ica のみ箱ひげ図
    # self.make_box_plot()

    # #! pca + ica 箱ひげ図
    # self.make_box_plot_pca_and_ica()

    # #! pca + ica 箱ひげ図（拡大）
    # self.make_box_plot_pca_and_ica(True)

    # #! ica + pca の固有ベクトル
    # df = pd.DataFrame(self.pca_components, columns=self.dfs.columns, index=["PC{}".format(i+1) for i in range(10)])
    # self.output_to_sheet(df.T, sheet_name="PCA_固有ベクトル")
    # df = pd.DataFrame(self.ica_components, columns=self.dfs.columns, index=["IC{}".format(i+1) for i in range(10)])
    # self.output_to_sheet(df.T, sheet_name="ICA_固有ベクトル")

    # #! pca + ica に 群と性別を追加
    target = np.stack([self.feature[:,1], self.feature[:,3], self.feature[:,5], self.feature[:,6], self.X_transformed[:,0], self.X_transformed[:,3], self.X_transformed[:,6]]).T
    self.target_df = pd.DataFrame(target, columns=self.target_components)
    group = pd.Series(self.df["群"], name='group')
    gender = pd.Series(self.df["性別"], name='gender')
    self.target_df = pd.concat([self.target_df, group, gender], axis=1)

    # #! 主成分散布図
    # self.make_scatter()

    # #! ヒストグラム
    # self.make_histgran()

    # #! 寄与度相関
    # target_c = np.stack([pca.components_[1], pca.components_[3], pca.components_[5], pca.components_[6], ica.components_[0], ica.components_[3], ica.components_[6]]).T
    # self.target_df = pd.DataFrame(target_c, columns=self.target_components)
    # self.make_relations()

  def main(self):
    self.init_data()                    #! data/arange から必要データを DataFrame に整形
    self.normalize_data()
    self.calc_pca()


if __name__ == "__main__":
  today = str(datetime.date.today())
  date_format = today[2:4] + today[5:7] + today[8:10]
  #? >>>> ここは変更する >>>>
  input_file_name = "220606 調査報告書+IDs.xlsx"
  output_file_name = date_format + "_集計.xlsx"
  output_chart_name = date_format
  dir_names = ["04_pickup_params"]
  this_dir = 0                                    #! {0: 04_pickup_params}
  #? <<<< ここは変更する <<<<
  input_file_path = os.path.join("../data/arange", dir_names[this_dir], input_file_name)
  output_file_path = os.path.join("../results/", dir_names[this_dir], output_file_name)
  output_chart_path = os.path.join("../results/", dir_names[this_dir], output_chart_name)
  mi = make_ica(input_file_path, output_file_path, output_chart_path)
  mi.main()


# %%
