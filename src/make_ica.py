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
from pyparsing import alphas
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import sklearn
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA

class make_ica:
  def __init__(self, input_file_name, output_file_name, output_chart_path):
    self.input_file_name = input_file_name
    self.output_file_name = output_file_name
    self.output_chart_path = output_chart_path
    self.df = pd.DataFrame()
    self.dfs = pd.DataFrame()
    self.X_transformed = []
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
  
  def arange_pca_and_ica(self, methods, startAt, endAt):
    df = pd.DataFrame({"component": "PC{}".format(startAt+1), "value": self.feature[:,startAt], "group": self.df["群"]})
    for i in range(startAt+1, endAt):
      df_pc = pd.DataFrame({"component": "PC{}".format(i+1), "value": self.feature[:,i], "group": self.df["群"]})
      df = pd.concat([df, df_pc])
    for i in range(startAt, endAt):
      df_ic = pd.DataFrame({"component": "IC{}".format(i+1), "value": self.X_transformed[:,i], "group": self.df["群"]})
      df = pd.concat([df, df_ic])
    return df
  
  def make_box_plot_pca_and_ica(self):
    pdf = PdfPages(self.output_chart_path+"_主成分+独立成分箱ひげ図.pdf")
    sns.set(font='IPAexGothic', font_scale = 1)
    sns.set_style("whitegrid")
    color_palette = {"A": "#FF8080", "B": "#8080FF"}
    for i in range(self.f_len//5):
      if i%2==0:
        if i != 0:
          pdf.savefig()
          plt.clf()
        f, axs = plt.subplots(1, 2, figsize=(16, 8))
        plt.subplots_adjust(wspace=0.4, hspace=0.8, bottom=0.17, top=0.93)
      temp_df = self.arange_pca_and_ica(["PC"] if i == 0 else ["IC"], 0, 10)
      sns.boxplot(
        x="component",
        y="value",
        hue="group",
        data=temp_df,
        ax=axs[i%2],
        palette=color_palette
      )
    pdf.savefig()
    plt.clf()
    # temp_df = self.arange_pca_and_ica(0, 10)
    # f, axs = plt.subplots(1, 1, figsize=(32, 26))
    # plt.subplots_adjust(wspace=0.4, hspace=0.8, bottom=0.17, top=0.93)
    # sns.boxplot(
    #   x="component",
    #   y="value",
    #   hue="group",
    #   data=temp_df,
    #   ax=axs,
    #   palette=color_palette
    # )
    # pdf.savefig()
    # plt.clf()
    pdf.close()

  def calc_pca(self):
    '''
    n_samples     : 被験体数
    n_features    : 特徴量数
    n_components  : 主成分数
    '''
    n_components = 10
    ICA = FastICA(n_components=n_components, random_state=0)
    self.X_transformed = ICA.fit_transform(self.dfs)
    self.f_len = len(self.X_transformed[0])
    pca = PCA(n_components=n_components)
    pca.fit(self.dfs)
    self.feature = pca.transform(self.dfs)         #? (n_samples, n_features) => (n_samples, n_components): 各サンプルがそれぞれの主成分をどれだけ有しているかを分布する
    self.f_len = n_components
    # self.f_len = len(self.feature[0])

    # #! ica のみ箱ひげ図
    # self.make_box_plot()

    #! pca + ica 箱ひげ図
    self.make_box_plot_pca_and_ica()


  def main(self):
    self.init_data()                    #! data/arange から必要データを DataFrame に整形
    self.normalize_data()
    self.calc_pca()


if __name__ == "__main__":
  today = str(datetime.date.today())
  date_format = today[2:4] + today[5:7] + today[8:10]
  #? >>>> ここは変更する >>>>
  input_file_name = "220325 調査報告書+IDs_A先.xlsx"
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


#%%
for i in range(5):
  print("PC" if i%2 == 0 else "IC")
# %%
