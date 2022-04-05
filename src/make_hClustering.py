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
from scipy.cluster.hierarchy import linkage, dendrogram

class make_hClustering:
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
  
  def change_row_name(self):
    new_row_name = [str(sub_id) + "_" + sub_group for sub_id, sub_group in zip(self.df.iloc[:, 0].values, self.df.iloc[:, 1].values)]
    self.dfs.set_axis(new_row_name, axis=0, inplace=True)
    print(self.dfs)
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

  def make_dendrogram(self):
    pdf = PdfPages(self.output_chart_path+"_樹形図.pdf")
    new_row_name = [str(sub_id) + "_" + sub_group for sub_id, sub_group in zip(self.df.iloc[:, 0].values, self.df.iloc[:, 1].values)]
    fig, axes = plt.subplots(1, 1, figsize=(20, 10))
    linked = linkage(self.dfs, metric = 'euclidean', method = 'ward')
    #! color_thread は クラスタ間の距離の最大値 * 0.7 (default設定のまま)を採用
    dendrogram(linked, ax=axes, truncate_mode='lastp', labels=new_row_name)
    pdf.savefig()
    plt.show()
    plt.clf()
    pdf.close()

  def main(self):
    self.init_data()                    #! data/arange から必要データを DataFrame に整形
    self.normalize_data()
    self.make_dendrogram()


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
  mhc = make_hClustering(input_file_path, output_file_path, output_chart_path)
  mhc.main()


#%%
