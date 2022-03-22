#%%
'''
python @ 3.10.2
'''
import pandas as pd
from pandas import plotting 
import numpy as np
import os.path
import matplotlib
import japanize_matplotlib
import matplotlib.pyplot as plt
import sklearn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class make_pca:
  def __init__(self, input_file_name, output_file_name, output_chart_path):
    self.input_file_name = input_file_name
    self.output_file_name = output_file_name
    self.output_chart_path = output_chart_path
    self.df = pd.DataFrame()
    self.dfs = pd.DataFrame()
    self.key = []

  def init_data(self):
    if os.path.exists(self.input_file_name):
      xls = pd.ExcelFile(self.input_file_name)
      sheets = xls.sheet_names
      input_sheet = sheets[4]
      self.df = pd.DataFrame(xls.parse(input_sheet))
      self.key = list(self.df.columns)[2:]
    else:
      print("ファイルが存在しません．")
      exit()
    return
  
  def normalize_data(self):
    self.dfs = self.df.iloc[:, 2:].apply(lambda x: (x-x.mean())/x.std(), axis=0)
    for k in self.key:
      self.dfs[k] = self.dfs[k].fillna(self.dfs[k].mean())
  
  def output_to_sheet(self, df, sheet):
    with pd.ExcelWriter(self.output_file_name, mode='a') as writer:
      df.to_excel(writer, sheet_name=sheet)
  
  def make_tsne(self):
    tsne = TSNE(n_components = 2, random_state = 0, perplexity = 30, n_iter = 1000)
    dfs_tsne = tsne.fit_transform(self.dfs)

    plt.figure(figsize=(5,5))
    plt.xlim(dfs_tsne[:,0].min(), dfs_tsne[:,0].max()+1)
    plt.ylim(dfs_tsne[:,1].min(), dfs_tsne[:,1].max()+1)

    colors = {"A": "r", "B": "b"}
    handles = []
    labels=["A", "B"]

    for key, val in colors.items():
      plot = plt.scatter([], [], label=key, color=val) #凡例Aのダミープロット
      handles.append(plot)

    for i in range(len(self.dfs)):
      plot = plt.scatter(dfs_tsne[i, 0], dfs_tsne[i, 1], label=self.df.iloc[i, 1], alpha=0.8, s=10, color=colors[self.df.iloc[i, 1]])
      plt.text(dfs_tsne[i, 0], dfs_tsne[i, 1], str(self.df.iloc[i, 0]), fontdict={"weight":"bold", "size":10})

    plt.xlabel("t-SNE feature 0")
    plt.ylabel("t-SNE feature 1")
    
    plt.legend(handles, labels, fontsize=8)
    plt.savefig(self.output_chart_path+"_tsne.png")
    plt.show()



  def main(self):
    self.init_data()                    #! data/arange から必要データを DataFrame に整形
    self.normalize_data()
    self.make_tsne()


if __name__ == "__main__":
  #? >>>> ここは変更する >>>>
  input_file_name = "220208 調査報告書+IDs.xlsx"
  output_file_name = "result.xlsx"
  output_chart_name = "result"
  dir_names = ["04_pickup_params"]
  this_dir = 0                                    #! {0: 04_pickup_params}
  #? <<<< ここは変更する <<<<
  input_file_path = os.path.join("../data/arange", dir_names[this_dir], input_file_name)
  output_file_path = os.path.join("../results/", dir_names[this_dir], output_file_name)
  output_chart_path = os.path.join("../results/", dir_names[this_dir], output_chart_name)
  mp = make_pca(input_file_path, output_file_path, output_chart_path)
  mp.main()


#%%
