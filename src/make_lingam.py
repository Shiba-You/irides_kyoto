#%%
'''
python @ 3.10.2
'''
import datetime
from tokenize import group
import pandas as pd
from pandas import plotting 
import numpy as np
import os.path
import openpyxl
import glob
import graphviz
import lingam
from lingam.utils import make_dot

class make_lingam:
  def __init__(self, input_file_name, output_file_name, output_chart_path):
    self.input_file_name = input_file_name
    self.output_file_name = output_file_name
    self.output_chart_path = output_chart_path
    self.df = pd.DataFrame()
    self.dfs = pd.DataFrame()
    self.key = []
    self.target_df = pd.DataFrame()
    self.target_components = {"PC2": 5, "PC4": 6, "PC6": 7, "PC7": 8, "IC1": 9, "IC4": 10, "IC7": 11}

  def init_data(self, sheet_num):
    if os.path.exists(self.input_file_name):
      xls = pd.ExcelFile(self.input_file_name)
      sheets = xls.sheet_names
      input_sheet = sheets[sheet_num]
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
  
  def output_to_sheet(self, df, sheet_name):
    if not os.path.isfile(self.output_file_name):
      wb = openpyxl.Workbook()
      sheet = wb.active
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

  def calc_lingam(self, component):
    model = lingam.DirectLiNGAM()
    model.fit(self.dfs)

    print(f"因果関係の順番:  {model.causal_order_}")
    print(f"因果の変数間の結合強度:  {model.adjacency_matrix_}")
    print("結合モデル")
    dot = make_dot(model.adjacency_matrix_, labels=self.key)
    dot.render(f"{self.output_chart_path}_{component}")
    return

  def main(self):
    for component, sheet_num in self.target_components.items():
      self.init_data(sheet_num)                    #! data/arange から必要データを DataFrame に整形
      self.normalize_data()
      self.calc_lingam(component)


if __name__ == "__main__":
  today = str(datetime.date.today())
  date_format = today[2:4] + today[5:7] + today[8:10]
  #? >>>> ここは変更する >>>>
  input_file_name = "220606 調査報告書+IDs_A先.xlsx"
  output_file_name = date_format + "_集計.xlsx"
  output_chart_name = date_format
  dir_names = ["05_PC2,4,6,7,IC1,4,7"]
  this_dir = 0                                    #! {0: 04_pickup_params}
  #? <<<< ここは変更する <<<<
  input_file_path = os.path.join("../data/arange", dir_names[this_dir], input_file_name)
  output_file_path = os.path.join("../results/", dir_names[this_dir], output_file_name)
  output_chart_path = os.path.join("../results/", dir_names[this_dir], output_chart_name)
  ml = make_lingam(input_file_path, output_file_path, output_chart_path)
  ml.main()


# %%

for i, val in {"PC2": 5, "PC4": 6, "PC6": 7, "PC7": 8}.items():
  print(i, val)
# %%
