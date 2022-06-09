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
  
  def output_to_sheet(self, df, sheet_name):
    if not os.path.isfile(self.output_file_name):
      wb = openpyxl.Workbook()
      sheet = wb.active
      # sheet.title = '_'
      wb.save(self.output_file_name)
      glob.glob("*.xlsx")
    with pd.ExcelWriter(self.output_file_name, mode='a') as writer:
      print("df    : ", df.shape)
      print("sheet : ", sheet_name)
      print("writer: ", writer)
      df.to_excel(writer, sheet_name=sheet_name)


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
