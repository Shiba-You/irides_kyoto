import os
import sys
import pandas as pd
import openpyxl
import glob
from PyPDF2 import PdfFileMerger

'''
データの初期化
input :   input_file_name       ::  <string>      入力ファイル名
          target_sheet_number   ::  <string>      入力ファイルの内参照するシート番号
output:   df                    ::  <DataFrame>   元のデータから group, gender を除外して 正規化 & 欠損値処理 した dataFrame
          group_and_gender      ::  <DataFrame>   (n_samples = 31, 群 + 性別)  
'''
def init_data(input_file_name, target_sheet_number, start_columns=3, gg_flag=True):
  if os.path.exists(input_file_name):
    xls = pd.ExcelFile(input_file_name)
    sheets = xls.sheet_names
    input_sheet = sheets[target_sheet_number]
    df = pd.DataFrame(xls.parse(input_sheet))
    key = list(df.columns)[start_columns:]
  else:
    print("ファイルが存在しません．")
    sys.exit()
  group_and_gender = []
  if gg_flag:
    group = pd.Series(df["群"], name='group')
    gender = pd.Series(df["性別"], name='gender')
    group_and_gender = pd.concat([group, gender], axis=1)
  df = _normalize_data(df, key, gg_flag)
  return df, group_and_gender

def _normalize_data(df, key, gg_flag):
  if gg_flag:
    normalized_df = df.iloc[:, 3:]
    normalized_df = normalized_df.apply(lambda x: (x-x.mean())/x.std(), axis=0)
    for k in key:
      normalized_df[k] = normalized_df[k].fillna(normalized_df[k].mean())
  else:
    normalized_df = df.iloc[:, 2:]
  return normalized_df

def _calc_lim(axs):
  min_x, max_x = axs.get_xlim()
  min_y, max_y = axs.get_ylim()
  dx = (max_x - min_x) / 100 * 2
  dy = (max_y - min_y) / 100 * 2
  return dx, dy

def output_to_sheet(df, output_file_path, sheet_name):
  if not os.path.isfile(output_file_path):
    wb = openpyxl.Workbook()
    wb.save(output_file_path)
    glob.glob("*.xlsx")
  try:
    with pd.ExcelWriter(output_file_path, mode='a') as writer:
      print("df    : ", df.shape)
      print("sheet : ", sheet_name)
      print("writer: ", writer)
      df.to_excel(writer, sheet_name=sheet_name)
  except:
    print("既に作成済みです．")

def arange_target_components_obj(target_components, target_sheet_num):
  target_components_obj = {}
  for key, val in target_components.items():
    for i in val:
      target_components_obj[f"{key}{i+1}"] = target_sheet_num
      target_sheet_num += 1
  return target_components_obj

def merge_same_size_fig(output_file_path, target_components_obj):
  merger = PdfFileMerger()
  for target_component_name in target_components_obj:
    print(f"{output_file_path}_{target_component_name}.pdf")
    merger.append(f"{output_file_path}_{target_component_name}.pdf")
  merger.write(f"{output_file_path}_LiNGAM")
  