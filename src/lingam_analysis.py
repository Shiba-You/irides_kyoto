#%%
'''
python @ 3.10.2
'''
import pandas as pd
import datetime
import os
import sys

from utils import props
from utils.figure_drawers import draw_lingam
import importlib

importlib.reload(props)
importlib.reload(draw_lingam)

today = str(datetime.date.today())
date_format = today[2:4] + today[5:7] + today[8:10]

#? >>>> ここは変更する >>>>
input_file_name = "220707 arange_data.xlsx"
output_file_name = date_format
dir_names = "01_main_feature"                   #! input と result のディレクトリ名
target_sheet_number = 3                         #! {3: 03_PC_IC,    4: _03_PC_IC_A後}
n_components = 10                               #! components をどこまで加味するか
target_components = {
  "PC" : [0, 1, 4, 8],
  "IC" : [1, 2, 4]
}
target_sheet_num = 7                            #! 最初の要素のシート番号
#? <<<< ここは変更する <<<<

input_file_path = os.path.join("../data/arange", dir_names, input_file_name)
output_file_path = os.path.join("../results/", dir_names, output_file_name)

#! 対象となる component の名前と excel の sheet 番号を辞書化
target_components_obj = props.arange_target_components_obj(target_components, target_sheet_num)

def main():
  #? ============================ データ生成 ============================
  for target_component_name, target_sheet_num in target_components_obj.items():
    print(f"対象成分: {target_component_name}, シート番号: {target_sheet_num}")
    #! データの読み込み + 初期化
    df, _ = props.init_data(input_file_path, target_sheet_num)
    #! lingam グラフの作成
    draw_lingam.draw(df, output_file_path, target_component_name)
  #! pdf を一つ目にまとめる
  props.merge_same_size_fig(output_file_path, target_components_obj)
  
  #? ==================================================================


if __name__ == "__main__":
  main()

# %%
