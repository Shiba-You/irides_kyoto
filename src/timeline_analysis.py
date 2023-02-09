#%%
'''
python @ 3.10.2
'''
import datetime
import importlib
import os
import sys

import pandas as pd
from utils import arange_data, props
from utils.data_generators import (generate_components,
                                   generate_target_components,
                                   generate_target_features)
from utils.figure_drawers import (draw_boxplot,
                                  draw_cumulative_contribution_rate,
                                  draw_histgram, draw_relations, draw_scatter,
                                  draw_ticker, draw_bar, draw_linegraph)
from utils.loggers import checker, mathematical_checker
from utils.table_makers import make_components_information, make_eigenvector

importlib.reload(props)
importlib.reload(arange_data)
importlib.reload(generate_components)
importlib.reload(generate_target_features)
importlib.reload(generate_target_components)
importlib.reload(draw_boxplot)
importlib.reload(draw_cumulative_contribution_rate)
importlib.reload(draw_histgram)
importlib.reload(draw_relations)
importlib.reload(draw_scatter)
importlib.reload(draw_ticker)
importlib.reload(draw_bar)
importlib.reload(draw_linegraph)
importlib.reload(make_eigenvector)
importlib.reload(make_components_information)
importlib.reload(checker)
importlib.reload(mathematical_checker)

today = str(datetime.date.today())
date_format = today[2:4] + today[5:7] + today[8:10]

#? >>>> ここは変更する >>>>
input_file_name = "221226 arange_data.xlsx"
output_file_name = date_format
dir_names = "03_timeline"                   #! input と result のディレクトリ名
target_sheet_number = 3                         #! {3: 03_PC_IC,    4: _03_PC_IC_A後}
#? <<<< ここは変更する <<<<

input_file_path = os.path.join("../data/arange", dir_names, input_file_name)
output_file_path = os.path.join("../results/", dir_names, output_file_name)
target_components_columns = []

def main():
  #? ============================ データ生成 ============================
  # df, group_and_gender = props.init_data(input_file_path, target_sheet_number, start_columns=1, gg_flag=False)
  xls = pd.ExcelFile(input_file_path)
  sheets = xls.sheet_names
  input_sheet = sheets[target_sheet_number]
  df = pd.DataFrame(xls.parse(input_sheet))
  # print(df_part)
  #? ==================================================================
  
  #? ============================ データ生成 ============================

  draw_linegraph.draw(output_file_path, df)

  
  #? ==================================================================

  #? ============================ ログ出力 ==============================
  #! 各 component 毎の外れ値出力
  # checkers.output_check(target_features_df, target_components_columns)
  #! 数学的なテスト
  # mathematical_checker.checker(feature, pca_components, pca, df, n_components)
  #? ==================================================================


if __name__ == "__main__":
  main()


# %%





