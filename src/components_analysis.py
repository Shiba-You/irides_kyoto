#%%
'''
python @ 3.10.2
'''
import pandas as pd
import datetime
import os
import sys

from utils import props
from utils import arange_data
from utils.data_generators import generate_components
from utils.data_generators import generate_target_features
from utils.data_generators import generate_target_components
from utils.figure_drawers import draw_boxplot
from utils.figure_drawers import draw_cumulative_contribution_rate
from utils.figure_drawers import draw_histgram
from utils.figure_drawers import draw_relations
from utils.figure_drawers import draw_scatter
from utils.figure_drawers import draw_ticker
from utils.table_makers import make_eigenvector
from utils.table_makers import make_components_information
from utils.loggers import checkers
import importlib


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
importlib.reload(make_eigenvector)
importlib.reload(make_components_information)
importlib.reload(checkers)

today = str(datetime.date.today())
date_format = today[2:4] + today[5:7] + today[8:10]

#? >>>> ここは変更する >>>>
input_file_name = "220707 arange_data.xlsx"
output_file_name = date_format
dir_names = "01_main_feature"                   #! input と result のディレクトリ名
target_sheet_number = 3                         #! {3: 03_PC_IC,    4: _03_PC_IC_A後}
n_components = 10                               #! components をどこまで加味するか
target_components = {
  "PC": [0, 1, 4, 8],
  "IC": [1, 2, 4]
}
#? <<<< ここは変更する <<<<

input_file_path = os.path.join("../data/arange", dir_names, input_file_name)
output_file_path = os.path.join("../results/", dir_names, output_file_name)
target_components_columns = []
for key, val in target_components.items():
  for i in val:
    target_components_columns.append(f"{key}{i+1}")

def main():
  #? ============================ データ生成 ============================
  df, group_and_gender = props.init_data(input_file_path, target_sheet_number)
  #! components 生成
  X_transformed, ica_components, feature, pca_components, ica, pca \
    = generate_components.generate(
      n_components, df
    )
  #! 正負を調整
  X_transformed, ica_components, feature, pca_components \
    = arange_data.convert_A_positive(
      n_components,
      df,
      X_transformed,
      ica_components, 
      feature,
      pca_components,
      group_and_gender
    )
  #? ==================================================================

  #? ============================ グラフ生成 ============================
  # #! 累積寄与率
  # draw_ticker.draw(output_file_path, df, n_components, pca)
  # #! 固有ベクトルの累積寄与率
  # draw_cumulative_contribution_rate.draw(output_file_path, df, n_components, pca_components)
  # #! pca + ica 箱ひげ図（拡大）
  # draw_boxplot.draw(output_file_path, feature, X_transformed, group_and_gender)
  #? ==================================================================

  #? ========================== テーブル生成 =============================
  # #! 成分得点_寄与率_固有値の生成
  # make_components_information.make(n_components, output_file_path, feature, X_transformed, pca, df)
  # #! 固有ベクトルの生成
  # make_eigenvector.make(pca_components, ica_components, df, output_file_path)
  #? ==================================================================




  #? ******************************************************************
  #? ************************** 注目特徴量  *****************************
  #? ******************************************************************
  
  #? ============================ データ変製 ============================
  target_features_df = generate_target_features.generate(target_components, target_components_columns, feature, X_transformed, group_and_gender)
  # target_components_df  = generate_target_components.generate(target_components, target_components_columns, pca, ica)
  #? ==================================================================

  #? ============================ グラフ生成 ============================
  #! 主成分散布図
  # draw_scatter.draw(output_file_path, target_features_df, target_components_columns, outliers = True)
  #! 主成分散布図 - 外側に凡例を表示
  # draw_scatter.draw_out_legend(output_file_path, target_features_df, target_components_columns, outliers = True)
  # #! ヒストグラム
  # draw_histgram.draw(output_file_path, target_features_df, target_components_columns)
  # #! 寄与度相関
  # draw_relations.draw(output_file_path, target_components_columns, target_components_df, df)
  #? ==================================================================

  #? ============================ ログ出力 ==============================
  #! 各 component 毎の外れ値出力
  checkers.output_check(target_features_df, target_components_columns)
  #? ==================================================================


if __name__ == "__main__":
  main()

# %%
