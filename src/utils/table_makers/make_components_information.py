import pandas as pd
from .. import props

def make(n_components, output_file_path, feature, X_transformed, pca, df):
  output_file_path = output_file_path + "_成分得点_寄与率_固有値.xlsx"
  cols_pca = ["PC{}".format(x + 1) for x in range(n_components)]
  cols_ica = ["PC{}".format(x + 1) for x in range(n_components)]
  #! 主成分得点
  df = pd.DataFrame(feature, columns=cols_pca)
  props.output_to_sheet(df, output_file_path, sheet_name="主成分得点")
  #! 独立成分得点
  df = pd.DataFrame(X_transformed, columns=cols_ica)
  props.output_to_sheet(df, output_file_path, sheet_name="独立成分得点")
  #! 寄与率
  df = pd.DataFrame(pca.explained_variance_ratio_, index=cols_pca)
  props.output_to_sheet(df, output_file_path, sheet_name="寄与率")
  #! 固有値
  df = pd.DataFrame(pca.explained_variance_, index=cols_pca)
  props.output_to_sheet(df, output_file_path, sheet_name="固有値")