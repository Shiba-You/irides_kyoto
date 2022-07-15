import pandas as pd
from .. import props

def make(pca_components, ica_components, df, output_file_path):
  output_file_path = output_file_path + "_固有ベクトル.xlsx"
  pca_df = pd.DataFrame(pca_components, columns=df.columns, index=["PC{}".format(i+1) for i in range(10)])
  props.output_to_sheet(pca_df.T, output_file_path, sheet_name="PCA_固有ベクトル")
  ica_df = pd.DataFrame(ica_components, columns=df.columns, index=["IC{}".format(i+1) for i in range(10)])
  props.output_to_sheet(ica_df.T, output_file_path, sheet_name="ICA_固有ベクトル")