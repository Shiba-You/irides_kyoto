import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from .. import arange_data

def draw(output_file_path, target_features_df, target_components_columns, outliers = True):
  pdf = PdfPages(output_file_path+"_ヒストグラム.pdf")
  feature_A = target_features_df.query('group == "A"')
  feature_B = target_features_df.query('group == "B"')
  i = 0
  for c in target_components_columns:
    if i%2==0:
      if i != 0:
        pdf.savefig()
        plt.clf()
      f, axs = plt.subplots(1, 2, figsize=(16, 8))
      plt.subplots_adjust(wspace=0.4, hspace=0.8, bottom=0.17, top=0.93)
    if outliers:
      df_c = arange_data._drop_outlier(target_features_df[["group", "gender", c]], c)
    else:
      df_c = target_features_df[["group", "gender", c]]
    feature_A = df_c.query('group == "A"')
    feature_B = df_c.query('group == "B"')
    x_range = (min(target_features_df[c]), max(target_features_df[c]))
    axs[i%2].hist(feature_A[c].values, range=x_range, bins=10, alpha=0.5, color="red", ec="darkred", label="A")
    axs[i%2].hist(feature_B[c].values, range=x_range, bins=10, alpha=0.5, color="blue", ec="darkblue", label="B")
    axs[i%2].grid()
    axs[i%2].set_title(c)
    axs[i%2].legend()
    i += 1
  pdf.savefig()
  plt.clf()
  pdf.close()