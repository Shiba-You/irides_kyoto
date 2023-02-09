import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from .. import arange_data
import pandas as pd
import itertools

def draw(output_file_path, feature):
  pdf = PdfPages(output_file_path+"_棒グラフ.pdf")
  sns.set(font='IPAexGothic', font_scale = 1)
  sns.set_style("whitegrid")

  color_palette = {"Action": "#FF8080", "Diff": "#8080FF"}
  f, axs = plt.subplots(1, 1, figsize=(16, 10))
  df_ndarray = feature.to_numpy()
  df = pd.DataFrame({"feature": feature.columns[2], "value": df_ndarray[:,2], "action": df_ndarray[:,1]})
  for i in range(3, 21):
    df_tmp = pd.DataFrame({"feature": feature.columns[i], "value": df_ndarray[:,i], "action": df_ndarray[:,0]})
    df = pd.concat([df, df_tmp])
  j = 0
  for _f, _a in itertools.product(df["feature"].unique(), df["action"].unique()):
    if j == 0:
      mean_df = pd.DataFrame({"feature": _f, "value": df.query(f"feature == '{_f}' & action == '{_a}'").mean(), "action": _a})
    else:
      tmp_df = pd.DataFrame({"feature": _f, "value": df.query(f"feature == '{_f}' & action == '{_a}'").mean(), "action": _a})
      mean_df = pd.concat([mean_df, tmp_df])
    j += 1
  sns.barplot(
    x='feature',
    y='value',
    hue='action',
    data=mean_df,
    palette=color_palette,
  )
  axs.legend(loc='upper right')
  axs.set()

  pdf.savefig()
  plt.clf()
  pdf.close()
