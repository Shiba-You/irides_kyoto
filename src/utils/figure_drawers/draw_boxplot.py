import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from .. import arange_data

def draw(output_file_path, feature, X_transformed, group_and_gender):
  pdf = PdfPages(output_file_path+"_箱ひげ図.pdf")
  sns.set(font='IPAexGothic', font_scale = 1)
  sns.set_style("whitegrid")
  color_palette = {"A": "#FF8080", "B": "#8080FF"}
  for i in range(2):
    if i%2==0:
      if i != 0:
        pdf.savefig()
        plt.clf()
      f, axs = plt.subplots(2, 1, figsize=(16, 10))
      plt.subplots_adjust(wspace=0.4, hspace=0.8, bottom=0.17, top=0.93)
    temp_df = arange_data.arange_pca_and_ica("PC" if i == 0 else "IC", feature, X_transformed, group_and_gender)
    sns.boxplot(
      x="component",
      y="value",
      hue="group",
      data=temp_df,
      ax=axs[i%2],
      palette=color_palette,
    )
    axs[i%2].legend(loc='upper right')
    # axs[i%2].legend([],[],frameon=False)
    lim_range = (-5,5) if i == 0 else (-.5,.5)
    axs[i%2].set(ylim=(lim_range))
  pdf.savefig()
  plt.clf()
  pdf.close()