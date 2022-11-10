import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from .. import arange_data

def draw(output_file_path, feature, X_transformed, group_and_gender, component_flag=True):
  pdf = PdfPages(output_file_path+"_箱ひげ図.pdf")
  sns.set(font='IPAexGothic', font_scale = 1)
  sns.set_style("whitegrid")
 
  if component_flag:
    _with_components(pdf, feature, X_transformed, group_and_gender)
  else:
    _plane(pdf, feature)
  pdf.savefig()
  plt.clf()
  pdf.close()


def _with_components(pdf, feature, X_transformed, group_and_gender):
  color_palette = {"A": "#FF8080", "B": "#8080FF"}
  for i in range(2):
    if i%2==0:
      if i != 0:
        pdf.savefig()
        plt.clf()
      f, axs = plt.subplots(2, 1, figsize=(16, 10))
      plt.subplots_adjust(wspace=0.4, hspace=0.8, bottom=0.17, top=0.93)
    print("--------------------- feature ---------------------")
    print(type(feature), feature.shape)
    print(feature)
    print("")
    print("")
    print("----------------- X_transformed -------------------")
    print(type(X_transformed), X_transformed.shape)
    print(X_transformed)
    print("")
    print("")
    
    temp_df = arange_data.arange_pca_and_ica("PC" if i == 0 else "IC", feature, X_transformed, group_and_gender)
    print(temp_df.head)
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

def _plane(pdf, df):
  color_palette = {"Action": "#FF8080", "Diff": "#8080FF"}
  for i in range(2):
    if i%2==0:
      if i != 0:
        pdf.savefig()
        plt.clf()
      f, axs = plt.subplots(2, 1, figsize=(16, 10))
      plt.subplots_adjust(wspace=0.4, hspace=0.8, bottom=0.17, top=0.93)
    tmp_df = arange_data.arange_action_and_diff(df, i)
    print(tmp_df)
    sns.boxplot(
      x='feature',
      y='value',
      hue='action',
      data=tmp_df,
      ax=axs[i%2],
      palette=color_palette,
    )
    axs[i%2].legend(loc='upper right')
    # lim_range = (-5,5) if i == 0 else (-.5,.5)
    # axs[i%2].set(ylim=(lim_range))
    axs[i%2].set()
