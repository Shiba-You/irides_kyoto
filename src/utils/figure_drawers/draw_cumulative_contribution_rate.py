import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from .. import props

'''
固有ベクトル毎の累積寄与率のPDF出力
input :   output_file_path      ::  <string>           PDFの出力先
          df                    ::  <DataFrame>        元のデータから group, gender を除外して 正規化 & 欠損値処理 した dataFrame
          n_components          ::  <number>           加味する components 数
          pca                   ::  <Custom>           PCA
'''
def draw(output_file_path, df, n_components, pca_components):
  pdf = PdfPages(output_file_path+"_固有ベクトルの累積寄与率.pdf")
  i = 0
  for feature_idx in range(len(pca_components[0])):
    if i%2==0:
      if i != 0:
        pdf.savefig()
        plt.clf()
      i = 0
      f, axs = plt.subplots(1, 2, figsize=(20, 10))
      plt.subplots_adjust(wspace=0.4, hspace=0.8, bottom=0.17, top=0.93)
    contribution_rate = [0]
    for idx, val in enumerate(pca_components[:, feature_idx]):
      contribution_rate.append(contribution_rate[idx]+val**2)
    contribution_rate.pop(0)
    components_list = np.arange(n_components)+1
    axs[i%2].plot(components_list, contribution_rate, "-o")
    axs[i%2].set_ylim(0, 1)
    dx, dy = props._calc_lim(axs[i%2])
    for x, y in zip(components_list, contribution_rate):
      axs[i%2].text(x-dx, y+dy, x)
    axs[i%2].set_xlabel("Number of principal components")
    axs[i%2].set_ylabel("Cumulative contribution rate of " + df.columns[feature_idx])
    axs[i%2].grid()
    i += 1
    if feature_idx == len(pca_components[0])-1:
      pdf.savefig()
      plt.clf()
  pdf.close()