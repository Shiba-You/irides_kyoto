import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as ticker
import numpy as np
from .. import props

'''
固有ベクトル毎の累積寄与率のPDF出力
input :   output_file_path      ::  <string>           PDFの出力先
          df                    ::  <DataFrame>        元のデータから group, gender を除外して 正規化 & 欠損値処理 した dataFrame
          pca                   ::  <Custom>           PCA
          n_components          ::  <number>           加味する components 数
'''
def draw(output_file_path, df, n_components, pca):
  pdf = PdfPages(output_file_path+"_累積寄与率.pdf")
  plt.figure(figsize=(20,10))
  axes = plt.gca()
  axes.get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
  dx, dy = props._calc_lim(axes)
  plt.plot([0] + list(np.cumsum(pca.explained_variance_ratio_)), "-o")
  for x, y in zip(df.iloc[:, 0], list(np.cumsum(pca.explained_variance_ratio_))):
    plt.text(x-dx, y+dy, x)
  plt.xlabel("Number of principal components")
  plt.ylabel("Cumulative contribution rate")
  plt.grid()
  plt.hlines(0.9, -1, n_components+1, "red", linestyles='dashed')
  plt.xlim(-dx*30, n_components+dx*30)
  plt.ylim(-dy*3, 1.+dy*3)
  pdf.savefig()
  plt.clf()
  pdf.close()