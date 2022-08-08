from operator import le
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
def draw(output_file_path, n_components, pca):
  pdf = PdfPages(output_file_path+"_累積寄与率.pdf")
  fig = plt.figure(figsize=(10,10))
  ax = fig.add_subplot(111)
  dx, dy = props._calc_lim(ax)
  accumulation_list = [0] + list(np.cumsum(pca.explained_variance_ratio_))
  ratio_list = [0] * len(accumulation_list)
  print(accumulation_list)
  for i in range(len(accumulation_list)):
    if i != 0:
      ratio_list[i] = accumulation_list[i] - accumulation_list[i-1]
  #! 折れ線グラフ
  ax.plot(accumulation_list, "-o")
  # for x, y in enumerate(accumulation_list):
  #   if x != 0:
  #     ax.text(x-dx-.3, y+dy, x)
  ax.grid()
  ax.hlines(0.9, -1, n_components+1, "red", linestyles='dashed')
  ax.set_xlim(-dx*30, n_components+dx*30)
  ax.set_ylim(0, 1.+dy)
  ax.set_xlabel("Number of principal components")
  ax.set_ylabel("Cumulative contribution rate")
  # #! 棒グラフ
  shift_ratio_list = []
  for i in range(len(ratio_list)):
    shift_ratio_list.append(i-.5)
  ax.bar(shift_ratio_list, ratio_list, width=1.0, alpha=0.6)
  pdf.savefig()
  plt.show()
  plt.clf()
  pdf.close()

# def draw(output_file_path, n_components, pca):
#   pdf = PdfPages(output_file_path+"_累積寄与率.pdf")
#   plt.figure(figsize=(10,10))
#   axes = plt.gca()
#   axes.get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
#   dx, dy = props._calc_lim(axes)
#   accumulation_list = [0] + list(np.cumsum(pca.explained_variance_ratio_))
#   #! 折れ線グラフ
#   plt.plot(accumulation_list, "-o")
#   # #! 棒グラフ
#   plt.bar(len(accumulation_list), accumulation_list, width=1.0)
#   for x, y in enumerate(accumulation_list):
#     if x != 0:
#       plt.text(x-dx-.3, y+dy, x)
#   plt.xlabel("Number of principal components")
#   plt.ylabel("Cumulative contribution rate")
#   plt.grid()
#   plt.hlines(0.9, -1, n_components+1, "red", linestyles='dashed')
#   plt.xlim(-dx*30, n_components+dx*30)
#   plt.ylim(-dy*3, 1.+dy*3)
#   pdf.savefig()
#   plt.show()
#   plt.clf()
#   pdf.close()