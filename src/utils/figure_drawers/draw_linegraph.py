from operator import le
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as ticker
import numpy as np
from .. import props

'''
固有ベクトル毎のタイムラインのPDF出力
input :   output_file_path      ::  <string>           PDFの出力先
          df                    ::  <DataFrame>        元のデータから group, gender を除外して 正規化 & 欠損値処理 した dataFrame
'''

def draw(output_file_path, df):
  pdf = PdfPages(output_file_path+"_タイムライン.pdf")
  i = 0
  for c in df.columns[1:]:
    print(c)
    if i%2==0:
      if i != 0:
        pdf.savefig()
        plt.clf()
      f, axs = plt.subplots(1, 2, figsize=(16, 8))
      plt.subplots_adjust(wspace=0.4, hspace=0.8, bottom=0.17, top=0.93)
    axs[i%2].plot(df["part"], df[c])
    axs[i%2].grid()
    axs[i%2].set_title(c)
    i += 1
  pdf.savefig()
  plt.clf()
  pdf.close()