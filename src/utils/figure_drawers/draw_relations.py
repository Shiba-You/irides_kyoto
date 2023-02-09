import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import itertools
from adjustText import adjust_text

'''
2成分の散布図のPDF出力
input :   output_file_path            ::  <string>           PDFの出力先
          target_components_columns   ::  <string[]>         加味する成分名の配列
          target_components_df        ::  <DataFrame>        加味する成分の成分得点のdataFrame
          df                          ::  <DataFrame>        元のデータから group, gender を除外して 正規化 & 欠損値処理 した dataFrame
'''
def draw(output_file_path, target_components_columns, target_components_df, df):
  pdf = PdfPages(output_file_path+"_寄与度相関.pdf")
  i = 0
  for c_f, c_s in itertools.combinations(target_components_columns, 2):
    if i%2==0:
      if i != 0:
        pdf.savefig()
        plt.clf()
      i = 0
      f, axs = plt.subplots(1, 2, figsize=(20, 10))
      plt.subplots_adjust(wspace=0.4, hspace=0.8, bottom=0.17, top=0.93)
    texts = []
    for x, y, name in zip(target_components_df[c_f].values, target_components_df[c_s].values, df.columns):
      t = axs[i%2].text(x, y, name)
      texts.append(t)
    axs[i%2].scatter(target_components_df[c_f].values, target_components_df[c_s].values, alpha=0.8, s=10, color="red")
    axs[i%2].grid()
    axs[i%2].set_xlabel(c_f)
    axs[i%2].set_ylabel(c_s)
    adjust_text(texts, ax=axs[i%2], arrowprops=dict(arrowstyle='->', color='m'))
    i += 1
  pdf.savefig()
  plt.clf()
  pdf.close()