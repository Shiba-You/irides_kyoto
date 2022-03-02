#%%
'''
python @ 3.10.2
'''
from cmath import phase
from traceback import print_tb
from turtle import clear
import pandas as pd
import os.path
import numpy as np
import matplotlib
import japanize_matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import itertools
import fitz
from collections import deque
import requests

class make_chart:
  def __init__(self, input_file_name, output_file_name, output_chart_path, target_params):
    self.input_file_name = input_file_name
    self.output_file_name = output_file_name
    self.output_chart_path = output_chart_path
    self.df_all = pd.DataFrame()
    self.df_a = pd.DataFrame()
    self.df_b = pd.DataFrame()
    self.df_output = pd.DataFrame()
    self.df_relative_ratio = pd.DataFrame()
    self.df_standard_error = pd.DataFrame()
    self.methods = ["MEAN", "MAX", "MIN", "MED", "STD", "VAR"]
    self.group = ["A", "B"]
    self.color = ["red", "blue"]
    self.target_params = target_params
    self.dfs = []
    self.key = []

  def init_data(self):
    if os.path.exists(self.input_file_name):
      xls = pd.ExcelFile(self.input_file_name)
      sheets = xls.sheet_names
      input_sheet = sheets[2]
      self.df_all = pd.DataFrame(xls.parse(input_sheet))
      self.df_a = self.df_all.query('群 == "A"')
      self.df_b = self.df_all.query('群 == "B"')
      #! ID, 性別, 年齢, 群を除く
      self.key = list(self.df_all.columns)[4:] #! 01, 02 用
      # self.key = ["HF_実験教示", "HF_津波避難VR", "LF/HF_実験教示", "LF/HF_津波避難VR", "CVRR_実験教示", "CVRR_津波避難VR", "2. 教示後_唾液kU/l", "6. fantasy後_唾液kU/l", "2. 教示後_状態不安", "6. fantasy後_状態不安", "特性不安total", "誠実性", "情緒不安定性", "外向性", "開放性", "調和性", "楽観的自己感情"] #! 03 用
      self.dfs = [self.df_a, self.df_b]
    else:
      print("ファイルが存在しません．")
      exit()
    return
  
  def output_data(self, df, path):
    df.to_excel(path, sheet_name='result')
    return
  
  # def arange_IDs(self):
  #   # phases = ["実験前準備・評価", "実験教示", "テスト歩行", "津波避難VR準備", "津波避難VR", "5分休憩", "心理的安定化"] #! 全ての特徴量を抽出 (data/arange/01_all_params 用)
  #   phases = ["実験教示", "津波避難VR"] #! 実験中のみの特徴量とする (data/arange/02_experiment_params 用)
  #   df_output = pd.DataFrame()
  #   for i in range(1,32):
  #     if i == 28: continue
  #     id_path = os.path.join("../data/IDs", "ID{}/analyzer_ID{}_TimeData_加工.xlsx".format(i,i))
  #     xls = pd.ExcelFile(id_path)
  #     sheets = xls.sheet_names
  #     input_sheet = sheets[0]
  #     df_id = pd.DataFrame(xls.parse(input_sheet))
  #     df_row = pd.DataFrame()
  #     for phase in phases:
  #       tmp = pd.Series(df_id.query("Phase == '{}'".format(phase)).mean(), name=i)
  #       if len(df_row) == 0:
  #         df_row = pd.DataFrame([tmp])
  #         df_row = df_row.add_suffix("_{}".format(phase))
  #       else:
  #         df_tmp = pd.DataFrame([tmp])
  #         df_tmp = df_tmp.add_suffix("_{}".format(phase))
  #         df_row = pd.concat([df_row, df_tmp], axis=1)
  #     df_output = pd.concat([df_output, df_row])
  #   self.output_data(df_output, "../data/220209 IDs_only_experiment.xlsx")
  #   return

  def calc_params(self, output_flag):
    #! df_output の初期化
    cols = []
    for k in self.key:
      for x in self.group:
        cols.append(k+"_"+x)
    self.df_output = pd.DataFrame(columns=cols)
    for method in self.methods:
      tmp = np.arange(0)
      for k in self.key:
        for df in self.dfs:
          match method:
            case "MEAN":
              tmp = np.append(tmp, df[k].mean())
            case "MAX":
              tmp = np.append(tmp, df[k].max())
            case "MIN":
              tmp = np.append(tmp, df[k].min())
            case "MED":
              tmp = np.append(tmp, df[k].median())
            case "STD":
              tmp = np.append(tmp, df[k].std())
            case "VAR":
              tmp = np.append(tmp, df[k].var())
      match method:
        case "MEAN":
          self.df_output.loc["MEAN"] = tmp
        case "MAX":
          self.df_output.loc["MAX"] = tmp
        case "MIN":
          self.df_output.loc["MIN"] = tmp
        case "MED":
          self.df_output.loc["MED"] = tmp
        case "STD":
          self.df_output.loc["STD"] = tmp
        case "VAR":
          self.df_output.loc["VAR"] = tmp
    if output_flag:
      self.output_data(self.df_output.T, self.output_file_name)
    return

  def calc_relative_ratio(self):
    self.df_relative_ratio = pd.DataFrame(columns=self.key)
    for ratio in ["A/B", "B/A"]:
      tmp = np.arange(0)
      for k in self.key:
        tmp = np.append(tmp, self.df_output.at["MEAN", k+"_"+ratio[0]] / self.df_output.at["MEAN", k+"_"+ratio[-1]])
      self.df_relative_ratio.loc[ratio] = tmp
    self.key = self.df_relative_ratio.sort_values(self.target_params, axis=1, ascending=False).columns
    self.output_data(self.df_relative_ratio.sort_values(self.target_params, axis=1, ascending=False).T, "./result_relative.xlsx")
    # self.output_data(self.df_relative_ratio.T, "./result_relative.xlsx")
  
  def calc_standard_error(self):
    self.df_standard_error = pd.DataFrame(columns=self.key)
    df_tmp_a = pd.DataFrame(columns=self.key)
    df_tmp_b = pd.DataFrame(columns=self.key)
    for idx, df in enumerate(self.dfs):
      for k in self.key:
        std =  self.df_all[k].std()
        mean =  self.df_all[k].mean()
        _t = np.arange(0)
        for val in df[k]:
          _t = np.append(_t, (val-mean)/std)
        if idx == 0:
          df_tmp_a[k] = _t
        elif idx == 1:
          df_tmp_b[k] = _t
    for k in self.key:
      __t = np.arange(0)
      __t = np.append(__t, df_tmp_a[k].mean() - df_tmp_b[k].mean())
      self.df_standard_error[k] = __t
    self.df_standard_error = self.df_standard_error.rename(index={0: 'SE'})
    self.key = self.df_standard_error.sort_values(self.target_params, axis=1, ascending=False).columns
    self.output_data(self.df_standard_error.sort_values(self.target_params, axis=1, ascending=False).T, "./result_standard_error.xlsx")
    # self.output_data(self.df_standard_error.T, "./result_standard_error.xlsx")
  
  def make_box_hist_chart(self):
    pdf = PdfPages(self.output_chart_path+"_box_hist.pdf")
    # key = ['Acc_x_津波避難VR準備', 'Acc_x_5分休憩', 'LF_心理的安定化', 'LF_津波避難VR準備', 'LF/HF_津波避難VR準備', '唾液（kU/l）_実験教示', 'Acc_z_5分休憩', 'LF_テスト歩行', '2. 教示後_唾液kU/l', 'LF/HF_心理的安定化']
    i = 0
    sns.set(font='IPAexGothic', font_scale = 3)
    for k in self.key:
      if i%8==0:
        if i != 0:
          pdf.savefig()
          plt.clf()
        f, axs = plt.subplots(2, 4, figsize=(50, 25))
        plt.subplots_adjust(wspace=0.4, hspace=0.8, bottom=0.17, top=0.93)
      
      sns.boxplot(
        x="群",
        y=self.df_all[k],
        data=self.df_all,
        ax=axs[i//4%2, i%4]
      )
      i += 1
      for df, idx, color in zip(self.dfs, self.group, self.color):
        sns.distplot(
          df[k],
          color=color, 
          kde=True,
          label=idx,
          ax=axs[i//4%2, i%4]
        )
      i += 1
      if k == self.key[-1]:
        pdf.savefig()
        plt.clf()
    pdf.close()
  
  def make_scatter_chart(self):
    pdf = PdfPages(self.output_chart_path+"_scatter.pdf")
    for kx, ky in itertools.combinations(self.key, 2):
      for df, idx, color in zip(self.dfs, self.group, self.color):
        sns.regplot(
          x=kx,
          y=ky,
          data=df,
          color=color,
          ci=None,
          line_kws={'lw': 2},
          truncate=False,
          label=idx
        )
      plt.legend(fontsize=10)
      pdf.savefig()
      plt.clf()
    pdf.close()
    return
  
  def make_heatmap_chart(self):
    pdf = PdfPages(self.output_chart_path+"_heatmap.pdf")
    for df in self.dfs:
      df_house_corr = df.corr()
      sns.heatmap(
        df_house_corr,
        square=True,
        vmax=1,
        vmin=-1,
        center=0,
      )
      pdf.savefig()
      plt.clf()
    pdf.close()
    return
  
  def insert_text_output_pdf_fitz(self):
    pdf_file_path = self.output_chart_path+"_box_hist.pdf"
    all_calc_results = deque([])
    all_key = deque(self.key)
    match self.target_params:
      case "SE":
        suffix = ["SE"]
        df_target = self.df_standard_error
      case "A/B" | "B/A":
        suffix = ["A/B", "B/A"]
        df_target = self.df_relative_ratio
    rank = 0
    # for k in self.df_output.columns:
    #   all_calc_results.append(self.df_output[k].values)
    for k in self.key:
      for x in self.group:
        all_calc_results.append(self.df_output[k+"_"+x].values)
    def mm_to_pts(mm) -> float:
      return float(mm) / 0.352778
    # 既存PDFの読み取り
    reader = fitz.open(pdf_file_path)
    # 総ページ数
    total_page_num = reader.page_count
    # 新規PDFの作成
    writer = fitz.open()
    # 新規PDFの作成
    writer.insertPDF(reader)
    for page_num in range(total_page_num):    
      # 既存PDFの1ページを読み込む
      page = writer.loadPage(page_num)
      # 文字列の出力座標
      # 1, 2
      # 3, 4 の順に出力していることに注意
      # [タイトルx座標, タイトルy座標, パラメータx座標, パラメータy座標]
      index = [[60, 30, 60, 265], [670, 30, 670, 265], [60, 350, 60, 580], [670, 350, 670, 580]]
      for tx, ty, cx, cy in index:
        # 挿入位置(mmをptsに変えて指定)
        target_title_x, target_title_y = mm_to_pts(tx), mm_to_pts(ty)
        target_graph_x, target_graph_y = mm_to_pts(cx), mm_to_pts(cy)
        if len(all_calc_results) == 0:
          break
        A_param = all_calc_results.popleft()
        B_param = all_calc_results.popleft()
        title = all_key.popleft()
        rank += 1
        # 初期値は最左列
        insert_texts = ["\nA\nB"]
        for i, method in enumerate(self.methods):
          _t = "{}\n{:.3g}\n{:.3g}".format(method, A_param[i], B_param[i])
          insert_texts.append(_t)
        # calc_relative を追加
        for i, val in enumerate(suffix):
          _t = "{}\n{:.3g}".format(val, df_target.at[val ,title])
          insert_texts.append(_t)
        p = fitz.Point(target_title_x, target_title_y)  # start point of 1st line
        rc = page.insertText( p,  # bottom-left of 1st char
                              str(rank)+"__"+title,  # the text (honors '\n')
                              fontname="japan",  # the default font
                              fontsize=40,  # the default font size
                              rotate=0,  # also available: 90, 180, 270
                              color=(0,0,0),
                            )
        for idx, text in enumerate(insert_texts):
          p = fitz.Point(target_graph_x+idx*200 - (idx!=0)*100, target_graph_y)  # start point of 1st line
          if self.target_params in text:
            rc = page.insertText( p,  # bottom-left of 1st char
                                  text,  # the text (honors '\n')
                                  # fontname="cjk",  # the default font
                                  fontsize=40,  # the default font size
                                  rotate=0,  # also available: 90, 180, 270
                                  color=(1,0,0),
                                  
                                )
          else:
            rc = page.insertText( p,  # bottom-left of 1st char
                                  text,  # the text (honors '\n')
                                  # fontname="cjk",  # the default font
                                  fontsize=40,  # the default font size
                                  rotate=0,  # also available: 90, 180, 270
                                  color=(0,0,0)
                                )
    # 出力名
    output_name = self.output_chart_path + "_box_hist_with_calc_" + self.target_params + ".pdf"
    writer.save(output_name)
    return

  def notification(self):
    """
    LINEに通知する
    """
    line_notify_token = 'GOqN8UvbQwCcr4F3xx7CtaY5dw3z32M17SK9Y1Jv80x'
    line_notify_api = 'https://notify-api.line.me/api/notify'
    headers = {'Authorization': f'Bearer {line_notify_token}'}
    data = {'message': f'解析アルバイト: 完了'}
    requests.post(line_notify_api, headers = headers, data = data)


  def main(self):
    # self.arange_IDs()                   #! IDs/ から必要なデータを抽出
    self.init_data()                    #! data/arange から必要データを DataFrame に整形
    self.calc_params(False)             #! 特徴量の数値分析 { output_flag: excel に出力するか否か}
    self.calc_relative_ratio()          #! 相対比を整形
    # self.calc_standard_error()          #! 標準値の誤差を整形
    # self.make_box_hist_chart()          #! 箱ひげ図 / ヒストグラム 作図
    # self.make_scatter_chart()           #! 散布図 作成
    # self.make_heatmap_chart()           #! ヒートマップ 作成 （各特徴量の相関係数）
    # self.insert_text_output_pdf_fitz()  #! 分析結果を箱ひげ図 / ヒストグラムに加筆
    # self.notification()


if __name__ == "__main__":
  #? >>>> ここは変更する >>>>
  input_file_name = "220208 調査報告書+IDs.xlsx"
  output_file_name = "result.xlsx"
  output_chart_name = "result"
  dir_names = ["01_all_params", "02_experiment_params", "03_important_params"]
  this_dir = 0                                    #! {0: 01_all_params, 1: 02_experiment_params, 2: 03_important_params}
  target_params = "A/B"                            #! {A/B: Aの平均/Bの平均, B/A: Bの平均/Aの平均, SE: 標準値の誤差}
  #? <<<< ここは変更する <<<<
  input_file_path = os.path.join("../data/arange", dir_names[this_dir], input_file_name)
  output_file_path = os.path.join("../results/", dir_names[this_dir], output_file_name)
  output_chart_path = os.path.join("../results/", dir_names[this_dir], output_chart_name)
  mc = make_chart(input_file_path, output_file_path, output_chart_path, target_params)
  mc.main()


# %%
