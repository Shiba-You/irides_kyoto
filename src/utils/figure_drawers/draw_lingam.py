
import lingam
from lingam.utils import make_dot

'''
固有ベクトル毎の累積寄与率のPDF出力
input :   df                    ::  <DataFrame>        元のデータから group, gender を除外して 正規化 & 欠損値処理 した dataFrame
          output_file_path      ::  <string>           PDFの出力先
          component             ::  <string>           対象の component 名
'''
def draw(df, output_file_path, component):
  model = lingam.DirectLiNGAM()
  model.fit(df)
  dot = make_dot(model.adjacency_matrix_, labels=list(df.columns))
  output_full_path = f"{output_file_path}_{component}"
  dot.render(output_full_path)
  return