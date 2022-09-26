import re
import itertools
import lingam
from lingam.utils import make_dot, make_prior_knowledge

'''
固有ベクトル毎の累積寄与率のPDF出力
input :   df                    ::  <DataFrame>        元のデータから group, gender を除外して 正規化 & 欠損値処理 した dataFrame
          output_file_path      ::  <string>           PDFの出力先
          component             ::  <string>           対象の component 名
'''
TIMELINE_DICT = {
  'h2D': 0,
  '2D': 0,
  'hVRf0': 1,
  'VRf0': 1,
  'hVRf1': 2,
  'VRf1': 2,
  'hVR': 3,
  'VR': 3,
  'hBreak': 4,
  'Break': 4,
  'hVRf2': 5,
  'VRf2': 5,

}
# def draw(df, output_file_path, component, prior_relationship):
def draw(df, output_file_path, component):
  steps = []
  no_paths = []
  for column in df.columns:
    column = re.sub('.*_', '', column)
    column = re.sub('\n.*', '', column)
    steps.append(column)
  print(steps)
  for step1, step2 in itertools.permutations(enumerate(steps), 2):
    if TIMELINE_DICT[step1[1]] > TIMELINE_DICT[step2[1]]:
      no_paths.append((step1[0], step2[0]))
  print(no_paths)
  prior_knowledge = make_prior_knowledge(
    n_variables=len(steps),
    no_paths=no_paths
  )
  model = lingam.DirectLiNGAM(prior_knowledge=prior_knowledge)
  model.fit(df)
  dot = make_dot(model.adjacency_matrix_, labels=list(df.columns))
  output_full_path = f"{output_file_path}_{component}"
  dot.render(output_full_path)
  return