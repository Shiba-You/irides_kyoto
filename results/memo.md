# ディレクトリ構成
.
├── 01_all_params
│   ├── result_box_hist.pdf
│   ├── result_scatter_hist.pdf
│   └── result.xlsx
├── 02_experiment_params
│   ├── result_box_hist.pdf
│   ├── result_scatter_hist.pdf
│   └── result.xlsx
├── 03_important_params
│   ├── result_box_hist.pdf
│   ├── result_scatter_hist.pdf
│   └── result.xlsx
└── 99_data
    ├── 01_all_params_data.xlsx
    └── 02_experiment_params_data.xlsx

# 各ファイル詳細
* `01_all_params/`
  * 概要
    * *`./99_data/01_all_params_data.xlsx`* に集約された特徴量を対象に分析
  * ファイル詳細
    * `result_box_hist.pdf`
      * 対象の箱ひげ図及び，ヒストグラムを図示
        * 箱髭図
          * 縦軸：各特徴量値
          * 横軸：各群
          * 菱形は外れ値を示す
        * ヒストグラム
          * 縦軸：密度（総面積が1になるように正規化した時の各値）
          * 横軸：各特徴量値
          * カーネル密度推定を実施
    * `result_scatter_hist.pdf`
      * 対象の散布図及び，線形回帰モデルを図示
        * 散布図
          * 縦軸・横軸共に，各特徴量の散布を示す
        * 線形回帰モデル
          * 信頼区間0%の線形回帰を示す
    * `result.xlsx`
      * 対象における以下の値を集約
        { MEAN: 平均, MAX: 最大値, MIN: 最小値, MED: 中央値, STD: 標準偏差, VAR: 分散}
      * それぞれの特徴量は，[特徴量名]_{A, B, All}と命名
        { A: A群, B: B群, All: A群+B群 }
* `02_experiment_params`
  * 概要
    * *`./99_data/02_experiment_params_data.xlsx`* に集約された特徴量を対象に分析
  * ファイル詳細
    * `01_all_params/`と同じ
* `03_impotant_params`
  * 概要
    * *`./99_data/03_important_params_data.xlsx`* に集約された特徴量を対象に分析
  * ファイル詳細
    * `01_all_params/`と同じ
* `99_data/`
  * 概要
    * 今回の分析で用いたデータ群
  * ファイル詳細
    * `01_all_params_data.xlsx`
      * 元データにおける `02_唾液・不安・調査票回答データ.xlsx` に記載されている橙色に網掛けされた特徴量及び，元データにおける `各IDフォルダ/ID*/analyzer_ID*_TimeData_加工.xlsx` に記載された特徴量の*各Phase*ごとの平均値を集約
    * `02_experiment_params_data.xlsx`
      * 元データにおける `各IDフォルダ/ID*/analyzer_ID*_TimeData_加工.xlsx` に記載された特徴量のうち，*実験教示*と*津波避難VR*時の平均値のみを集約
    * `03_important_params_data.xlsx`
      * 前回の議論(220201)で有意であるとされた特徴量のみ抽出
        * 該当特徴量
          * 心拍（HF, LF/HF, CVRR)[教示後, fantasy後]
          * 2.教示後_唾液kU/l
          * 6. fantasy後_唾液kU/l
          * 2. 教示後_状態不安
          * 6. fantasy後_状態不安
          * 特性不安total
          * 誠実性
          * 情緒不安定性
          * 外向性
          * 開放性
          * 調和性
          * 楽観的自己感情
    