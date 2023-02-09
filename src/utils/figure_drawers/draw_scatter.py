import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import itertools
from utils.loggers.checker import _multiple_checker
import pandas as pd
from .. import arange_data
from sklearn.linear_model import LinearRegression

def draw(output_file_path, target_features_df, target_components_columns, outliers = True):
  pdf = PdfPages(output_file_path+"_成分散布図.pdf")
  i = 0
  for c_f, c_s in itertools.combinations(target_components_columns, 2):
    if i%2==0:
      if i != 0:
        pdf.savefig()
        plt.clf()
      i = 0
      f, axs = plt.subplots(1, 2, figsize=(16, 8))
      plt.subplots_adjust(wspace=0.4, hspace=0.8, bottom=0.17, top=0.93)
    if outliers:
      df_c = arange_data._drop_outliers(target_features_df[["group", "gender", c_f, c_s]], c_f, c_s)
    else:
      df_c = target_features_df[["group", "gender", c_f, c_s]]
    feature_A = df_c.query('group == "A"')
    feature_A_M = feature_A.query('gender == "男"')
    feature_A_W = feature_A.query('gender == "女"')
    feature_B = df_c.query('group == "B"')
    feature_B_M = feature_B.query('gender == "男"')
    feature_B_W = feature_B.query('gender == "女"')

    c_f_max, c_f_min, c_f_a_ave, c_f_b_ave, dx = df_c[c_f].max(), df_c[c_f].min(), feature_A[c_f].mean(), feature_B[c_f].mean(), (df_c[c_f].max() - df_c[c_f].min())/100*5
    c_s_max, c_s_min, c_s_a_ave, c_s_b_ave, dy = df_c[c_s].max(), df_c[c_s].min(), feature_A[c_s].mean(), feature_B[c_s].mean(), (df_c[c_s].max() - df_c[c_s].min())/100*5

    _multiple_checker(c_f, c_s, feature_A_M, feature_A_W, feature_B_M, feature_B_W, c_f_a_ave, c_f_b_ave, c_s_a_ave, c_s_b_ave)

    axs[i%2].scatter(feature_A_M[c_f].values, feature_A_M[c_s].values, alpha=0.8, s=100, c="red", label="A-male", marker="o")
    axs[i%2].scatter(feature_A_W[c_f].values, feature_A_W[c_s].values, alpha=0.8, s=100, c="red", label="A-female", marker="^")
    axs[i%2].scatter(feature_B_M[c_f].values, feature_B_M[c_s].values, alpha=0.8, s=100, c="blue", label="B-male", marker="o")
    axs[i%2].scatter(feature_B_W[c_f].values, feature_B_W[c_s].values, alpha=0.8, s=100, c="blue", label="B-female", marker="^")
    axs[i%2].vlines(c_f_a_ave, c_s_min, c_s_max, "red", linestyles='dashed', label='A-Average')
    axs[i%2].vlines(c_f_b_ave, c_s_min, c_s_max, "blue", linestyles='dashed', label='B-Average')
    axs[i%2].hlines(c_s_a_ave, c_f_min, c_f_max, "red", linestyles='dashed')
    axs[i%2].hlines(c_s_b_ave, c_f_min, c_f_max, "blue", linestyles='dashed')
    plt.xlim(c_f_min-dx, c_f_max+dx)
    plt.ylim(c_s_min-dy, c_s_max+dy)
    axs[i%2].legend(fontsize="x-large")
    axs[i%2].set_xlabel(c_f)
    axs[i%2].set_ylabel(c_s)

    i += 1
  pdf.savefig()
  plt.clf()
  pdf.close()

def draw_out_legend(output_file_path, target_features_df, target_components_columns, outliers = True):
  pdf = PdfPages(output_file_path+"_成分散布図(外凡例).pdf")
  for c_f, c_s in itertools.combinations(target_components_columns, 2):
    f, axs = plt.subplots(1, 1, figsize=(8, 8))
    if outliers:
      df_c = arange_data._drop_outliers(target_features_df[["group", "gender", c_f, c_s]], c_f, c_s)
    else:
      df_c = target_features_df[["group", "gender", c_f, c_s]]
    feature_A = df_c.query('group == "A"')
    feature_A_M = feature_A.query('gender == "男"')
    feature_A_W = feature_A.query('gender == "女"')
    feature_B = df_c.query('group == "B"')
    feature_B_M = feature_B.query('gender == "男"')
    feature_B_W = feature_B.query('gender == "女"')

    c_f_max, c_f_min, c_f_a_ave, c_f_b_ave, dx = df_c[c_f].max(), df_c[c_f].min(), feature_A[c_f].mean(), feature_B[c_f].mean(), (df_c[c_f].max() - df_c[c_f].min())/100*5
    c_s_max, c_s_min, c_s_a_ave, c_s_b_ave, dy = df_c[c_s].max(), df_c[c_s].min(), feature_A[c_s].mean(), feature_B[c_s].mean(), (df_c[c_s].max() - df_c[c_s].min())/100*5

    _multiple_checker(c_f, c_s, feature_A_M, feature_A_W, feature_B_M, feature_B_W, c_f_a_ave, c_f_b_ave, c_s_a_ave, c_s_b_ave)

    axs.scatter(feature_A_M[c_f].values, feature_A_M[c_s].values, alpha=0.8, s=100, c="red", label="A-male", marker="o")
    axs.scatter(feature_A_W[c_f].values, feature_A_W[c_s].values, alpha=0.8, s=100, c="red", label="A-female", marker="^")
    axs.scatter(feature_B_M[c_f].values, feature_B_M[c_s].values, alpha=0.8, s=100, c="blue", label="B-male", marker="o")
    axs.scatter(feature_B_W[c_f].values, feature_B_W[c_s].values, alpha=0.8, s=100, c="blue", label="B-female", marker="^")
    axs.vlines(c_f_a_ave, c_s_min, c_s_max, "red", linestyles='dashed', label='A-Average')
    axs.vlines(c_f_b_ave, c_s_min, c_s_max, "blue", linestyles='dashed', label='B-Average')
    axs.hlines(c_s_a_ave, c_f_min, c_f_max, "red", linestyles='dashed')
    axs.hlines(c_s_b_ave, c_f_min, c_f_max, "blue", linestyles='dashed')
    plt.xlim(c_f_min-dx, c_f_max+dx)
    plt.ylim(c_s_min-dy, c_s_max+dy)
    axs.set_xlabel(c_f)
    axs.set_ylabel(c_s)

    box = axs.get_position()
    print(box.x0, box.y0)
    axs.set_position([box.x0*1.5, box.y0*1.5, box.width * 0.8, box.height*0.8])
    # Put a legend to the right of the current axis
    axs.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    pdf.savefig()
    plt.clf()
  pdf.close()

def draw_out_legend_2d_3d(output_file_path, feature, gender_and_group):
  pdf = PdfPages(output_file_path+"_2Dvs3D散布図.pdf")
  df_ndarray = feature.to_numpy()
  df = pd.DataFrame({"feature": feature.columns[0], "value": df_ndarray[:,0], "action": gender_and_group['group'], "gender": gender_and_group['gender']})
  for i in range(1, 20):
    df_tmp = pd.DataFrame({"feature": feature.columns[i], "value": df_ndarray[:,i], "action": gender_and_group['group'], "gender": gender_and_group['gender']})
    df = pd.concat([df, df_tmp])
  feature_columns = df["feature"].unique()
  for _f in feature_columns:
    lr_m = LinearRegression()
    lr_w = LinearRegression()
    lr_a = LinearRegression()

    plt.rcParams["font.size"] = 20

    f, axs = plt.subplots(1, 1, figsize=(8, 8))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    
    df_crt = pd.DataFrame({
      "value"  : df.query(f"feature == '{_f}'")['value'],
      "action" : df.query(f"feature == '{_f}'")["action"],
      "gender" : df.query(f"feature == '{_f}'")["gender"]
    })
    
    feature_2d   = df_crt.query('action == "2d"')
    feature_3d   = df_crt.query('action == "3d"')

    val_3d = feature_3d[["value"]].reset_index()
    feature_mix_d = pd.concat([feature_2d, val_3d.loc[:, "value"]], axis=1)
    feature_mix_d = feature_mix_d.set_axis(['dim2', 'action', 'gender', 'dim3'], axis=1)
    feature_mix_d = feature_mix_d[["gender", "dim2", "dim3"]]

    feature_mix_d = arange_data._drop_outliers(feature_mix_d[["gender", "dim2", "dim3"]], "dim2", "dim3")

    feature_M = feature_mix_d.query('gender == "男"')
    feature_W = feature_mix_d.query('gender == "女"')

    axs.scatter(feature_M["dim2"].values, feature_M["dim3"].values, alpha=0.8, s=100, c="red", label="male", marker="o")
    axs.scatter(feature_W["dim2"].values, feature_W["dim3"].values, alpha=0.8, s=100, c="blue", label="female", marker="^")

    X_m, Y_m = feature_M["dim2"].values.reshape(-1,1), feature_M["dim3"].values.reshape(-1,1)
    X_w, Y_w = feature_W["dim2"].values.reshape(-1,1), feature_W["dim3"].values.reshape(-1,1)
    X_a, Y_a = feature_mix_d["dim2"].values.reshape(-1,1), feature_mix_d["dim3"].values.reshape(-1,1)
    lr_m.fit(X_m, Y_m)
    lr_w.fit(X_w, Y_w)
    lr_a.fit(X_a, Y_a)

    axs.plot(X_m, lr_m.predict(X_m), c="red")
    axs.plot(X_w, lr_w.predict(X_w), c="blue")
    axs.plot(X_a, lr_a.predict(X_a), c="black")

    print('特徴量　: ', _f)
    print('回帰係数: ', lr_a.coef_[0]) # 説明変数の係数を出力
    print('切片　　: ', lr_a.intercept_) # 切片を出力
    print('')


    # axs.set_title(f"{_f}")

    feature_2d_max, feature_2d_min, dx = feature_mix_d["dim2"].max(), feature_mix_d["dim2"].min(), (feature_mix_d["dim2"].max() - feature_mix_d["dim2"].min())/100*5
    feature_3d_max, feature_3d_min, dy = feature_mix_d["dim3"].max(), feature_mix_d["dim3"].min(), (feature_mix_d["dim3"].max() - feature_mix_d["dim3"].min())/100*5


    axs.set_xlim(feature_2d_min-dx, feature_2d_max+dx)
    axs.set_ylim(feature_2d_min-dy, feature_2d_max+dy)

    axs.set_xlabel("2D")
    axs.set_ylabel("3D")

    box = axs.get_position()
    axs.set_position([box.x0*1.5, box.y0*1.5, box.width * 0.8, box.height*0.8])
    _f = _f.replace('/', '').replace('_', '').replace('saliva', '').replace('(', '').replace(')', '')
    plt.savefig(f"scatter_{_f}.png")
    
    # axs.legend(loc='center left', bbox_to_anchor=(1, 0.5))
  
    pdf.savefig()
    plt.clf()
  pdf.close()