import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import itertools
from utils.loggers.checkers import _multiple_checker
from .. import arange_data

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
    # plt.subplots_adjust(wspace=0.5, hspace=0.5)
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
    
    # plt.savefig(output_file_path+f"_{c_f}-{c_s}散布図.svg")
    pdf.savefig()
    plt.clf()
  pdf.close()