import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PLOTS = "../data/plots"
RESULTS = "../data/data"

def plot_tree_size_dist(data, save=True, fn="treesize.pdf", plot=False):
    plt.hist(data['treesize'])
    plt.xlabel("Tree Size", size=14)
    plt.ylabel("Count", size=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    if save:
        plt.savefig(os.path.join(PLOTS, fn), format="pdf", bbox_inches="tight")
    if plot:
        plt.show()
    else:
        plt.clf()

def plot_tree_size_time(tree_df, save=True, fn="treesize_vs_time.pdf", plot=False):
    plt.scatter(tree_df.treesize, tree_df.time, alpha=0.25)
    plt.plot(np.unique(tree_df.treesize), np.poly1d(np.polyfit(tree_df.treesize, tree_df.time, 1))(np.unique(tree_df.treesize)), "red", linewidth=2)
    plt.xlabel("Average size of tree in batch", size=14)
    plt.ylabel("time (s)", size=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    if save:
        plt.savefig(os.path.join(PLOTS, "treesize_vs_time.pdf"), format="pdf", bbox_inches="tight")

    if plot:
        plt.show()
    plt.clf()

def load_results(fn):
    with open(os.path.join(RESULTS, fn)) as fd:
        data = json.load(fd)
    return data

def main():

    cuda_nofold = load_results("results_fold-False-True-full.json")
    cuda_fold = load_results("results_fold-True-True-full.json")
    nocuda_nofold = load_results("results_fold-False-False-full.json")
    nocuda_fold = load_results("results_fold-True-False-full.json")


    cuda_nofold_df = pd.DataFrame(cuda_nofold)
    cuda_fold_df = pd.DataFrame(cuda_fold)
    nocuda_nofold_df = pd.DataFrame(nocuda_nofold)
    nocuda_fold_df = pd.DataFrame(nocuda_fold)

    ############################
    ## Median inference times ##
    ############################
    df1 = cuda_nofold_df.groupby('batch_size').median().rename({"time":"time_tf"}, axis="columns")
    df2 = cuda_fold_df.groupby('batch_size').median().rename({"time":"time_tt"}, axis="columns")
    df3 = nocuda_nofold_df.groupby('batch_size').median().rename({"time":"time_ff"}, axis="columns")
    df4 = nocuda_fold_df.groupby('batch_size').median().rename({"time":"time_ft"}, axis="columns")

    print("cuda: True, fold: False\n")
    print(df1)

    print("\ncuda: True, fold: True")
    print(df2)

    print("\ncuda: False, fold: False")
    print(df3)

    print("\ncuda: False, fold: True")
    print(df4)

    summary1 = df1.join((df2.time_tt, df3.time_ff, df4.time_ft))

    summary1['cuda_comp'] = summary1.time_tf / summary1.time_tt
    summary1['cpu_comp'] = summary1.time_ff / summary1.time_ft

    summary1['fold_comp'] = summary1.time_ft / summary1.time_tt
    summary1['nofold_comp'] = summary1.time_ff / summary1.time_tf

    print(summary1)

    ##################################
    ## Distribution of testing data ##
    ##################################

    plot_tree_size_dist(load_results("results_fold-False-1-False-backup.json"))

    #######################
    ## Tree size vs time ##
    #######################

    tree_df = cuda_fold_df.loc[cuda_fold_df.batch_size == 1]
    plot_tree_size_time(tree_df)

if __name__ == "__main__":
    main()
