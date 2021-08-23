#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description:
    Evaluates the performance of PPI predictions by plotting ROC and Precision-Recall curves.
    Plots prevalence-corrected curves for hypothetically imbalanced data.
    
    Requires files to have no header and be whitespace-separated (.tsv).
    
Usage:
    python evaluate_ppi.py -s SCORES/ -l labels.tsv -d 0.5 -r RESULTS/
    
    Input arguements:
        -s <str> Can be either:
            - a directory path where tested PPI prediction k-fold subset files exist (file names must contain the word 'fold' and a number)
            - a file path for the tested PPI predictions
        -l <str> is the file path to the labelled PPIs
        -d <float> is the hypothetical imbalance ratio of positives/all PPIs, default is 0.5
            e.g.
            - balanced data would be 0.5 (number of positives == number of negatives)
            - imbalanced data where 1 positive for every 100 negatives would be 1/101, so 0.0099
        -r <str> is a directory path for saving the results, default is current directory
        -n <str> name for plot titles and files, default is result directory name

@author: Eric Arezza
"""

__all__ = ['recalculate_precision',
           'get_matching_pairs',
           ]

__version__ = '1.0'
__author__ = 'Eric Arezza'

import os
import argparse
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt


describe_help = 'python evaluate_ppi.py -s SCORES/ -l labels.tsv -d 0.5 -r RESULTS/'
parser = argparse.ArgumentParser(description=describe_help)
parser.add_argument('-s', '--scores', help='Full path to scored PPIs (directory to cross-validation files or single test file path)', type=str)
parser.add_argument('-l', '--labels', help='Full path to labelled PPIs (.tsv file, no header, using labels 0 (neg) and 1 (pos))', type=str)
parser.add_argument('-r', '--results', help='Path to directory for saving evaluation files and plots', 
                    type=str, default=os.getcwd()+'/EVALUATION/')
parser.add_argument('-d', '--delta', help='Imbalance ratio as positives/total (e.g. balanced = 0.5) for estimate of performance on hypothetical imbalanced data', 
                    type=float, nargs=1, required=False, default=0.5)
parser.add_argument('-n', '--name', help='Name for saving files, default basename will be results directory name', 
                    type=str, default='')
args = parser.parse_args()

RESULTS_DIR = args.results
if not os.path.exists(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)
    
if args.name == '':
    args.name = RESULTS_DIR.split('/')[-2].lower()

# Display ratio of positives:negatives
RATIO = '1:' + str(int((1/args.delta) - 1))

# Calculate estimate for prevalence-corrected precision on imbalanced data
def recalculate_precision(df, precision, thresholds):
    delta = 2*args.delta - 1
    new_precision = precision.copy()
    for t in range(0, len(thresholds)):
        tn, fp, fn, tp = metrics.confusion_matrix(df[1], (df[0] >= thresholds[t]).astype(int)).ravel()
        tp = tp*(1 + delta)
        fp = fp*(1 - delta)
        new_precision[t] = tp/(tp+fp)
    return new_precision

# Get labels for PPIs
# NOTE: For SPRINT, labels are already in column in df
def get_matching_pairs(df_1, df_2):
    # Get matches using PPI ordering of smaller df
    if df_1.shape > df_2.shape:
        df_test = df_1.copy()
        df_train = df_2.copy()
    else:
        df_test = df_2.copy()
        df_train = df_1.copy()
    
    # Include all PPIs where A-B is B-A
    df_test_rev = df_test.copy()
    df_test_rev[[df_test_rev.columns[0], df_test_rev.columns[1]]] = df_test_rev[[df_test_rev.columns[1], df_test_rev.columns[0]]]
    df = df_test.append(df_test_rev)
    df = df.drop_duplicates(subset=[df.columns[0], df.columns[1]])
    df.reset_index(drop=True, inplace=True)
    
    # Merge to match all PPIs found between df
    matches = df_train.merge(df, on=[df.columns[0], df.columns[1]])
    matches = matches.drop_duplicates(subset=[df.columns[0], df.columns[1]])
    matches.reset_index(drop=True, inplace=True)
    # Returns as <ProteinA> <ProteinB> <label> <score>
    return matches
    

if __name__ == '__main__':

    # Get PPI scores
    if os.path.isdir(args.scores):
        # For cross-validation tested PPI subsets
        files = os.listdir(path=args.scores)
        files = [ x for x in files if 'fold' in x and 'pos' not in x and 'neg' not in x and 'train' not in x.lower() ]
        files.sort()
    else:
        # For single file tested PPIs
        files = [ args.scores.split('/')[-1] ]
        
    # Get PPI labels for entire dataset
    df_labels = pd.read_csv(args.labels, delim_whitespace=True, header=None)
    
    # Metrics for evaluation
    # For ROC curve
    tprs = {}
    roc_aucs = {}
    fprs = {}
    # For PR curve
    precisions = {}
    pr_aucs = {}
    recalls = {}

    df_pred_avg = pd.DataFrame()
    fold = 0
    for k in files:
        
        # Isolate k-fold subset
        print('===== Fold - %s ====='%fold)
        
        # Read predictions for k-fold
        df_pred = pd.read_csv(args.scores + k, delim_whitespace=True, header=None)
        
        # Get labels from cross-validation subsets
        if '_SPRINT_' not in k and ('SPRINT' not in args.scores and 'CME' not in args.scores):
            df_pred = get_matching_pairs(df_pred, df_labels)
            df_pred.drop(columns=[0, 1], inplace=True)
            df_pred.rename(columns={'2_x': 0, '2_y': 1}, inplace=True)

        df_pred_avg = df_pred_avg.append(df_pred)
        
        # Evaluate k-fold performance and adjust for hypothetical imbalance
        precision, recall, thresholds = metrics.precision_recall_curve(df_pred[1], df_pred[0])
        precision = recalculate_precision(df_pred, precision, thresholds)
        fpr, tpr, __ = metrics.roc_curve(df_pred[1], df_pred[0])
        if args.delta == 0.5:
            pr_auc = metrics.average_precision_score(df_pred[1], df_pred[0])
        else:
            pr_auc = metrics.auc(recall, precision)
        roc_auc = metrics.roc_auc_score(df_pred[1], df_pred[0])
        print('auc_roc=', roc_auc, '\nauc_pr=', pr_auc)
        
        # Add k-fold performance for overall average performance
        tprs[fold] = tpr
        fprs[fold] = fpr
        roc_aucs[fold] = roc_auc
        precisions[fold] = precision
        recalls[fold] = recall
        pr_aucs[fold] = pr_auc
    
        fold += 1
        
    # Get average performance across all k-folds
    precision, recall, thresholds = metrics.precision_recall_curve(df_pred_avg[1], df_pred_avg[0])
    precision = recalculate_precision(df_pred_avg, precision, thresholds)
    fpr, tpr, __ = metrics.roc_curve(df_pred_avg[1], df_pred_avg[0])
    if args.delta == 0.5:
        pr_auc = metrics.average_precision_score(df_pred_avg[1], df_pred_avg[0])
    else:
        pr_auc = metrics.auc(recall, precision)
    roc_auc = metrics.roc_auc_score(df_pred_avg[1], df_pred_avg[0])
    
    # Plot and save curves
    # Precision-Recall
    plt.figure
    plt.plot(recall, precision, color='black', label='AUC = %0.4f +/- %0.4f' % (pr_auc, np.std(np.fromiter(pr_aucs.values(), dtype=float))))
    for i in recalls.keys():
        plt.plot(recalls[i], precisions[i], alpha=0.25)
    plt.xlabel('Recall')
    plt.ylabel('Precision') 
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.title("Precision-Recall Curve - %s %s"%(args.name.capitalize(), RATIO))
    plt.legend(loc='lower right', handlelength=0)
    plt.savefig(RESULTS_DIR + args.name + '_PR.png', format='png')
    plt.close()
    
    # ROC
    plt.figure
    plt.plot(fpr, tpr, color='black', label='AUC = %0.4f +/- %0.4f' % (roc_auc, np.std(np.fromiter(roc_aucs.values(), dtype=float))))
    for i in fprs.keys():
        plt.plot(fprs[i], tprs[i], alpha=0.25)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.title("ROC Curve - %s %s"%(args.name.capitalize(), RATIO))
    plt.legend(loc='lower right', handlelength=0)
    plt.savefig(RESULTS_DIR + args.name + '_ROC.png', format='png')
    plt.close()