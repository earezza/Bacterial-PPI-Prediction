#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description:
    Compares the performance of PPI prediction results between methods.
    Performs comparisons by significance/hypothesis testing using ANOVA first, then two-tailed t-tests.
    Writes to file with comparison results.
    
    Requires files in PPI results to have no header and be whitespace-separated (.tsv).
    
Usage:
    eg.
    python compare_performance.py -s SCORES_1/ SCORES_2/ SCORES_3/ -l labels_1.tsv labels_2.tsv labels_3.tsv -d 0.5 -r RESULTS/
    python compare_performance.py -s SCORES_1.tsv SCORES_2.tsv SCORES_3.tsv -l labels.tsv -d 0.5 -r RESULTS/ -m auc_pr -t paired
    
    Input arguements:
        -s list of <str> Can be either:
            - a directory path where tested PPI prediction k-fold subset files exist (file names must contain the word 'prediction' and a number)
            - a file path for the tested PPI predictions
        -l list of <str> is the file path to the labelled PPIs
            - order of provided list corresponds to order of provided scores list
            - e.g. SCORES_1 predictions will be evaluated against labels_1.tsv, then SCORES_2 with labels_2.tsv
            - if only one file of labels is provided, all scores will be evaluated against those labels
        -d <float> is the hypothetical imbalance ratio of positives/all PPIs, default is 0.5
            e.g.
            - balanced data would be 0.5 (number of positives == number of negatives)
            - imbalanced data where 1 positive for every 100 negatives would be 1/101, so 0.0099
        -r <str> is a directory path for saving the results, default is current directory + 'COMPARISONS/'
        -n <str> name for saving files, default is result directory name
        -m <str> metric used in significance tests to compare performance, default is area under precision-recall curve
        -t <str> type of two-tailed t-test performed (paired or independent), default is independent

@author: Eric Arezza
"""

__all__ = ['recalculate_precision',
           'recalculate_metrics_to_imbalance',
           'get_matching_pairs',
           'get_metrics',
           'test_anova',
           'test_t',
           ]

__version__ = '1.0'
__author__ = 'Eric Arezza'

import os, sys
import argparse
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, ttest_ind, ttest_rel
import statsmodels.api as sm
from statsmodels.formula.api import ols
from itertools import combinations
import time

describe_help = 'python compare_performance.py -s SCORES_1/ SCORES_2 -l labels.tsv -d 0.5 -r RESULTS/ -n scores1_vs_scores2'
parser = argparse.ArgumentParser(description=describe_help)
parser.add_argument('-s', '--scores', help='Full path to scored PPIs (directory to cross-validation files or single test file path)',
                    nargs="+", type=str)
parser.add_argument('-l', '--labels', help='Full path to labelled PPIs (.tsv file, no header, using labels 0 (neg) and 1 (pos))'
                    , nargs="+", type=str)
parser.add_argument('-r', '--results', help='Path to directory for saving files', 
                    type=str, default=os.getcwd()+'/COMPARISONS/')
parser.add_argument('-d', '--delta', help='Imbalance ratio as positives/total (e.g. balanced = 0.5) for estimate of performance on hypothetical imbalanced data', 
                    type=float, default=0.5)
parser.add_argument('-n', '--name', help='Name for saving files, default basename will be results directory name', 
                    type=str, default='')
parser.add_argument('-m', '--metric', help='Metric used to compare performance', 
                    type=str, default='auc_pr', choices=['auc_pr', 'auc_roc', 'precision', 'recall', 'accuracy', 'specificity', 'f1', 'mcc'])
parser.add_argument('-t', '--ttest_type', help='Paired if same samples tested under variable, independent if different samples tested under variable', 
                    type=str, default='ind', choices=['ind', 'paired'])
args = parser.parse_args()

RESULTS_DIR = args.results
if not os.path.exists(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)
if args.name == '':
    args.name = RESULTS_DIR.split('/')[-2].lower().capitalize() + '_' + args.labels.split('/')[-1].split('.')[0]

# Calculate estimate for prevalence-corrected precision on imbalanced data
def recalculate_precision(df, precision, thresholds, d):
    delta = 2*d - 1
    new_precision = precision.copy()
    for t in range(0, len(thresholds)):
        tn, fp, fn, tp = metrics.confusion_matrix(df[1], (df[0] >= thresholds[t]).astype(int)).ravel()
        lpp = tp/(tp+fn)
        lnn = tn/(tn+fp)
        if d != 0.5:
            new_precision[t] = (lpp*(1 + delta)) / ( (lpp*(1 + delta)) + ((1 - lnn)*(1 - delta)) )
        else:
            new_precision[t] = (lpp)/(lpp + (1-lnn))
    return new_precision

# Recalculate metrics for imbalanced classification where d is num_positives/(num_positives + num_negatives)
def recalculate_metrics_to_imbalance(tp, tn, fp, fn, d):
    delta = 2*d - 1
    # recall and specificity are unchanged
    lpp = tp/(tp+fn)
    lnn = tn/(tn+fp)
    if d != 0.5:
        accuracy = lpp*((1 + delta)/2) + lnn*((1 - delta)/2)
        precision = (lpp*(1 + delta)) / ( (lpp*(1 + delta)) + ((1 - lnn)*(1 - delta)) )
        f1 = (2*lpp*(1 + delta)) / ( ((1+lpp)*(1+delta)) + ((1-lnn)*(1-delta)) )
        mcc = (lpp+lnn-1) / np.sqrt((lpp + (1-lnn)*( (1-delta)/(1+delta) ) )*( lnn + (1-lpp)*( (1+delta)/(1-delta) ) ))
    else:
        accuracy = (lpp + lnn)/2
        precision = (lpp)/(lpp + (1-lnn))
        f1 = (2*lpp)/(2 + lpp - lnn)
        mcc = (lpp + lnn - 1) / np.sqrt( (lpp + (1-lnn))*(lnn + (1-lpp)) )
        
    return round(accuracy, 5), round(precision, 5), round(lpp, 5), round(lnn, 5), round(f1, 5), round(mcc, 5)

# Get labels for PPIs
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

def get_metrics(scores, labels, delta):
    print('Calculating performance for predictions:\n\t%s\nUsing labels:\n\t%s\nDelta:\t%s'%(scores, labels, delta))
    # Get PPI scores
    if os.path.isdir(scores):
        # For cross-validation tested PPI subsets
        files = os.listdir(path=scores)
        files = [ x for x in files if 'prediction' in x and '.pos' not in x and '.neg' not in x ]
        files.sort()
    else:
        # For single file tested PPIs
        files = [ scores.split('/')[-1] ]
        
    # Get PPI labels for entire dataset
    df_labels = pd.read_csv(labels, delim_whitespace=True, header=None)
    
    # Metrics for evaluation
    # For ROC curve
    tprs = {}
    roc_aucs = {}
    fprs = {}
    # For PR curve
    precisions = {}
    pr_aucs = {}
    recalls = {}

    # Additional metrics
    fold_accuracy = []
    fold_precision = []
    fold_recall = []
    fold_specificity = []
    fold_f1 = []
    fold_mcc = []

    df_pred_total = pd.DataFrame()
    fold = 0
    for k in files:
        
        # Isolate k-fold subset
        print('\n===== Fold - %s ====='%fold)
        
        # Read predictions for k-fold set or single test set
        if os.path.isdir(scores):
            df_pred = pd.read_csv(scores + k, delim_whitespace=True, header=None)
        else:
            df_pred = pd.read_csv(scores, delim_whitespace=True, header=None)
        
        # Remove any extra columns if exists to prevent subsequent problems in functions
        # predictions files should be ProteinA ProteinB Score
        if df_pred.shape[1] > 3:
            df_pred.drop(columns=df_pred.columns[3:].tolist(), inplace=True)
        
        # Get matching PPI labels for predictions
        #if '_SPRINT_' not in k and ('SPRINT' not in args.scores and 'CME' not in args.scores):
        df_pred = get_matching_pairs(df_pred, df_labels)
        df_pred.drop(columns=[0, 1], inplace=True)
        df_pred.rename(columns={'2_x': 0, '2_y': 1}, inplace=True)
        #df_pred[[0, 1]] = df_pred[[1, 0]]
        
        df_pred_total = df_pred_total.append(df_pred)
        
        # Get other metrics at 0.5 threshold if predictions are probabilities (0 to 1) i.e. not SPRINT predictions
        if df_pred[0].min() >= 0 and df_pred[0].max() <= 1:# and 'SPRINT' not in scores:
            
            tn, fp, fn, tp = metrics.confusion_matrix(df_pred[1], (df_pred[0] + 1e-12).round()).ravel()
            print('TP = %0.0f \nFP = %0.0f \nTN = %0.0f \nFN = %0.0f'%(tp, fp, tn, fn))
            print('Total_samples = %s'%(tn+fp+fn+tp))
            # For imbalanced classification metrics
            if delta != 0.5:
                accuracy, precision, recall, specificity, f1, mcc = recalculate_metrics_to_imbalance(tp, tn, fp, fn, delta)
                if np.isnan(accuracy) == False:
                    fold_accuracy.append(accuracy)
                if np.isnan(precision) == False:
                    fold_precision.append(precision)
                if np.isnan(recall) == False:
                    fold_recall.append(recall)
                if np.isnan(specificity) == False:
                    fold_specificity.append(specificity)
                if np.isnan(f1) == False:
                    fold_f1.append(f1)
                if np.isnan(mcc) == False:
                    fold_mcc.append(mcc)
            else:
                try:
                    accuracy = round((tp+tn)/(tp+fp+tn+fn), 5)
                    fold_accuracy.append(accuracy)
                except ZeroDivisionError:
                    accuracy = np.nan
                try:
                    precision = round(tp/(tp+fp), 5)
                    fold_precision.append(precision)
                except ZeroDivisionError:
                    precision = np.nan
                try:
                    recall = round(tp/(tp+fn), 5)
                    fold_recall.append(recall)
                except ZeroDivisionError:
                    recall = np.nan
                try:
                    specificity = round(tn/(tn+fp), 5)
                    fold_specificity.append(specificity)
                except ZeroDivisionError:
                    specificity = np.nan
                try:
                    f1 = round((2*tp)/(2*tp+fp+fn), 5)
                    fold_f1.append(f1)
                except ZeroDivisionError:
                    f1 = np.nan
                try:
                    mcc = round( ( ((tp*tn) - (fp*fn)) / np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) ), 5)
                    fold_mcc.append(mcc)
                except ZeroDivisionError:
                    mcc = np.nan
            
            print('Accuracy =', accuracy, '\nPrecision =', precision, '\nRecall =', recall, '\nSpecificity =', specificity, '\nF1 =', f1, '\nMCC =', mcc)
        np.seterr(invalid='ignore')
        # Evaluate k-fold performance and adjust for hypothetical imbalance
        precision, recall, thresholds = metrics.precision_recall_curve(df_pred[1], df_pred[0])
        if delta != 0.5:
            precision = recalculate_precision(df_pred, precision, thresholds, delta)
        fpr, tpr, __ = metrics.roc_curve(df_pred[1], df_pred[0])
        if delta == 0.5:
            pr_auc = metrics.average_precision_score(df_pred[1], df_pred[0])
        else:
            pr_auc = metrics.auc(recall, precision)
        roc_auc = metrics.roc_auc_score(df_pred[1], df_pred[0])
        
        print('AUC_ROC = %0.5f'%roc_auc, '\nAUC_PR = %0.5f'%pr_auc)
        
        # Add k-fold performance for overall average performance
        tprs[fold] = tpr
        fprs[fold] = fpr
        roc_aucs[fold] = roc_auc
        precisions[fold] = precision
        recalls[fold] = recall
        pr_aucs[fold] = pr_auc
    
        fold += 1
    
    # Get total performance from all PPI predictions (concatenated k-fold tested subsets)
    precision, recall, thresholds = metrics.precision_recall_curve(df_pred_total[1], df_pred_total[0])
    if delta != 0.5:
        precision = recalculate_precision(df_pred_total, precision, thresholds, delta)
    fpr, tpr, __ = metrics.roc_curve(df_pred_total[1], df_pred_total[0])
    if delta == 0.5:
        pr_auc = metrics.average_precision_score(df_pred_total[1], df_pred_total[0])
    else:
        pr_auc = metrics.auc(recall, precision)
    roc_auc = metrics.roc_auc_score(df_pred_total[1], df_pred_total[0])
    
    if df_pred_total[0].min() >= 0 and df_pred_total[0].max() <= 1:# and 'SPRINT' not in scores:
        evaluation = ('accuracy = %.5f (+/- %.5f)'%(np.mean(fold_accuracy), np.std(fold_accuracy))
                      + '\nprecision = %.5f (+/- %.5f)'%(np.mean(fold_precision), np.std(fold_precision)) 
                      + '\nrecall = %.5f (+/- %.5f)'%(np.mean(fold_recall), np.std(fold_recall)) 
                      + '\nspecificity = %.5f (+/- %.5f)'%(np.mean(fold_specificity), np.std(fold_specificity)) 
                      + '\nf1 = %.5f (+/- %.5f)'%(np.mean(fold_f1), np.std(fold_f1)) 
                      + '\nmcc = %.5f (+/- %.5f)'%(np.mean(fold_mcc), np.std(fold_mcc))
                      + '\nroc_auc = %.5f (+/- %.5f)' % (np.mean(np.fromiter(roc_aucs.values(), dtype=float)), np.std(np.fromiter(roc_aucs.values(), dtype=float)))
                      + '\npr_auc = %.5f (+/- %.5f)' % (np.mean(np.fromiter(pr_aucs.values(), dtype=float)), np.std(np.fromiter(pr_aucs.values(), dtype=float)))
                      + '\nroc_auc_overall = %.5f' % (roc_auc)
                      + '\npr_auc_overall = %.5f' % (pr_auc)
                      + '\n')
    else:
        evaluation = ('roc_auc = %.5f (+/- %.5f)' % (np.mean(np.fromiter(roc_aucs.values(), dtype=float)), np.std(np.fromiter(roc_aucs.values(), dtype=float)))
                      + '\npr_auc = %.5f (+/- %.5f)' % (np.mean(np.fromiter(pr_aucs.values(), dtype=float)), np.std(np.fromiter(pr_aucs.values(), dtype=float)))
                      + '\nroc_auc_overall = %.5f' % (roc_auc)
                      + '\npr_auc_overall = %.5f' % (pr_auc)
                      + '\n')  
    print('\n===== EVALUATION =====')
    print(evaluation)
    performance = pd.DataFrame(data={'precision': pd.Series(fold_precision, dtype=float), 'recall': pd.Series(fold_recall, dtype=float), 'specificity': pd.Series(fold_specificity, dtype=float), 'f1': pd.Series(fold_f1, dtype=float), 'mcc': pd.Series(fold_mcc, dtype=float), 'auc_pr': pd.Series(np.fromiter(pr_aucs.values(), dtype=float), dtype=float), 'auc_roc': pd.Series(np.fromiter(roc_aucs.values(), dtype=float), dtype=float)}, dtype=float)
    overall_curves = pd.DataFrame(data={'precision': pd.Series(precision, dtype=float), 'recall': pd.Series(recall, dtype=float), 'fpr': pd.Series(fpr, dtype=float), 'tpr': pd.Series(tpr, dtype=float)}, dtype=float)
    
    # Interpolate k-fold curves for overall std plotting
    interp_precisions = {}
    #pr_auc_interp = {}
    for i in recalls.keys():
        interp_precision = np.interp(recall, recalls[i], precisions[i], period=precision.shape[0]/precisions[i].shape[0])
        interp_precisions[i] = interp_precision
        #pr_auc_interp[i] = pr_aucs[i]
    df_interp_precisions = pd.DataFrame(data=interp_precisions)
    df_interp_precisions.insert(df_interp_precisions.shape[1], 'mean', df_interp_precisions.mean(axis=1))
    df_interp_precisions.insert(df_interp_precisions.shape[1], 'std', df_interp_precisions.std(axis=1))
    
    # Interpolate k-fold curves for overall std plotting
    interp_tprs = {}
    #roc_auc_interp = {}
    for i in fprs.keys():
        interp_tpr = np.interp(fpr, fprs[i], tprs[i], period=tpr.shape[0]/tprs[i].shape[0])
        interp_tprs[i] = interp_tpr
        #roc_auc_interp[i] = roc_aucs[i]
    df_interp_tprs = pd.DataFrame(data=interp_tprs)
    df_interp_tprs.insert(df_interp_tprs.shape[1], 'mean', df_interp_tprs.mean(axis=1))
    df_interp_tprs.insert(df_interp_tprs.shape[1], 'std', df_interp_tprs.std(axis=1))
    
    return performance, overall_curves, df_interp_precisions, df_interp_tprs, pr_auc, roc_auc

def test_anova(to_test):
    # Run ANOVA on all performance metric from all methods
    print('VALUES COMPARED:')
    df = to_test.copy()
    print(df)
    print('\nANOVA RESULTS:')
    if df.shape[1] > 1:
        df_melt = pd.melt(df.reset_index(), id_vars=['index'], value_vars=df.columns.tolist())
        anova = f_oneway(*[df[df.columns[i]] for i in range(df.shape[1])])
    else:
        print('Customize test manually')
        return
    df_melt.columns = ['index', 'method', 'value']

    # Ordinary Least Squares to estimate
    model = ols('value ~ method', data=df_melt).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)
    print('\nF-statistic: %s'%anova.statistic)
    print('p-value: %s\n'%anova.pvalue)
    
    a1 = 0.05
    a2 = 0.01
    '''
    If there are n total data points collected, then there are n−1 total degrees of freedom.
    If there are m groups being compared, then there are m−1 degrees of freedom associated with the factor of interest.
    If there are n total data points collected and m groups being compared, then there are n−m error degrees of freedom.
    '''
    # If there's a small likelihood that difference is by chance (null hypothesis that all relatively equal can be rejected)
    if anova.pvalue < a1:
        print("Significant difference for a=0.05 -> YES")
    else:
        print("Significant difference for a=0.05 -> NO")
    if anova.pvalue < a2:
        print("Significant difference for a=0.01 -> YES")
    else:
        print("Significant difference for a=0.01 -> NO")

def test_t(to_test, paired_or_independent='ind'):
    # Run two-tailed t-test between each combination of methods compared
    df = to_test.copy()
    combos = list(combinations(df.columns, 2))
    a1 = 0.05
    a2 = 0.01
    for i in range(len(combos)):
        print('\n\t', combos[i][0], 'vs.', combos[i][1])
        if paired_or_independent == 'ind':
            ttest = ttest_ind(df[combos[i][0]], df[combos[i][1]])
        else:
            ttest = ttest_rel(df[combos[i][0]], df[combos[i][1]])
        print('T-statistic:', ttest.statistic)
        print('p-value:', ttest.pvalue)
        if ttest.pvalue < a1:
            print("Significant difference for a=0.05 -> YES")
        else:
            print("Significant difference for a=0.05 -> NO")
        if ttest.pvalue < a2:
            print("Significant difference for a=0.01 -> YES")
        else:
            print("Significant difference for a=0.01 -> NO")

if __name__ == '__main__':
    
    log = open("%s%s.log"%(args.results, args.name), "a")
    sys.stdout = log
    t_start = time.time()
    scores_labels_mapping = {}
    for i in range(0, len(args.scores)):
        if len(args.labels) == 1:
            scores_labels_mapping[args.scores[i]] = args.labels[0]
        else:
            scores_labels_mapping[args.scores[i]] = args.labels[i]
    
    performances = {}
    overall_curves = {}
    interp_precisions = {}
    interp_tprs = {}
    pr_aucs = {}
    roc_aucs = {}
    names = []
    for s, l in scores_labels_mapping.items():
        performance, overall_curve, interp_precision, interp_tpr, pr_auc, roc_auc = get_metrics(s, l, args.delta)
        name = ''.join([ i.replace('/', '') for i in s.split('RESULTS')[1:] ])
        names.append(name)
        performances[name] = performance
        overall_curves[name] = overall_curve
        interp_precisions[name] = interp_precision
        interp_tprs[name] = interp_tpr
        pr_aucs[name] = pr_auc
        roc_aucs[name] = roc_auc
    
    to_test = pd.DataFrame()
    for n in names:
        if performances[n][args.metric].shape[0] > 1:
            to_test.insert(to_test.shape[1], n, performances[n][args.metric])
        
    print('========== COMPARING PERFORMANCES ==========')
    print('----- ANOVA -----\n')
    test_anova(to_test)
    print('\n----- t-Test -----')
    test_t(to_test, args.ttest_type)
    
    print('\n===== PLOTTING CURVES =====')
    # Display ratio of positives:negatives
    RATIO = '1:' + str(int((1/args.delta) - 1))

    # Precision-Recall
    plt.figure
    for n in names:
        plt.plot(overall_curves[n]['recall'], overall_curves[n]['precision'], label='%s AUC = %0.4f +/- %0.4f' % (n, pr_aucs[n], performances[n]['auc_pr'].std()))
        plt.fill_between(overall_curves[n]['recall'], overall_curves[n]['precision'] - interp_precisions[n]['std'], overall_curves[n]['precision'] + interp_precisions[n]['std'], alpha=0.15)
        #plt.fill_between(overall_curves[n]['recall'], overall_curves[n]['precision'] - 2*interp_precisions[n]['std'], overall_curves[n]['precision'] + 2*interp_precisions[n]['std'], alpha=0.1)
    plt.xlabel('Recall')
    plt.ylabel('Precision') 
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.title("Precision-Recall Curve - %s %s"%(args.name, RATIO))
    if args.delta == 0.5:
        plt.legend(loc='lower right', handlelength=1, prop={'size': 8})
    else:
        plt.legend(loc='upper right', handlelength=1, prop={'size': 8})
    plt.savefig(RESULTS_DIR + args.name + '_PR.png', format='png')
    plt.close()
    
    # ROC
    plt.figure
    for n in names:
        plt.plot(overall_curves[n]['fpr'], overall_curves[n]['tpr'], label='%s AUC = %0.4f +/- %0.4f' % (n, roc_aucs[n], performances[n]['auc_roc'].std()))
        plt.fill_between(overall_curves[n]['fpr'], overall_curves[n]['tpr'] - interp_tprs[n]['std'], overall_curves[n]['tpr'] + interp_tprs[n]['std'], alpha=0.15)
        #plt.fill_between(overall_curves[n]['fpr'], overall_curves[n]['tpr'] - 2*interp_tprs[n]['std'], overall_curves[n]['tpr'] + 2*interp_tprs[n]['std'], alpha=0.1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.title("ROC Curve - %s %s"%(args.name, RATIO))
    plt.legend(loc='lower right', handlelength=1, prop={'size': 8})
    plt.savefig(RESULTS_DIR + args.name + '_ROC.png', format='png')
    plt.close()
    
    print('Done\nTime = %.3f seconds'%(time.time() - t_start))