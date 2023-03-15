#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description:
    Evaluates the performance of PPI predictions by plotting ROC and Precision-Recall curves.
    Plots prevalence-corrected curves for hypothetically imbalanced data.
    Writes to file with evaluated metric results.
    
    Requires files to have no header and be whitespace-separated (.tsv).
    
Usage:
    python evaluate_ppi.py -s SCORES/ -l labels.tsv -d 0.5 -r RESULTS/
    
    Input arguements:
        -s <str> Can be either:
            - a directory path where tested PPI prediction k-fold subset files exist (file names must contain the word 'prediction' and a number)
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
           'recalculate_metrics_to_imbalance',
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
                    type=float, default=0.5)
parser.add_argument('-n', '--name', help='Name for saving files, default basename will be results directory name', 
                    type=str, default='')
args = parser.parse_args()

RESULTS_DIR = args.results
if not os.path.exists(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)
if args.name == '' and len(args.labels) == 1:
    args.name = RESULTS_DIR.split('/')[-2].lower().capitalize() + '_' + args.labels[0].split('/')[-1].split('.')[0]
elif args.name == '' and len(args.labels) > 1:
    args.name = RESULTS_DIR.split('/')[-2].lower().capitalize()

# Display ratio of positives:negatives
RATIO = '1:' + str(int((1/args.delta) - 1))

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
    

if __name__ == '__main__':

    # Get PPI scores
    if os.path.isdir(args.scores):
        # For cross-validation tested PPI subsets
        files = os.listdir(path=args.scores)
        files = [ x for x in files if 'prediction' in x and '.pos' not in x and '.neg' not in x ]
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

    # Additional metrics from each k-fold subset
    fold_accuracy = []
    fold_precision = []
    fold_recall = []
    fold_specificity = []
    fold_f1 = []
    fold_mcc = []

    df_pred_total = pd.DataFrame()
    fold = 0
    output = args.name
    for k in files:
        
        # Isolate k-fold subset
        print('\n===== Fold - %s ====='%fold)
        output += '\n===== Fold - %s ====='%fold
        
        # Read predictions for k-fold set or single test set
        if os.path.isdir(args.scores):
            df_pred = pd.read_csv(args.scores + k, delim_whitespace=True, header=None)
        else:
            df_pred = pd.read_csv(args.scores, delim_whitespace=True, header=None)
        
        # Get matching PPI labels for predictions
        #if '_SPRINT_' not in k and ('SPRINT' not in args.scores and 'CME' not in args.scores):
        df_pred = get_matching_pairs(df_pred, df_labels)
        df_pred.drop(columns=[0, 1], inplace=True)
        df_pred.rename(columns={'2_x': 0, '2_y': 1}, inplace=True)
        #df_pred[[0, 1]] = df_pred[[1, 0]]
        
        df_pred_total = df_pred_total.append(df_pred)
        
        # Get other metrics at 0.5 threshold if predictions are probabilities (0 to 1) i.e. not SPRINT predictions
        if df_pred[0].min() >= 0 and df_pred[0].max() <= 1: #and 'SPRINT' not in args.scores:
            
            tn, fp, fn, tp = metrics.confusion_matrix(df_pred[1], (df_pred[0] + 1e-12).round()).ravel()
            print('TP = %0.0f \nFP = %0.0f \nTN = %0.0f \nFN = %0.0f'%(tp, fp, tn, fn))
            output += '\nTP = %0.0f \nFP = %0.0f \nTN = %0.0f \nFN = %0.0f'%(tp, fp, tn, fn)
            print('Total samples = %s'%(tn+fp+fn+tp))
            output += 'Total_samples = %s'%(tn+fp+fn+tp)
            
            # For imbalanced classification metrics
            if args.delta != 0.5:
                accuracy, precision, recall, specificity, f1, mcc = recalculate_metrics_to_imbalance(tp, tn, fp, fn, args.delta)
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
            output += '\nAccuracy = ' + str(accuracy) + '\nPrecision = ' + str(precision) + '\nRecall = '+ str(recall) + '\nSpecificity = ' + str(specificity) + '\nF1 = ' + str(f1) + '\nMCC = ' + str(mcc)
        np.seterr(invalid='ignore')
        # Evaluate k-fold performance and adjust for hypothetical imbalance
        precision, recall, thresholds = metrics.precision_recall_curve(df_pred[1], df_pred[0])
        if args.delta != 0.5:
            precision = recalculate_precision(df_pred, precision, thresholds, args.delta)
        fpr, tpr, __ = metrics.roc_curve(df_pred[1], df_pred[0])
        if args.delta == 0.5:
            pr_auc = metrics.average_precision_score(df_pred[1], df_pred[0])
        else:
            pr_auc = metrics.auc(recall, precision)
        roc_auc = metrics.roc_auc_score(df_pred[1], df_pred[0])
        
        print('AUC_ROC = %0.5f'%roc_auc, '\nAUC_PR = %0.5f'%pr_auc)
        output += '\nAUC_ROC = %0.5f'%roc_auc + '\nAUC_PR = %0.5f\n'%pr_auc
        
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
    if args.delta != 0.5:
        precision = recalculate_precision(df_pred_total, precision, thresholds, args.delta)
    fpr, tpr, __ = metrics.roc_curve(df_pred_total[1], df_pred_total[0])
    if args.delta == 0.5:
        pr_auc = metrics.average_precision_score(df_pred_total[1], df_pred_total[0])
    else:
        pr_auc = metrics.auc(recall, precision)
    roc_auc = metrics.roc_auc_score(df_pred_total[1], df_pred_total[0])
    
    if args.delta <= 0.5:
        leg_loc = 'lower right'
    else:
        leg_loc = 'upper right'
    
    # Get other metrics at 0.5 threshold if predictions are probabilities (0 to 1) i.e. not SPRINT predictions
    if df_pred_total[0].min() >= 0 and df_pred_total[0].max() <= 1:# and 'SPRINT' not in args.scores:
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
    output += '\n===== EVALUATION =====\n' + evaluation
    print('Writing output to file...')
    with open(RESULTS_DIR + 'evaluation_' + args.name + '.txt', 'w') as fp:
        fp.write(output)
    
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
    
    # Plot and save curves
    print("Plotting precision-recall")
    # Precision-Recall
    plt.figure
    print("\t...average curve...")
    plt.plot(recall, precision, color='black', label='AUC = %0.4f +/- %0.4f' % (pr_auc, np.std(np.fromiter(pr_aucs.values(), dtype=float))))
    '''
    for i in recalls.keys():
        print("\t...fold-%s curve..."%i)
        plt.plot(recalls[i], precisions[i], alpha=0.25)
    '''
    plt.fill_between(recall, precision - df_interp_precisions['std'], precision + df_interp_precisions['std'], facecolor='blue', alpha=0.25)
    plt.fill_between(recall, precision - 2*df_interp_precisions['std'], precision + 2*df_interp_precisions['std'], facecolor='blue', alpha=0.25)
    #plt.plot(recall, df_interp_precisions['mean'], color='orange', label='AUC_folds = %0.4f +/- %0.4f' % (np.mean(np.fromiter(pr_auc_interp.values(), dtype=float)), np.std(np.fromiter(pr_auc_interp.values(), dtype=float))))
    plt.xlabel('Recall')
    plt.ylabel('Precision') 
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.title("Precision-Recall Curve - %s %s"%(args.name, RATIO))
    plt.legend(loc=leg_loc, handlelength=0, prop={'size': 8})
    plt.savefig(RESULTS_DIR + args.name + '_PR.png', format='png')
    plt.close()
    
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
    
    # ROC
    print("Plotting ROC")
    plt.figure
    print("\t...average curve...")
    plt.plot(fpr, tpr, color='black', label='AUC = %0.4f +/- %0.4f' % (roc_auc, np.std(np.fromiter(roc_aucs.values(), dtype=float))))
    '''
    for i in fprs.keys():
        print("\t...fold-%s curve..."%i)
        plt.plot(fprs[i], tprs[i], alpha=0.25)
    '''
    plt.fill_between(fpr, tpr - df_interp_tprs['std'], tpr + df_interp_tprs['std'], facecolor='blue', alpha=0.25)
    plt.fill_between(fpr, tpr - 2*df_interp_tprs['std'], tpr + 2*df_interp_tprs['std'], facecolor='blue', alpha=0.25)
    #plt.plot(fpr, df_interp_tprs['mean'], color='orange', label='AUC_folds = %0.4f +/- %0.4f' % (np.mean(np.fromiter(roc_auc_interp.values(), dtype=float)), np.std(np.fromiter(roc_auc_interp.values(), dtype=float))))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.title("ROC Curve - %s %s"%(args.name, RATIO))
    plt.legend(loc=leg_loc, handlelength=0, prop={'size': 8})
    plt.savefig(RESULTS_DIR + args.name + '_ROC.png', format='png')
    plt.close()
