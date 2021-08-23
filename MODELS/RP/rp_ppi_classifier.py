#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description:
    Performs PPI prediction using RP datasets.
    RP datasets generated using rp_ppi.py based on PPI predictions from other models.
    
    Input arguements:
        -f: paths to files containing RP PPI datasets (.tsv)
        -k: number of k-folds to perform cross-validation of given files (int)
        -d: delta imbalance ratio of labelled RP data as positives/total (float)
        -c: perform CME (combines all dataset files) for PPI prediction (flag)
    
    Output files:
        Prediction probabilities for PPIs (.tsv)
        Performance results of PPI classification:
            - ROC curve
            - Precision-Recall curve
            - .txt of performance metrics
    
@author: Eric Arezza
Last Updated: May 11, 2021
"""

import os, argparse, time
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

'''
PIPR = '/home/erixazerro/CUBIC/PPIP/PREPROCESS/ECOLI/RP_biogrid_Ecoli_ID_511145_PIPR_interactions.tsv'
DEEPFE = '/home/erixazerro/CUBIC/PPIP/PREPROCESS/ECOLI/RP_biogrid_Ecoli_ID_511145_DEEPFE_interactions.tsv'
DPPI = '/home/erixazerro/CUBIC/PPIP/PREPROCESS/ECOLI/RP_biogrid_Ecoli_ID_511145_DPPI_interactions.tsv'
SPRINT = '/home/erixazerro/CUBIC/PPIP/PREPROCESS/ECOLI/RP_biogrid_Ecoli_ID_511145_SPRINT_interactions.tsv'
FILES = [PIPR, DEEPFE, DPPI, SPRINT]
'''

describe_help = 'python rp_ppi_classifier.py -f predictions.tsv -k 10'
parser = argparse.ArgumentParser(description=describe_help)
parser.add_argument('-f', '--files', help='Filepath(s) of dataset(s) (.tsv file)', type=str, nargs='+')
parser.add_argument('-train', '--train', help='Filepath(s) of training dataset(s) (.tsv file) to train model', type=str, nargs='+')
parser.add_argument('-test', '--test', help='Filepath(s) of testing dataset(s) (.tsv file) to make predictions', type=str, nargs='+')
parser.add_argument('-k', '--k_folds', help='Number of k-folds when cross-validating (int)', type=int, nargs=1, required=False)
parser.add_argument('-d', '--delta', help='Imbalance ratio as positives/total (e.g. balanced = 0.5)', type=float, nargs=1, required=False)
parser.add_argument('-c', '--cme', help='Perform a combination of multiple experts (combine datasets provided in -files)', action='store_true', default=False)
args = parser.parse_args()

FILES = args.files
if args.k_folds is None:
    K_FOLDS = 10
else:
    K_FOLDS = args.k_folds[0]
if args.delta is None:
    IMBALANCE = 0.5
else:
    IMBALANCE = args.delta[0]
RATIO = '1:' + str(int((1/IMBALANCE) - 1))

def recalculate_precision(df, precision, thresholds, ratio=IMBALANCE):
    delta = 2*ratio - 1
    new_precision = precision.copy()
    for t in range(0, len(thresholds)):
        tn, fp, fn, tp = metrics.confusion_matrix(df['Truth'], (df['Score'] >= thresholds[t]).astype(int)).ravel()
        tp = tp*(1 + delta)
        fp = fp*(1 - delta)
        new_precision[t] = tp/(tp+fp)
    return new_precision
'''
def plot_curves(filename, df):
    
    # Plot precision-recall curve
    precision, recall, thresholds = metrics.precision_recall_curve(df['Truth'], df['Score'])
    precision = recalculate_precision(df, precision, thresholds)
    if IMBALANCE == 0.5:
        auc_pr = metrics.average_precision_score(df['Truth'], df['Score'])
    else:
        auc_pr = metrics.auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, label='AUC='+str(round(auc_pr, 4)))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.title('Precision-Recall Curve - ' + RATIO)
    plt.legend(handlelength=0)
    plt.savefig(os.getcwd() + '/RESULTS/performance_' + filename + '_PR.png', format='png')
    
    # Plot ROC curve
    fpr, tpr, __ = metrics.roc_curve(df['Truth'], df['Score'])
    if IMBALANCE == 0.5:
        auc_roc = metrics.roc_auc_score(df['Truth'], df['Score'])
    else:
        auc_roc = metrics.auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label='AUC='+str(round(auc_roc, 4)))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.title('ROC Curve')
    plt.legend(handlelength=0)
    plt.savefig(os.getcwd() + '/RESULTS/performance_' + filename + '_ROC.png', format='png')
'''

if __name__ == '__main__':
    
    if len(FILES) == 0:
        print('No data provided')
        exit()
    if not os.path.exists(os.getcwd()+'/RESULTS/'):
        os.mkdir(os.getcwd()+'/RESULTS/')
    
    # Combine feature vectors for PPIs from all files
    if args.cme:
        print('Performing CME...')
        # Load first file
        df_cme = pd.read_csv(FILES[0], delim_whitespace=True)
        cme_name = FILES[0].split('/')[-1].split('_')
        # Append remaining files
        for f in range(1, len(FILES)):
            # Read files
            df = pd.read_csv(FILES[f], delim_whitespace=True)
            df_cme = df_cme.merge(df, on=[df.columns[0], df.columns[1]])
            df_cme.drop(columns=['label_x'], inplace=True)
            df_cme.rename(columns={'label_y': 'label'}, inplace=True)
            [cme_name.append(s) for s in FILES[f].split('/')[-1].split('_') if s not in cme_name]
        cme_name.sort()
        cme_name = '_'.join(i for i in cme_name)
        FILES = [cme_name]
        
    for f in FILES:
        
        # Run CME using previously created df from files, else use df from files individually
        if args.cme:
            df = df_cme.copy()
        else:       
            df = pd.read_csv(f, delim_whitespace=True)
        
        # Define data and labels
        pairs = np.array(df[df.columns[0:2]])
        X = np.array(df[df.columns[2:-1]])
        '''
        X = np.array(df[['FD_A_elbow', 'FD_B_elbow', 'FD_A_knee', 'FD_B_knee', 'Above_Global_Mean', 'Above_Global_Median']])
        X = np.array(df[['FD_A_elbow_y', 'FD_B_elbow_y', 'FD_A_knee_y', 'FD_B_knee_y', 'Above_Global_Mean_y', 'Above_Global_Median_y']])
        '''
        y = np.array(df[df.columns[-1]])
        
        
        # Define data partiioning
        kf = StratifiedKFold(n_splits=K_FOLDS)
        
        # Define classifier model and pipeline
        clf = RandomForestClassifier(random_state=13052021)
        pipe = Pipeline([('scaler', StandardScaler()), ('rndforest', clf)])
        
        # Metrics for evaluation
        avg_accuracy = []
        avg_precision = []
        avg_recall = []
        avg_specificity = []
        avg_f1 = []
        avg_mcc = []
        avg_roc_auc = []
        avg_pr_auc = []
        
        # For ROC curve
        tprs = {}
        roc_aucs = {}
        fprs = {}
        # For PR curve
        precisions = {}
        pr_aucs = {}
        recalls = {}
        
        # Perform k-fold predictions and evaluation
        k = 0
        prob_predictions = pd.DataFrame()
        bin_predictions = pd.DataFrame()
        t_start = time.time()
        for train, test in kf.split(X, y):
            
            print('===== Fold-%s ====='%k)
            
            # Fit model with training data
            pipe.fit(X[train], y[train])
            
            # Record PPI binary predictions
            pred = pipe.predict(X[test])
            ppi_bin = pd.DataFrame(pairs[test], columns=[df.columns[0], df.columns[1]])
            ppi_bin.insert(2, 2, pred)
            bin_predictions = bin_predictions.append(ppi_bin)
            
            # Record PPI prediction probabilities
            pred_probs = pipe.predict_proba(X[test])
            ppi_probs = pd.DataFrame(np.append(pairs[test], pred_probs, axis=1), columns=[df.columns[0], df.columns[1], 2, 3])
            ppi_probs.drop(columns=[2], inplace=True)
            prob_predictions = prob_predictions.append(ppi_probs)
            
            # Get performance metricss of binary evaluation (threshold=0.5)
            tn, fp, fn, tp = metrics.confusion_matrix(y[test], pred).ravel()
            print('tp='+str(tp), 'fp='+str(fp), 'tn='+str(tn), 'fn='+str(fn))
            
            accuracy = (tp + tn) / (tn + fp + fn + tp)
            prec = tp / (tp + fp + 1e-06)
            recall = tp / (tp + fn + 1e-06)
            spec = tn / (tn + fp + 1e-06)
            f1 = 2. * (prec * recall) / (prec + recall + 1e-06)
            mcc = (tp * tn - fp * fn) / (((tp + fp + 1e-06) * (tp + fn + 1e-06) * (fp + tn + 1e-06) * (tn + fn + 1e-06)) ** 0.5)
            
            auc_roc = metrics.roc_auc_score(y[test], pred_probs[:, 1])
            auc_pr = metrics.average_precision_score(y[test], pred_probs[:, 1])
            fpr, tpr, __ = metrics.roc_curve(y[test], pred_probs[:, 1])
            
            print('acc=' + str(round(accuracy, 4)), 'prec=' + str(round(prec, 4)), 'recall=' + str(round(recall, 4)), \
                  'spec=' + str(round(spec, 4)), 'f1=' + str(round(f1, 4)), 'mcc=' + str(round(mcc, 4)))
            
            avg_accuracy.append(accuracy)
            avg_precision.append(prec)
            avg_recall.append(recall)
            avg_specificity.append(spec)
            avg_f1.append(f1)
            avg_mcc.append(mcc)
            avg_roc_auc.append(auc_roc)
            avg_pr_auc.append(auc_pr)
            
            
            # Get performance metricss for curve plotting
            # Evaluate k-fold performance and adjust for hypothetical imbalance
            precision, recall, thresholds = metrics.precision_recall_curve(y[test], pred_probs[:, 1])
            truth_score = pd.DataFrame(data={'Truth':y[test], 'Score':pred_probs[:, 1]})
            precision = recalculate_precision(truth_score, precision, thresholds, ratio=IMBALANCE)
            fpr, tpr, __ = metrics.roc_curve(y[test], pred_probs[:, 1])
            if IMBALANCE == 0.5:
                pr_auc = metrics.average_precision_score(y[test], pred_probs[:, 1])
            else:
                pr_auc = metrics.auc(recall, precision)
            roc_auc = metrics.roc_auc_score(y[test], pred_probs[:, 1])
            print('auc_roc=', roc_auc, '\nauc_pr=', pr_auc)
            # Add k-fold performance for overall average performance
            tprs[k] = tpr
            fprs[k] = fpr
            roc_aucs[k] = roc_auc
            precisions[k] = precision
            recalls[k] = recall
            pr_aucs[k] = pr_auc
            
            k += 1
        
        duration = time.time() - t_start
        
        # Compile all predictions from k test folds
        prob_predictions = df.merge(prob_predictions, on=[df.columns[0], df.columns[1]])
        prob_predictions.drop(columns=df.columns[2:], inplace=True)
        prob_predictions.reset_index(drop=True, inplace=True)
        bin_predictions = df.merge(bin_predictions, on=[df.columns[0], df.columns[1]])
        bin_predictions.drop(columns=df.columns[2:], inplace=True)
        bin_predictions.reset_index(drop=True, inplace=True)
    
        # Write PPI predictions to file
        pred_filename = result_filename = os.getcwd() + '/RESULTS/predictions_' + f.split('/')[-1]
        prob_predictions.to_csv(pred_filename, sep='\t', header=None, index=False)
        
        results ='accuracy=%.4f (+/- %.4f)'%(np.mean(avg_accuracy), np.std(avg_accuracy)) \
                      + '\nprecision=%.4f (+/- %.4f)'%(np.mean(avg_precision), np.std(avg_precision)) \
                      + '\nrecall=%.4f (+/- %.4f)'%(np.mean(avg_recall), np.std(avg_recall)) \
                      + '\nspecificity=%.4f (+/- %.4f)'%(np.mean(avg_specificity), np.std(avg_specificity)) \
                      + '\nf1=%.4f (+/- %.4f)'%(np.mean(avg_f1), np.std(avg_f1)) \
                      + '\nmcc=%.4f (+/- %.4f)'%(np.mean(avg_mcc), np.std(avg_mcc)) \
                      + '\nroc_auc=%.4f (+/- %.4f)' % (np.mean(avg_roc_auc), np.std(avg_roc_auc)) \
                      + '\npr_auc=%.4f (+/- %.4f)' % (np.mean(avg_pr_auc), np.std(avg_pr_auc)) \
                      + '\ntime(sec.)=%.2f'%(duration) \
                      + '\n'
        print('----- Results -----\n' + results)
        # Write results to file
        result_filename = os.getcwd() + '/RESULTS/results_' + f.split('/')[-1].replace('.tsv', '.txt')
        with open(result_filename, 'w') as fp:
            fp.write(results)
        
        truth_score = pd.DataFrame(data={'Truth':y, 'Score':prob_predictions[prob_predictions.columns[-1]]})
        
        # Plot performance curves
        #plot_curves(f.split('/')[-1].replace('.tsv', ''), truth_score)
        
        # Get overall performance across all folds
        precision, recall, thresholds = metrics.precision_recall_curve(truth_score['Truth'], truth_score['Score'])
        precision = recalculate_precision(truth_score, precision, thresholds, ratio=IMBALANCE)
        fpr, tpr, __ = metrics.roc_curve(truth_score['Truth'], truth_score['Score'])
        if IMBALANCE == 0.5:
            pr_auc = metrics.average_precision_score(truth_score['Truth'], truth_score['Score'])
        else:
            pr_auc = metrics.auc(recall, precision)
        roc_auc = metrics.roc_auc_score(truth_score['Truth'], truth_score['Score'])
        
        # Plot and save curves
        plt.figure
        plt.plot(recall, precision, color='black', label='AUC = %0.4f +/- %0.4f' % (pr_auc, np.std(np.fromiter(avg_pr_auc, dtype=float))))
        for i in range(0, K_FOLDS):
            plt.plot(recalls[i], precisions[i], alpha=0.25)
        plt.xlabel('Recall')
        plt.ylabel('Precision') 
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.title("Precision-Recall Curve - %s %s"%(f.split('/')[-1].replace('.tsv', ''), RATIO))
        plt.legend(loc='best', handlelength=0)
        plt.savefig(os.getcwd() + '/RESULTS/performance_' + f.split('/')[-1].replace('.tsv', '_PR.png'), format='png')
        plt.close()
        
        plt.figure
        plt.plot(fpr, tpr, color='black', label='AUC = %0.4f +/- %0.4f' % (roc_auc, np.std(np.fromiter(avg_roc_auc, dtype=float))))
        for i in range(0, K_FOLDS):
            plt.plot(fprs[i], tprs[i], alpha=0.25)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.title("ROC Curve - %s %s"%(f.split('/')[-1].replace('.tsv', ''), RATIO))
        plt.legend(loc='lower right', handlelength=0)
        plt.savefig(os.getcwd() + '/RESULTS/performance_' + f.split('/')[-1].replace('.tsv', '_ROC.png'), format='png')
        plt.close()
        
