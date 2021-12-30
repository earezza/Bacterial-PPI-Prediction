#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description:
    Performs PPI prediction using RP datasets.
    RP datasets generated using rp_ppi.py based on PPI predictions from other models.
    
    Input arguements:
        -f: paths to files containing RP PPI datasets (.tsv) (cross-validation will be performed, otherwise input -train and -test)
        -train: Filepath(s) of training dataset(s) (.tsv file)
        -test: Filepath(s) of testing dataset(s) (.tsv file)
        -k: number of k-folds to perform cross-validation of given files (int)
        -d: delta imbalance ratio of labelled RP data as positives/total (float)
        -c: perform CME (combines all dataset files provided by -f) for PPI prediction (flag)
    
    Output files:
        Prediction probabilities for PPIs (.tsv)
        Performance results of PPI classification:
            - ROC curve
            - Precision-Recall curve
            - .txt of performance metrics
    
@author: Eric Arezza
Last Updated: September 18, 2021
"""

import os, argparse, time
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from lightgbm.sklearn import LGBMClassifier
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

describe_help = 'python rp_ppi_classifier.py -f predictions1.tsv predictions2.tsv predictions3.tsv -d 0.5 -c -k 10' + '\nOR\n' \
    + 'python rp_ppi_classifier.py -train trainData.tsv -test testData.tsv -d 0.5'
parser = argparse.ArgumentParser(description=describe_help)
parser.add_argument('-f', '--files', help='Filepath(s) of dataset(s) (.tsv file) if cross-validation', type=str, nargs='+')
parser.add_argument('-train', '--train', help='Filepath(s) of training dataset(s) (.tsv file) to train model', type=str)
parser.add_argument('-test', '--test', help='Filepath(s) of testing dataset(s) (.tsv file) to make predictions', type=str)
parser.add_argument('-k', '--k_folds', help='Number of k-folds when cross-validating (int)', type=int, nargs=1, required=False, default=10)
parser.add_argument('-d', '--delta', help='Imbalance ratio as positives/total (e.g. balanced = 0.5)', type=float, nargs=1, required=False, default=0.5)
parser.add_argument('-c', '--cme', help='Perform a combination of multiple experts (combine datasets provided in -files)', action='store_true', default=False)
parser.add_argument('-r', '--results', help='Path to directory for saving prediction results', type=str, default=os.getcwd()+'/RESULTS/')
parser.add_argument('-n', '--name', help='Name for saving files (optional, will default to modified filenames)', type=str, default='')
args = parser.parse_args()

FILES = args.files
K_FOLDS = args.k_folds
IMBALANCE = args.delta
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

if __name__ == '__main__':
    
    if not os.path.exists(args.results):
        os.mkdir(args.results)
    
    # ================== FOR SINGLE TRAIN/TEST RUNS ==================
    if FILES == None and args.train != None and args.test != None:
        t_start = time.time()
        print('Training on %s\nTesting on %s\n'%(args.train.split('/')[-1], args.test.split('/')[-1]))
        
        # Load data
        df_train = pd.read_csv(args.train, delim_whitespace=True)
        df_test = pd.read_csv(args.test, delim_whitespace=True)
        
        # Define data and labels
        pairs_train = np.array(df_train[df_train.columns[0:2]])
        X_train = np.array(df_train[df_train.columns[2:-1]])
        y_train = np.array(df_train[df_train.columns[-1]])
        pairs_test = np.array(df_test[df_test.columns[0:2]])
        X_test = np.array(df_test[df_test.columns[2:-1]])
        y_test = np.array(df_test[df_test.columns[-1]])
        
        # Filename for saving predictions
        if args.name == '':
            save_name = args.test.split('.')[0].split('/')[-1]
        else:
            save_name = args.name
            
        # Define classifier model and pipeline
        #clf = RandomForestClassifier(random_state=13052021)
        clf = LGBMClassifier(random_state=13052021, 
                             boosting_type='goss', 
                             num_leaves=40, 
                             learning_rate=0.15)
        '''
        clf = xgb.XGBClassifier(random_state=13052021, 
                                use_label_encoder=False, 
                                eval_metric='mlogloss')
        '''
        #pipe = Pipeline([('scaler', StandardScaler()), ('rndforest', clf)])
        pipe = Pipeline([('scaler', StandardScaler()), ('lgbm', clf)])
        
        # Fit model with training data
        pipe.fit(X_train, y_train)
        
        # Record PPI binary predictions
        pred = pipe.predict(X_test)
        ppi_bin = pd.DataFrame(pairs_test, columns=[df_test.columns[0], df_test.columns[1]])
        ppi_bin.insert(2, 2, pred)
        
        # Record PPI prediction probabilities
        pred_probs = pipe.predict_proba(X_test)
        ppi_probs = pd.DataFrame(np.append(pairs_test, pred_probs, axis=1), columns=[df_test.columns[0], df_test.columns[1], 2, 3])
        ppi_probs.drop(columns=[2], inplace=True)
        ppi_probs.to_csv(args.results + 'predictions_' + save_name + '.tsv', sep='\t', header=None, index=False)
        
        # Get performance metricss of binary evaluation (threshold=0.5)
        tn, fp, fn, tp = metrics.confusion_matrix(y_test, pred).ravel()
        print('tp='+str(tp), 'fp='+str(fp), 'tn='+str(tn), 'fn='+str(fn))
        
        accuracy = (tp + tn) / (tn + fp + fn + tp)
        prec = tp / (tp + fp + 1e-06)
        rec = tp / (tp + fn + 1e-06)
        spec = tn / (tn + fp + 1e-06)
        f1 = 2. * (prec * rec) / (prec + rec + 1e-06)
        mcc = (tp * tn - fp * fn) / (((tp + fp + 1e-06) * (tp + fn + 1e-06) * (fp + tn + 1e-06) * (tn + fn + 1e-06)) ** 0.5)
        
        auc_roc = metrics.roc_auc_score(y_test, pred_probs[:, 1])
        auc_pr = metrics.average_precision_score(y_test, pred_probs[:, 1])
        fpr, tpr, __ = metrics.roc_curve(y_test, pred_probs[:, 1])
        
        print('acc=' + str(round(accuracy, 4)), 'prec=' + str(round(prec, 4)), 'recall=' + str(round(rec, 4)), \
              'spec=' + str(round(spec, 4)), 'f1=' + str(round(f1, 4)), 'mcc=' + str(round(mcc, 4)))
                
        # Get performance metricss for curve plotting
        # Evaluate performance and adjust for hypothetical imbalance
        precision, recall, thresholds = metrics.precision_recall_curve(y_test, pred_probs[:, 1])
        truth_score = pd.DataFrame(data={'Truth':y_test, 'Score':pred_probs[:, 1]})
        precision = recalculate_precision(truth_score, precision, thresholds, ratio=IMBALANCE)
        fpr, tpr, __ = metrics.roc_curve(y_test, pred_probs[:, 1])
        if IMBALANCE == 0.5:
            pr_auc = metrics.average_precision_score(y_test, pred_probs[:, 1])
        else:
            pr_auc = metrics.auc(recall, precision)
        roc_auc = metrics.roc_auc_score(y_test, pred_probs[:, 1])
        print('auc_roc=%.6f'%(roc_auc) + '\nauc_pr=%.6f'%(pr_auc))
        
        duration = time.time() - t_start
        
        results ='accuracy=%.4f'%(accuracy) + '\nprecision=%.4f'%(prec) + '\nrecall=%.4f'%(rec) + '\nspecificity=%.4f'%(spec) \
                + '\nf1=%.4f'%(f1) + '\nmcc=%.4f'%(mcc) + '\nroc_auc=%.4f'%(roc_auc) + '\npr_auc=%.4f'%(pr_auc) \
                + '\ntime(sec.)=%.2f'%(duration) + '\n'
        print('----- Results -----\n' + results)
        # Write results to file
        result_filename = args.results + 'results_' + save_name + '.txt'
        with open(result_filename, 'w') as fp:
            fp.write(results)
        
        # Plot and save curves
        plt.figure
        plt.plot(recall, precision, color='black', label='AUC = %0.4f' % (pr_auc))
        plt.xlabel('Recall')
        plt.ylabel('Precision') 
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.title("Precision-Recall Curve - %s %s"%(save_name, RATIO))
        plt.legend(loc='best', handlelength=0)
        if args.delta == 0.5:
            plt.savefig(args.results + 'performance_' + save_name + '_PR.png', format='png')
        else:
            plt.savefig(args.results + 'performance_' + save_name + 'Imbalanced_PR.png', format='png')
        plt.close()
        
        plt.figure
        plt.plot(fpr, tpr, color='black', label='AUC = %0.4f' % (roc_auc))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.title("ROC Curve - %s %s"%(save_name, RATIO))
        plt.legend(loc='lower right', handlelength=0)
        if args.delta == 0.5:
            plt.savefig(args.results + 'performance_' + save_name + '_ROC.png', format='png')
        else:
            plt.savefig(args.results + 'performance_' + save_name + '_Imbalanced_ROC.png', format='png')
        plt.close()
        
        exit()
        
    # ================== FOR CROSS-VALIDATION RUNS ==================
    # Combine feature vectors for PPIs from all files
    if args.cme:
        print('Performing CME...')
        # Load first file
        df_cme = pd.read_csv(FILES[0], delim_whitespace=True)
        cme_name = FILES[0].split('.')[0].split('/')[-1].split('_')
        # Append remaining files
        for f in range(1, len(FILES)):
            # Read files
            df = pd.read_csv(FILES[f], delim_whitespace=True)
            df_cme = df_cme.merge(df, on=[df.columns[0], df.columns[1]])
            df_cme.drop(columns=['label_x'], inplace=True)
            df_cme.rename(columns={'label_y': 'label'}, inplace=True)
            [cme_name.append(s) for s in FILES[f].split('.')[0].split('/')[-1].split('_') if s not in cme_name]
        #cme_name.sort()
        cme_name = '_'.join(i for i in cme_name)
        FILES = [cme_name]
        if args.name == '':
            save_name = cme_name
        else:
            save_name = args.name
        
    for f in FILES:
        print('Performing cross-validation run on %s'%([i.split('/')[-1] for i in FILES]))
        # Run CME using previously created df from files, else use df from files individually
        if args.cme:
            df = df_cme.copy()
        else:       
            df = pd.read_csv(f, delim_whitespace=True)
            if args.name == '':
                save_name = f.split('.')[0].split('/')[-1]
            else:
                save_name = args.name
        
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
        #clf = RandomForestClassifier(random_state=13052021)
        clf = xgb.XGBClassifier(random_state=13052021, 
                                use_label_encoder=False, 
                                eval_metric='mlogloss')
        #pipe = Pipeline([('scaler', StandardScaler()), ('rndforest', clf)])
        pipe = Pipeline([('scaler', StandardScaler()), ('boost', clf)])
        
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
            ppi_probs.to_csv(args.results + 'predictions_' + save_name + '_fold-%s'%k + '.tsv', sep='\t', header=None, index=False)
            prob_predictions = prob_predictions.append(ppi_probs)
            
            # Get performance metricss of binary evaluation (threshold=0.5)
            tn, fp, fn, tp = metrics.confusion_matrix(y[test], pred).ravel()
            print('tp='+str(tp), 'fp='+str(fp), 'tn='+str(tn), 'fn='+str(fn))
            
            accuracy = (tp + tn) / (tn + fp + fn + tp)
            prec = tp / (tp + fp + 1e-06)
            rec = tp / (tp + fn + 1e-06)
            spec = tn / (tn + fp + 1e-06)
            f1 = 2. * (prec * rec) / (prec + rec + 1e-06)
            mcc = (tp * tn - fp * fn) / (((tp + fp + 1e-06) * (tp + fn + 1e-06) * (fp + tn + 1e-06) * (tn + fn + 1e-06)) ** 0.5)
            
            auc_roc = metrics.roc_auc_score(y[test], pred_probs[:, 1])
            auc_pr = metrics.average_precision_score(y[test], pred_probs[:, 1])
            fpr, tpr, __ = metrics.roc_curve(y[test], pred_probs[:, 1])
            
            print('acc=' + str(round(accuracy, 4)), 'prec=' + str(round(prec, 4)), 'recall=' + str(round(rec, 4)), \
                  'spec=' + str(round(spec, 4)), 'f1=' + str(round(f1, 4)), 'mcc=' + str(round(mcc, 4)))
            
            avg_accuracy.append(accuracy)
            avg_precision.append(prec)
            avg_recall.append(rec)
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
            print('auc_roc=%.6f'%(roc_auc) + '\nauc_pr=%.6f'%(pr_auc))
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
        pred_filename = result_filename = args.results + 'predictions_' + save_name + '.tsv'
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
        result_filename = args.results + 'results_' + save_name + '.txt'
        with open(result_filename, 'w') as fp:
            fp.write(results)
        
        truth_score = pd.DataFrame(data={'Truth':y, 'Score':prob_predictions[prob_predictions.columns[-1]]})
        
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
        plt.title("Precision-Recall Curve - %s %s"%(save_name, RATIO))
        plt.legend(loc='best', handlelength=0)
        plt.savefig(args.results + 'performance_' + save_name + '_PR.png', format='png')
        plt.close()
        
        plt.figure
        plt.plot(fpr, tpr, color='black', label='AUC = %0.4f +/- %0.4f' % (roc_auc, np.std(np.fromiter(avg_roc_auc, dtype=float))))
        for i in range(0, K_FOLDS):
            plt.plot(fprs[i], tprs[i], alpha=0.25)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.title("ROC Curve - %s %s"%(save_name, RATIO))
        plt.legend(loc='lower right', handlelength=0)
        plt.savefig(args.results + 'performance_' + save_name + '_ROC.png', format='png')
        plt.close()
        
