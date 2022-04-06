#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description:
    Performs PPI prediction using RP datasets.
    RP datasets generated using extract_rp_features.py based on PPI predictions from other models.
    
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
#from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
#import xgboost as xgb
from lightgbm.sklearn import LGBMClassifier
from sklearn.svm import SVC
from lightgbm import plot_importance
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
    
    if not os.path.exists(args.results):
        os.mkdir(args.results)
    
    # ================== FOR SINGLE TRAIN/TEST RUNS ==================
    if FILES == None and args.train != None and args.test != None:
        t_start = time.time()
        output = args.name
        print('Training on %s\nTesting on %s\n'%(args.train.split('/')[-1], args.test.split('/')[-1]))
        output += '\nTraining on %s\nTesting on %s\n'%(args.train.split('/')[-1], args.test.split('/')[-1])
        
        # Load data
        df_train = pd.read_csv(args.train, delim_whitespace=True)
        df_train.replace(to_replace=np.nan, value=0, inplace=True)
        df_test = pd.read_csv(args.test, delim_whitespace=True)
        df_test.replace(to_replace=np.nan, value=0, inplace=True)
        
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
        if args.cme:
            clf = SVC(C=0.6,
                      kernel='sigmoid',
                      gamma='scale',
                      probability=True,
                      random_state=13052021,
                      )
        else:
            clf = LGBMClassifier(random_state=13052021,
                             boosting_type='goss', 
                             learning_rate=0.1, 
                             num_leaves=50,
                             max_depth=10, 
                             min_data_in_leaf=50,
                             n_estimators=150,
                             path_smooth=0.1,
                             )
        
        pipe = Pipeline([('scaler', StandardScaler()), ('metaclf', clf)])
        
        # Fit model with training data
        pipe.fit(X_train, y_train)
        #clf.fit(X_train, y_train)
        #var_imp_df = pd.DataFrame([df_train.columns, clf.feature_importances_]).T
        #var_imp_df.sort_values(by=[1], ascending=False, inplace=True)
        #plot_importance(clf.booster_)
        
        # Record PPI binary predictions
        pred = pipe.predict(X_test)
        #pred = clf.predict(X_test)
        ppi_bin = pd.DataFrame(pairs_test, columns=[df_test.columns[0], df_test.columns[1]])
        ppi_bin.insert(2, 2, pred)
        
        # Record PPI prediction probabilities
        pred_probs = pipe.predict_proba(X_test)
        #pred_probs = clf.predict_proba(X_test)
        ppi_probs = pd.DataFrame(np.append(pairs_test, pred_probs, axis=1), columns=[df_test.columns[0], df_test.columns[1], 2, 3])
        ppi_probs.drop(columns=[2], inplace=True)
        ppi_probs.to_csv(args.results + 'predictions_' + save_name + '.tsv', sep='\t', header=None, index=False)
        
        # If only one class in labels (predicting unknowns)
        if pd.Series(y_test).unique().shape[0] < 2:
            print('No evaluation to be done.')
            exit()
        
        # Get performance metricss of binary evaluation (threshold=0.5)
        tn, fp, fn, tp = metrics.confusion_matrix(y_test, pred).ravel()
        print('TP = %0.0f \nFP = %0.0f \nTN = %0.0f \nFN = %0.0f'%(tp, fp, tn, fn))
        output += '\nTP = %0.0f \nFP = %0.0f \nTN = %0.0f \nFN = %0.0f'%(tp, fp, tn, fn)
        
        # For imbalanced classification metrics
        if args.delta != 0.5:
            accuracy, precision, recall, specificity, f1, mcc = recalculate_metrics_to_imbalance(tp, tn, fp, fn, args.delta)
        else:
            try:
                accuracy = round((tp+tn)/(tp+fp+tn+fn), 5)
            except ZeroDivisionError:
                accuracy = np.nan
            try:
                precision = round(tp/(tp+fp), 5)
            except ZeroDivisionError:
                precision = np.nan
            try:
                recall = round(tp/(tp+fn), 5)
            except ZeroDivisionError:
                recall = np.nan
            try:
                specificity = round(tn/(tn+fp), 5)
            except ZeroDivisionError:
                specificity = np.nan
            try:
                f1 = round((2*tp)/(2*tp+fp+fn), 5)
            except ZeroDivisionError:
                f1 = np.nan
            try:
                mcc = round( ( ((tp*tn) - (fp*fn)) / np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) ), 5)
            except ZeroDivisionError:
                mcc = np.nan
        
        print('Accuracy =', accuracy, '\nPrecision =', precision, '\nRecall =', recall, '\nSpecificity =', specificity, '\nF1 =', f1, '\nMCC =', mcc)
        output += '\nAccuracy = ' + str(accuracy) + '\nPrecision = ' + str(precision) + '\nRecall = '+ str(recall) + '\nSpecificity = ' + str(specificity) + '\nF1 = ' + str(f1) + '\nMCC = ' + str(mcc)
        
        auc_roc = metrics.roc_auc_score(y_test, pred_probs[:, 1])
        auc_pr = metrics.average_precision_score(y_test, pred_probs[:, 1])
        fpr, tpr, __ = metrics.roc_curve(y_test, pred_probs[:, 1])
        
        # Get performance metrics for curve plotting
        # Evaluate performance and adjust for hypothetical imbalance
        precision, recall, thresholds = metrics.precision_recall_curve(y_test, pred_probs[:, 1])
        df_pred = pd.DataFrame(data={1:y_test, 0:pred_probs[:, 1]})
        
        # Evaluate performance and adjust for hypothetical imbalance
        precision, recall, thresholds = metrics.precision_recall_curve(df_pred[1], df_pred[0])
        if args.delta != 0.5:
            precision = recalculate_precision(df_pred, precision, thresholds, args.delta)
        fpr, tpr, __ = metrics.roc_curve(df_pred[1], df_pred[0])
        if args.delta == 0.5:
            pr_auc = metrics.average_precision_score(df_pred[1], df_pred[0])
        else:
            pr_auc = metrics.auc(recall, precision)
        roc_auc = metrics.roc_auc_score(df_pred[1], df_pred[0])
        
        print('auc_roc=%.6f'%(roc_auc) + '\nauc_pr=%.6f'%(pr_auc))
        output += '\nAUC_ROC = %0.5f'%roc_auc + '\nAUC_PR = %0.5f\n'%pr_auc
        duration = time.time() - t_start
        output += '\ntime(sec.)=%.2f'%(duration) + '\n'
        # Write results to file
        result_filename = args.results + 'results_' + save_name + '.txt'
        with open(result_filename, 'w') as fp:
            fp.write(output)
        
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
            df.replace(to_replace=np.nan, value=0, inplace=True)
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
            df.replace(to_replace=np.nan, value=0, inplace=True)
            if args.name == '':
                save_name = f.split('.')[0].split('/')[-1]
            else:
                save_name = args.name
        
        # Define data and labels
        pairs = np.array(df[df.columns[0:2]])
        X = np.array(df[df.columns[2:-1]])
        y = np.array(df[df.columns[-1]])
        
        
        # Define data partiioning
        kf = StratifiedKFold(n_splits=K_FOLDS)

        if args.cme:
            clf = SVC(C=0.6,
                      kernel='sigmoid',
                      gamma='scale',
                      probability=True,
                      random_state=13052021,
                      )
        else:
            clf = LGBMClassifier(random_state=13052021,
                             boosting_type='goss', 
                             learning_rate=0.1, 
                             num_leaves=50,
                             max_depth=10, 
                             min_data_in_leaf=50,
                             n_estimators=150,
                             path_smooth=0.1,
                             ) 
        
        pipe = Pipeline([('scaler', StandardScaler()), ('metaclf', clf)])
            
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
        
        # Perform k-fold predictions and evaluation
        k = 0
        prob_predictions = pd.DataFrame()
        bin_predictions = pd.DataFrame()
        output = args.name
        t_start = time.time()
        for train, test in kf.split(X, y):
            
            print('===== Fold-%s ====='%k)
            output += '===== Fold-%s ====='%k
            
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
            print('TP = %0.0f \nFP = %0.0f \nTN = %0.0f \nFN = %0.0f'%(tp, fp, tn, fn))
            output += '\nTP = %0.0f \nFP = %0.0f \nTN = %0.0f \nFN = %0.0f'%(tp, fp, tn, fn)
            
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
        
            auc_roc = metrics.roc_auc_score(y[test], pred_probs[:, 1])
            auc_pr = metrics.average_precision_score(y[test], pred_probs[:, 1])
            fpr, tpr, __ = metrics.roc_curve(y[test], pred_probs[:, 1])
            
            print('Accuracy =', accuracy, '\nPrecision =', precision, '\nRecall =', recall, '\nSpecificity =', specificity, '\nF1 =', f1, '\nMCC =', mcc)
            output += '\nAccuracy = ' + str(accuracy) + '\nPrecision = ' + str(precision) + '\nRecall = '+ str(recall) + '\nSpecificity = ' + str(specificity) + '\nF1 = ' + str(f1) + '\nMCC = ' + str(mcc)
            
            # Get performance metricss for curve plotting
            # Evaluate k-fold performance and adjust for hypothetical imbalance
            precision, recall, thresholds = metrics.precision_recall_curve(y[test], pred_probs[:, 1])
            df_pred = pd.DataFrame(data={1:y[test], 0:pred_probs[:, 1]})
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
            tprs[k] = tpr
            fprs[k] = fpr
            roc_aucs[k] = roc_auc
            precisions[k] = precision
            recalls[k] = recall
            pr_aucs[k] = pr_auc
            
            k += 1
        
        duration = time.time() - t_start
        print(duration)
        
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
        print('----- Results -----\n' + evaluation)
        output += evaluation
        # Write results to file
        result_filename = args.results + 'results_' + save_name + '.txt'
        with open(result_filename, 'w') as fp:
            fp.write(output)
        
        df_pred_total = pd.DataFrame(data={1:y, 0:prob_predictions[prob_predictions.columns[-1]]})
        
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
        plt.legend(loc=leg_loc, handlelength=0)
        plt.savefig(args.results + args.name + '_PR.png', format='png')
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
        plt.legend(loc=leg_loc, handlelength=0)
        plt.savefig(args.results + args.name + '_ROC.png', format='png')
        plt.close()

        
