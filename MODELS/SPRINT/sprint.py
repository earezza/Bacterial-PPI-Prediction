#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    ---------- Original Work (SPRINT) this script uses: ----------
    Published in BMC Bioinformatics (2017) 18:485
    Title: 'SPRINT: ultrafast protein-protein interaction prediction of the entire human interactome'
    Authors: Yiwei Li and Lucian Ilie
    Journal: BMC Bioinformatics
    Volume: 18
    Number: 485
    Year: 2017
    Month: 11

    DOI: https://doi.org/10.1186/s12859-017-1871-x
    git: https://github.com/lucian-ilie/SPRINT

    ---------- This file ----------

Description:
    Python wrapper to run SPRINT as a subproccess.
    Uses all the same arguements as SPRINT with additional options.
    Makes running cross-validation easier.
    Includes evaluation of predictions using ROC and precision-recall curves.

@author: Eric Arezza
Last Updated: June 12, 2021
"""

__all__ = ['compile_SPRINT',
           'compute_HSPs',
           'predict_interactions',
           ]
__version__ = '1.0'
__author__ = 'Eric Arezza'

import os
import subprocess
import argparse
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import matplotlib.pyplot as plt


describe_help = 'python sprint.py -s sequences.fasta -f data.tsv -h HSP/file.hsp -k5'
parser = argparse.ArgumentParser(description=describe_help)
# All original arguements used in SPRINT, can also just perform these directly
parser.add_argument('-cs', '--compile_serial', help='Flag for compiling SPRINT (serial) if not already done', action='store_true')
parser.add_argument('-cp', '--compile_parallel', help='Flag for compiling SPRINT (parallel) if not already done', action='store_true')
# Common args
parser.add_argument('-p', '--protein_sequences', help='.fasta file of protein sequences', type=str)
parser.add_argument('-hsp', '--hsp_file', help='Path to HSP file to use if it exists, if None then a new HSP file is created', type=str, default=None)
# Args for compute_hsps
parser.add_argument('-Thit', '--hit_threshold', help='Determines how similar two s-mers have to be to form a hit', type=int, default=15)
parser.add_argument('-Tsim', '--sim_threshold', help='Determines how similar two k-mer regions should be in order to be considered a similar region', type=int, default=35)
parser.add_argument('-M', '--matrix', help='Scoring matrix 1: PAM120, 2: BLOSUM80, 3: BLOSUM62',
                    choices=[1, 2, 3], type=int, default=1)
parser.add_argument('-a', '--add', help='Path to .fasta file to add HSPs to the .hsp file given in -h', type=str, default=None)
# Args for make_predictions
parser.add_argument('-Thc', '--hc_threshold', help='Threshold considered to be a high count for removing regions of high similarity from sequences', type=int, default=40)
parser.add_argument('-tr', '--training_file', help='File contaiing space-separated PPIs used for training', type=str)
parser.add_argument('-pos', '--positive_testing_file', help='File contaiing space-separated positive PPIs used for testing', type=str)
parser.add_argument('-neg', '--negative_testing_file', help='File contaiing space-separated negative PPIs used for testing', type=str)
parser.add_argument('-o', '--output_file', help='Name used for saving files', type=str, nargs='?', default='output.txt')
parser.add_argument('-e', '--entire_proteome', help='Flag for performing entire proteome (all-to-all) prediction', action='store_true')
# Additional arguements and options for easily performing cross-validation
parser.add_argument('-s', '--sprint', help='Full path to SPRINT location (can be omitted if SPRINT is in same directory)', type=str, nargs='?', default=os.getcwd()+'/')
parser.add_argument('-file', help='Full path to labelled PPI dataset in (.tsv file, no header, using labels 0 (neg) and 1 (pos))', type=str)
parser.add_argument('-k', '--kfolds', help='Number of k-fold splits for cross-validation (default 5)', type=int, default=5)
parser.add_argument('-r', '--results', help='Path to directory for saving dataset files', 
                    type=str, default=os.getcwd()+'/Results/')
parser.add_argument('-d', '--delta', help='Imbalance ratio as positives/total (e.g. balanced = 0.5) for estimate of performance on hypothetical imbalanced data', type=float, nargs=1, required=False)
args = parser.parse_args()

RESULTS_DIR = args.results
if not os.path.exists(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)
        
if args.delta == None:
    IMBALANCE = 0.5
else:
    IMBALANCE = args.delta[0]
# Display ratio of positives:negatives
RATIO = '1:' + str(int((1/IMBALANCE) - 1))

# Calculate estimate for prevalence-corrected precision on imbalanced data
def recalculate_precision(df, precision, thresholds, ratio=IMBALANCE):
    delta = 2*ratio - 1
    new_precision = precision.copy()
    for t in range(0, len(thresholds)):
        tn, fp, fn, tp = metrics.confusion_matrix(df[df.columns[-1]], (df[df.columns[2]] >= thresholds[t]).astype(int)).ravel()
        lpp = tp/(tp+fn)
        lnn = tn/(tn+fp)
        if ratio != 0.5:
            new_precision[t] = (lpp*(1 + delta)) / ( (lpp*(1 + delta)) + ((1 - lnn)*(1 - delta)) )
        else:
            new_precision[t] = (lpp)/(lpp + (1-lnn))
    return new_precision

def compile_SPRINT(sprint_location, serial=False, parallel=False):
    current_dir = os.getcwd()
    os.chdir(sprint_location)
    if serial:
        print('Compiling SPRINT (serial)...')
        try:
            cmd_HSP = 'make compute_HSPs_serial'
            result = subprocess.run(cmd_HSP.split(), capture_output=True, text=True)
            print(result.stdout)
            print(result.stderr)
            time.sleep(1)
            
            cmd_predict = 'make predict_interactions_serial'
            result = subprocess.run(cmd_predict.split(), capture_output=True, text=True)
            print(result.stdout)
            print(result.stderr)
            time.sleep(1)
            
        except Exception as e:
            print('require g++, boost library')
            print('only require g++ under any Unix-like environment')
            print(e)
    
    if parallel:
        print('Compiling SPRINT (parallel)...')
        try:
            cmd_HSP = 'make compute_HSPs_parallel'
            result = subprocess.run(cmd_HSP.split(), capture_output=True, text=True)
            print(result.stdout)
            print(result.stderr)
            time.sleep(1)
            
            cmd_predict = 'make predict_interactions_parallel'
            result = subprocess.run(cmd_predict.split(), capture_output=True, text=True)
            print(result.stdout)
            print(result.stderr)
            time.sleep(1)
            
        except Exception as e:
            print('require g++, boost library, and OpenMP')
            print('require g++ and OpenMP')
            print(e)
    # Return to original directory
    os.chdir(current_dir)
    
def compute_HSPs(sprint_location, protein_sequences, hsp_filename, thit=15, tsim=35, m=1, add=False):
    print('Computing HSPs...')
    try:
        if add:
            if not os.path.exists(hsp_filename):
                print('No HSP file named - %s - exists to append HSPs for %s'%(hsp_filename, protein_sequences))
            else:
                cmd = '%sbin/compute_HSPs -add %s %s -Thit %s -Tsim %s -m %s'%(sprint_location, protein_sequences, hsp_filename, thit, tsim, m)
        else:
            if not os.path.exists(hsp_filename):
                cmd = '%sbin/compute_HSPs -p %s -h %s -Thit %s -Tsim %s -m %s'%(sprint_location, protein_sequences, hsp_filename, thit, tsim, m)
                result = subprocess.run(cmd.split(), capture_output=True, text=True)
                print(result.stdout)
                print(result.stderr)
            else:
                print('%s exists...'%hsp_filename)
        return True
    except Exception as e:
        print(e)
        return False
    
def predict_interactions(sprint_location, protein_sequences, hsp_filename, thc=40, train_pos=None, pos=None, neg=None, output_name=None, entire_proteome=False):
    if train_pos == None or (pos == None and neg == None and entire_proteome == False):
        return
    print('Making predictions...')
    try:
        if entire_proteome:
            cmd = '%sbin/predict_interactions -p %s -h %s -Thc %s -tr %s -e -o %s'%(sprint_location, protein_sequences, hsp_filename, thc, train_pos, output_name)
        else:
            cmd = '%sbin/predict_interactions -p %s -h %s -Thc %s -tr %s -pos %s -neg %s -o %s'%(sprint_location, protein_sequences, hsp_filename, thc, train_pos, pos, neg, output_name)
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        print(result.stdout)
        print(result.stderr)
    except Exception as e:
        print(e)
    
    
if __name__ == '__main__':
    print(args)
    start = time.time()
    # Compile if flagged True
    compile_SPRINT(args.sprint, serial=args.compile_serial, parallel=args.compile_parallel)
    
    # Create HSP file if none exists or add sequences if HSP file exists and add flagged True
    if not os.path.exists(args.hsp_file) or args.add:
        compute_HSPs(args.sprint, args.protein_sequences, args.hsp_file, thit=args.hit_threshold, tsim=args.sim_threshold, m=args.matrix, add=args.add)
    
    # Make predictions
    if (args.kfolds != 0 or args.kfolds != 1) and args.file != None:
        
        # Compute HSPs
        if not os.path.exists(args.hsp_file) or args.add:
            compute_HSPs(args.sprint, args.protein_sequences, args.hsp_file, thit=args.hit_threshold, tsim=args.sim_threshold, m=args.matrix, add=args.add)
        
        # Perform kfold using provided labelled PPI data
        print('Performing %s-fold cross-validation on %s'%(args.kfolds, args.file))
        
        # Read PPI data
        df = pd.read_csv(args.file, delim_whitespace=True, header=None)
        
        # Metrics for evaluation
        # For ROC curve
        tprs = {}
        roc_aucs = {}
        fprs = {}
        # For PR curve
        precisions = {}
        pr_aucs = {}
        recalls = {}
        
        # Create subsets
        filename = args.file.split('/')[-1].replace('.tsv', '')
        if args.output_file == 'output.txt':
            output = filename
        else:
            output = args.output_file.split('.')[0]
            
        kf = StratifiedKFold(n_splits=args.kfolds)
        fold = 0
        df_pred_all = pd.DataFrame()
        for train_index, test_index in kf.split(df[df.columns[:2]], df[df.columns[-1]]):
            
            # Isolate k-fold subset
            print('===== Fold - %s ====='%str(fold))
            train, test = df.iloc[train_index].reindex(), df.iloc[test_index].reindex()
            pos_train, pos_test = train[train[train.columns[-1]] == 1], test[test[test.columns[-1]] == 1]
            neg_train, neg_test = train[train[train.columns[-1]] == 0], test[test[test.columns[-1]] == 0]
            
            # Save subsets for SPRINT to read from...for predicting interactions
            pos_train.to_csv(RESULTS_DIR + output + '_pos_train_fold-' + str(fold) + '.txt', sep=' ', columns=[0,1], header=None, index=False)
            pos_test.to_csv(RESULTS_DIR + output + '_pos_test_fold-' + str(fold) + '.txt', sep=' ', columns=[0,1], header=None, index=False)
            neg_test.to_csv(RESULTS_DIR + output + '_neg_test_fold-' + str(fold) + '.txt', sep=' ', columns=[0,1], header=None, index=False)
            
            # Remove any files of same name to allow proper overwrite instead of appending
            if os.path.exists(RESULTS_DIR + 'predictions_' + output + '_fold-%s.txt'%str(fold)):
                os.remove(RESULTS_DIR + 'predictions_' + output + '_fold-%s.txt'%str(fold))
            if os.path.exists(RESULTS_DIR + 'predictions_' + output + '_fold-%s.txt.pos'%str(fold)):
                os.remove(RESULTS_DIR + 'predictions_' + output + '_fold-%s.txt.pos'%str(fold))
            if os.path.exists(RESULTS_DIR + 'predictions_' + output + '_fold-%s.txt.neg'%str(fold)):
                os.remove(RESULTS_DIR + 'predictions_' + output + '_fold-%s.txt.neg'%str(fold))
            
            # Run predict interactions for k-fold
            predict_interactions(args.sprint, args.protein_sequences, args.hsp_file, thc=args.hc_threshold, 
                             train_pos=RESULTS_DIR + output + '_pos_train_fold-' + str(fold) + '.txt', 
                             pos=RESULTS_DIR + output + '_pos_test_fold-' + str(fold) + '.txt', 
                             neg=RESULTS_DIR + output + '_neg_test_fold-' + str(fold) + '.txt', 
                             output_name=RESULTS_DIR + 'predictions_' + output + '_fold-%s.txt'%str(fold), 
                             entire_proteome=args.entire_proteome)
            
            # Read predictions for k-fold
            df_pred = pd.read_csv(RESULTS_DIR + 'predictions_' + output + '_fold-%s.txt'%str(fold), delim_whitespace=True, header=None)
            df_pred.to_csv(RESULTS_DIR + 'predictions_' + output + '_fold-%s.txt'%str(fold), sep=' ', columns=[0,1,2], header=None, index=False)
            df_pred_all = df_pred_all.append(df_pred)
            
            # Format results
            tested = pos_test.append(neg_test, ignore_index=True)
            tested.to_csv(RESULTS_DIR + output + '_test_fold-' + str(fold) + '.txt', sep=' ', header=None, index=False)
            os.remove(RESULTS_DIR + output + '_pos_test_fold-' + str(fold) + '.txt')
            os.remove(RESULTS_DIR + output + '_neg_test_fold-' + str(fold) + '.txt')
            
            # Remove redundant files
            os.remove(RESULTS_DIR + 'predictions_' + output + '_fold-%s.txt.pos'%str(fold))
            os.remove(RESULTS_DIR + 'predictions_' + output + '_fold-%s.txt.neg'%str(fold))
            
            # Evaluate k-fold performance and adjust for hypothetical imbalance
            precision, recall, thresholds = metrics.precision_recall_curve(df_pred[df_pred.columns[-1]], df_pred[df_pred.columns[2]])
            fpr, tpr, __ = metrics.roc_curve(df_pred[df_pred.columns[-1]], df_pred[df_pred.columns[2]])
            if IMBALANCE == 0.5:
                pr_auc = metrics.average_precision_score(df_pred[df_pred.columns[-1]], df_pred[df_pred.columns[2]])
            else:
                precision = recalculate_precision(df_pred, precision, thresholds)
                pr_auc = metrics.auc(recall, precision)
            roc_auc = metrics.roc_auc_score(df_pred[df_pred.columns[-1]], df_pred[df_pred.columns[2]])
            print('auc_roc=', roc_auc, '\nauc_pr=', pr_auc)
            
            # Add k-fold performance for overall average performance
            tprs[fold] = tpr
            fprs[fold] = fpr
            roc_aucs[fold] = roc_auc
            precisions[fold] = precision
            recalls[fold] = recall
            pr_aucs[fold] = pr_auc
            
            fold += 1
        
        # Get overall performance across all folds
        precision, recall, thresholds = metrics.precision_recall_curve(df_pred_all[df_pred_all.columns[-1]], df_pred_all[df_pred_all.columns[2]])
        fpr, tpr, __ = metrics.roc_curve(df_pred_all[df_pred_all.columns[-1]], df_pred_all[df_pred_all.columns[2]])
        if IMBALANCE == 0.5:
            pr_auc = metrics.average_precision_score(df_pred_all[df_pred_all.columns[-1]], df_pred_all[df_pred_all.columns[2]])
        else:
            precision = recalculate_precision(df_pred_all, precision, thresholds)
            pr_auc = metrics.auc(recall, precision)
        roc_auc = metrics.roc_auc_score(df_pred_all[df_pred_all.columns[-1]], df_pred_all[df_pred_all.columns[2]])
        
        # Write results to text file
        with open(RESULTS_DIR + output + '_results.txt', 'w') as f:
            f.write(('roc_auc=%.4f (+/- %.4f)' % (roc_auc, np.std(np.fromiter(roc_aucs.values(), dtype=float)))
                      + '\npr_auc=%.4f (+/- %.4f)' % (pr_auc, np.std(np.fromiter(pr_aucs.values(), dtype=float)))
                      + '\ntime=%.2f'%(time.time()-start) + '\n'))
        
        # Plot and save curves
        plt.figure
        plt.plot(recall, precision, color='black', label='AUC = %0.4f +/- %0.4f' % (pr_auc, np.std(np.fromiter(pr_aucs.values(), dtype=float))))
        for i in range(0, args.kfolds):
            plt.plot(recalls[i], precisions[i], alpha=0.25)
        plt.xlabel('Recall')
        plt.ylabel('Precision') 
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.title("Precision-Recall Curve - %s %s"%(args.output_file.split('.')[0], RATIO))
        plt.legend(loc='lower right', handlelength=0)
        plt.savefig(RESULTS_DIR + output + '_PR.png', format='png')
        plt.close()
        
        plt.figure
        plt.plot(fpr, tpr, color='black', label='AUC = %0.4f +/- %0.4f' % (roc_auc, np.std(np.fromiter(roc_aucs.values(), dtype=float))))
        for i in range(0, args.kfolds):
            plt.plot(fprs[i], tprs[i], alpha=0.25)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.title("ROC Curve - %s %s"%(args.output_file.split('.')[0], RATIO))
        plt.legend(loc='lower right', handlelength=0)
        plt.savefig(RESULTS_DIR + output + '_ROC.png', format='png')
        plt.close()

    else:
        predict_interactions(args.sprint, args.protein_sequences, args.hsp_file, thc=args.hc_threshold, 
                             train_pos=args.training_file, pos=args.positive_testing_file, neg=args.negative_testing_file, 
                             output_name=args.results + args.output_file, entire_proteome=args.entire_proteome)
        
        