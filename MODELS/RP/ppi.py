#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description:
    Useful functions for handling PPI data.

@author: Eric Arezza
"""

__all__ = [
    'format_uniprot_fasta',
    'fasta_to_df',
    'remove_redundant_pairs',
    'get_matching_pairs',
    'remove_matching_pairs',
    'read_df',
    'average_all_to_all',
    ]

import os
import pandas as pd
import numpy as np

# Requires df as pd.read_csv(file, sep='\n', header=None)
def format_uniprot_fasta(df):
    uniprot = df.copy()
    uniprot.dropna(inplace=True)
    uniprot.reset_index(drop=True, inplace=True)
    prots = uniprot[uniprot[0].str.contains('>')]
    mapping = {}
    for i in range(1, len(prots.index)):
        mapping['>' + prots.iloc[i-1][0].split('|')[1]] = ''.join(uniprot[0].loc[prots.index[i-1]+1:prots.index[i]-1].values)
    '''
    for i in range(1, len(prots.index)):
        mapping[prots.iloc[i-1][0]] = ''.join(uniprot[0].loc[prots.index[i-1]+1:prots.index[i]-1].values)
    mapping[prots.iloc[-1][0]] = ''.join(uniprot[0].loc[prots.index[-1]+1:uniprot.shape[0]].values)
    '''
    # Add last
    mapping['>' + prots.iloc[-1][0].split('|')[1]] = ''.join(uniprot[0].loc[prots.index[-1]+1:uniprot.shape[0]].values)
    # Create formatted df
    df_formatted = pd.DataFrame(data={0: np.array(list(mapping.keys())), 1: np.array(list(mapping.values()))})
    return df_formatted

# Requires df as pd.read_csv(fastafile, sep='\n', header=None)
def fasta_to_df(df):
    fasta = df.copy()
    prot = fasta.iloc[::2, :].reset_index(drop=True)
    seq = fasta.iloc[1::2, :].reset_index(drop=True)
    prot.insert(1, 1, seq)
    return prot
    
# Requires df as pd.read_csv(file, delim_whitespace=True, header=None) columns as <proteinA> <proteinB>
def remove_redundant_pairs(df_ppi):
    df = df_ppi.copy()
    df.sort_values(by=[df.columns[0], df.columns[1]], ignore_index=True, inplace=True)
    # Get only unique PPIs (using set automatically sorts AB and BA such that they will all be AB)
    pairs = pd.DataFrame([set(p) for p in df[df.columns[:2]].values])
    # Fill in for self-interacting proteins
    pairs[pairs.columns[1]] = pairs[pairs.columns[1]].fillna(pairs[pairs.columns[0]])
    pairs = pairs.drop_duplicates()
    df_unique = pairs.copy()
    df_unique.reset_index(drop=True, inplace=True)
    # Keep PPI labels/scores if exists
    if len(df.columns) > 2:
        df_unique_rev = df_unique.copy()
        df_unique_rev[[df_unique_rev.columns[0], df_unique_rev.columns[1]]] = df_unique_rev[[df_unique_rev.columns[1], df_unique_rev.columns[0]]]
        
        df_labelled = df.merge(df_unique, on=[df_unique.columns[0], df_unique.columns[1]])
        df_labelled = df_labelled.append(df.merge(df_unique_rev, on=[df_unique_rev.columns[0], df_unique_rev.columns[1]]))
        df_labelled.sort_values(by=[df_labelled.columns[-1]], ascending=False, ignore_index=True, inplace=True)
        df_labelled = df_labelled.drop_duplicates(subset=[df_labelled.columns[0], df_labelled.columns[1]])
        df_out = df_labelled.reset_index(drop=True)
    else:
        df_out = df_unique.copy()
        df_out.sort_values(by=[df_out.columns[0], df_out.columns[1]], ignore_index=True, inplace=True)
    
    return df_out

# ========= Intended for processing cross-validation all-to-all predictions for Reciprocal Perspective ===========
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
    
def remove_matching_pairs(df_1, df_2):
    # First get matches
    if df_1.shape > df_2.shape:
        df_test = df_1.copy()
        df_train = df_2.copy()
    else:
        df_test = df_2.copy()
        df_train = df_1.copy()
    match = get_matching_pairs(df_train, df_test)
    # Get all matches to be removed (consider PPIs where A-B is reversed as B-A)
    match_rev = match.copy()
    match_rev[[match_rev.columns[0], match_rev.columns[1]]] = match_rev[[match_rev.columns[1], match_rev.columns[0]]]
    match_all = match.append(match_rev)
    match_all = match_all.drop_duplicates()
    match_all.reset_index(drop=True, inplace=True)
    merge = match_all.merge(df_test, on=[match_all.columns[0], match_all.columns[1]], how='outer', indicator=True)
    # Remove matched PPIs from df_test predictions
    removed = merge[merge['_merge'] == 'right_only']
    removed = removed.drop(columns=['2_x', '2_y', '_merge'])
    removed.reset_index(drop=True, inplace=True)
    # Returns as <ProteinA> <ProteinB> <score>
    return removed

def read_df(file):
    # Read PPIs accounting for formatting differences
    if 'PIPR' in file.upper():
        df = pd.read_csv(file, delim_whitespace=True)
        df.rename(columns={df.columns[0]: 0, df.columns[1]: 1, df.columns[2]: 2}, inplace=True)
    elif 'SPRINT' in file.upper():
        df = pd.read_csv(file, delim_whitespace=True, header=None)
        df.insert(len(df.columns), len(df.columns), np.ones(df.shape[0]).astype(int))
    elif 'DEEPFE' in file.upper():
        deepfe_files = os.listdir(path=file + '/')
        posA = pd.DataFrame()
        posB = pd.DataFrame()
        negA = pd.DataFrame()
        negB = pd.DataFrame()
        for f in deepfe_files:
            if 'pos' in f.lower() and 'proteina' in f.lower():
                posA = pd.read_csv(file +'/' + f, sep='\n', header=None)
                posA = fasta_to_df(posA)[0].str.replace('>', '')
            if 'pos' in f.lower() and 'proteinb' in f.lower():
                posB = pd.read_csv(file +'/' + f, sep='\n', header=None)
                posB = fasta_to_df(posB)[0].str.replace('>', '')
            if 'neg' in f.lower() and 'proteina' in f.lower():
                negA = pd.read_csv(file +'/' + f, sep='\n', header=None)
                negA = fasta_to_df(negA)[0].str.replace('>', '')
            if 'neg' in f.lower() and 'proteinb' in f.lower():
                negB = pd.read_csv(file +'/' + f, sep='\n', header=None)
                negB = fasta_to_df(negB)[0].str.replace('>', '')
        pos = pd.DataFrame(data={0: posA, 1: posB, 2: np.ones(posA.shape[0]).astype(int)})
        neg = pd.DataFrame(data={0: negA, 1: negB, 2: np.ones(negA.shape[0]).astype(int)})
        df = pos.append(neg, ignore_index=True)
    elif 'DPPI' in file.upper():
        df = pd.read_csv(file, sep=',', header=None)
    else:
        df = pd.read_csv(file, delim_whitespace=True, header=None)
    
    return df

def average_all_to_all(cv_subset_dir_path, prediction_results_dir_path):
    # Consolidates cross-validation data for all-to-all predictions
    
    # Get training PPI subsets to remove from all-to-all PPI predictions
    cv_files = os.listdir(path=cv_subset_dir_path)
    train_files = [ x for x in cv_files if 'train' in x and '_neg' not in x ]
    train_files.sort()
    
    # Get all-to-all PPI predictions to compile avgerage PPI scores
    prediction_files = os.listdir(path=prediction_results_dir_path)
    pred_files = [ x for x in prediction_files if 'train' in x and 'prediction' in x ]
    pred_files.sort()
    
    if len(train_files) != len(pred_files):
        print('CV subsets and all-to-all predictions mismatch')
        return pd.DataFrame()
    
    # Compile all_to_all predictions from cross-validation
    df_cv = pd.DataFrame()
    for i in range(0, len(train_files)):
        
        # Read trained PPI subset
        #trained = pd.read_csv(cv_subset_dir_path + train_files[i], delim_whitespace=True, header=None)
        trained = read_df(cv_subset_dir_path + train_files[i])
        
        # Read all-to-all PPI predictions
        predictions = pd.read_csv(prediction_results_dir_path + pred_files[i], delim_whitespace=True, header=None)
        #predictions = read_df(prediction_results_dir_path + pred_files[i])
        
        # Remove training PPIs from all-to-all tested predictions
        tested = remove_matching_pairs(trained, predictions)
        
        if i == 0:
            df_cv = tested.copy()
        else:
            df_cv = df_cv.merge(tested, how='outer', on=[df_cv.columns[0], df_cv.columns[1]], suffixes=(None , '_fold_%s'%i))
    
    # Format and average scores
    df_cv.rename(columns={2: '2_fold_0'}, inplace=True)
    df_cv.columns = df_cv.columns[:2].append(pd.Index([i.replace('2_', '') for i in df_cv.columns.values if type(i) == str]))
    df_cv['mean'] = df_cv.iloc[:, 2:].mean(axis=1)
    df_cv.sort_values(by=[df_cv.columns[0], df_cv.columns[1]], inplace=True)
    df_cv.reset_index(drop=True, inplace=True)
    
    return df_cv
    
