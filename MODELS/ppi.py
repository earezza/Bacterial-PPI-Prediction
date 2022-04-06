#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description:
    Functions for handling PPI data and script testing.

@author: Eric Arezza
"""

__all__ = [
    'format_uniprot_fasta',
    'fasta_to_df',
    'remove_redundant_pairs',
    'get_matching_pairs',
    'remove_matching_pairs',
    'read_df',
    'map_uniprot',
    'check_ppi_confidence',
    'run_cdhit',
    'perform_rp_traintests',
    'recalculate_precision',
    'recalculate_metrics_to_imbalance',
    'get_top_interactors',
    ]

import os
import pandas as pd
import numpy as np
import math
#import tqdm
import urllib.parse
import urllib.request
from io import StringIO
import subprocess
import time
from shutil import copy2
from sklearn import metrics
import matplotlib.pyplot as plt
#from matplotlib_venn import venn2, venn3, venn3_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
#from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from lgbm import LGBMClassifier
from lightgbm import plot_split_value_histogram, plot_importance
import xgb
#from sklearn.neural_network import MLPClassifier
#from scipy.stats import f_oneway, ttest_ind

# Requires df as pd.read_csv(file, sep='\n', header=None)
def format_uniprot_fasta(df):
    uniprot = df.copy()
    uniprot.dropna(inplace=True)
    uniprot.reset_index(drop=True, inplace=True)
    prots = uniprot[uniprot[0].str.contains('>')]
    mapping = {}
    for i in range(1, len(prots.index)):
        mapping['>' + prots.iloc[i-1][0].split('|')[1]] = ''.join(uniprot[0].loc[prots.index[i-1]+1:prots.index[i]-1].values)
    
    # Add last
    mapping['>' + prots.iloc[-1][0].split('|')[1]] = ''.join(uniprot[0].loc[prots.index[-1]+1:uniprot.shape[0]].values)

    # Create formatted df
    df_formatted = pd.DataFrame(data={0: pd.Series(mapping.keys()), 1: pd.Series(mapping.values())})
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
    df_unique.rename(columns={df_unique.columns[0]: df.columns[0], df_unique.columns[1]: df.columns[1]}, inplace=True)
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
    
def map_uniprot(ppi, query_from='GENENAME', query_to='SWISSPROT', columns='id,sequence,reviewed,organism'):
    df = ppi.copy()
    geneIDs = df[df.columns[0]].append(df[df.columns[1]]).unique()
    geneIDs_query = str(geneIDs.tolist()).strip('[').strip(']').replace("'", "").replace(',', '')
    url = 'https://www.uniprot.org/uploadlists/'
    params = {
    'from': query_from,
    'to': query_to,
    'format': 'tab',
    'columns': columns,
    'query': geneIDs_query,
    }
    print('\tQuerying UniProt for mappings...')
    response = ''
    for x in range(0, 3):
        try:
            data = urllib.parse.urlencode(params)
            data = data.encode('utf-8')
            req = urllib.request.Request(url, data)
            with urllib.request.urlopen(req) as webresults:
               response = webresults.read().decode('utf-8')
        except:
            print('\tError connecting to UniProt, trying again...')
    if response == '':
        print('\tNo UniProt mapping results found...No dataset created.')
        return pd.DataFrame(), pd.DataFrame()
    else:
        df_uniprot = pd.read_csv(StringIO(response), sep='\t', dtype=str)
        gene_list = df_uniprot.columns.tolist()[-1]
        df_uniprot.rename(columns={gene_list: 'GeneID', 'Entry': 'ProteinID'}, inplace=True)
        
        # Map IDs to BioGRID dataset, remove unmapped genes, and rename columns
        mapped = df.copy()
        # For ProteinIDs with more than one Entrez Gene ID
        df_uniprot['GeneID'] = df_uniprot['GeneID'].str.split(',')
        df_uniprot = df_uniprot.explode('GeneID', ignore_index=True)
        
        refdict = pd.Series(df_uniprot['ProteinID'].values, index=df_uniprot['GeneID']).to_dict()
        
        mapped[mapped.columns[0]] = mapped[mapped.columns[0]].map(refdict)
        mapped[mapped.columns[1]] = mapped[mapped.columns[1]].map(refdict)
        mapped.dropna(subset=[mapped.columns[0], mapped.columns[1]], inplace=True)
        mapped.rename(columns={mapped.columns[0]:'Protein A', mapped.columns[1]:'Protein B'}, inplace=True)
        mapped.reset_index(inplace=True, drop=True)
        
        # Repeat for protein sequences mapping for .fasta
        proteins = pd.Series(mapped['Protein A'].append(mapped['Protein B']).unique(), name='ProteinID')
        sequences = pd.Series(mapped['Protein A'].append(mapped['Protein B']).unique(), name='Sequence')
        refdictseq = pd.Series(df_uniprot['Sequence'].values, index=df_uniprot['ProteinID']).to_dict()
        fasta = pd.DataFrame(proteins).join(sequences)
        fasta['Sequence'] = fasta['Sequence'].map(refdictseq)
        fasta.dropna(inplace=True)
        fasta.drop_duplicates(inplace=True)
        fasta.reset_index(drop=True, inplace=True)
        
        # Drop all mapped interactions containing proteins with no sequence
        proteins_with_sequence = fasta['ProteinID']
        mapped = mapped[mapped['Protein A'].isin(proteins_with_sequence)]
        mapped = mapped[mapped['Protein B'].isin(proteins_with_sequence)]
        mapped.dropna(inplace=True)
        mapped.drop_duplicates(inplace=True)
        mapped.reset_index(drop=True, inplace=True)
        
        fasta['ProteinID'] = '>' + fasta['ProteinID']
        
        return mapped, fasta

def check_ppi_confidence(df_biogrid, level=2):
    df = df_biogrid.copy()
    pubmed_col = 'Publication Source'
    # Get PPIs resetting order such that protein interactions AB and BA are all listed as AB
    ppi = pd.DataFrame([set(p) for p in df[[df.columns[0], df.columns[1]]].values])
    # Fill in for self-interacting proteins
    if ppi.empty == False:
        ppi[ppi.columns[1]] = ppi[ppi.columns[1]].fillna(ppi[ppi.columns[0]])
    else:
        return pd.DataFrame()
    
    # Level 0: all unique PPIs
    if level == 0:
        ppi.drop_duplicates(inplace=True)
        df = df.iloc[ppi.index].reset_index(drop=True)
        return df
    
    # Level 1: all unique PPIs with multiple instances
    elif level == 1:
        ppi = ppi[ppi.duplicated(keep='first')]
        ppi.drop_duplicates(inplace=True)
        df = df.iloc[ppi.index].reset_index(drop=True)
        return df
    
    # Level 2: all unique PPIs with multiple instances having more than 1 publication source
    elif level == 2:
        ppi = ppi[ppi.duplicated(keep=False)]
        ppi.insert(len(ppi.columns), pubmed_col, df.iloc[ppi.index][pubmed_col])
        group = ppi.groupby([ppi.columns[0], ppi.columns[1]])[pubmed_col].apply(lambda x: x.unique()).reset_index()
        ppi_2 = group[[len(group.iloc[i][pubmed_col]) > 1 for i in group.index]]
        if not ppi_2.empty:
            # Reset PPI order in df so pairs are ordered as in ppi_2 pairs for merging
            ppi_2 = ppi_2.rename(columns={ppi_2.columns[0]: df.columns[0], ppi_2.columns[1]: df.columns[1]})
            pairs = pd.DataFrame([set(p) for p in df[[df.columns[0], df.columns[1]]].values])
            pairs[pairs.columns[1]] = pairs[pairs.columns[1]].fillna(pairs[pairs.columns[0]])
            df[[df.columns[0], df.columns[1]]] = pairs[[0,1]].values
            df = df.merge(ppi_2, on=[df.columns[0], df.columns[1]])
            df.drop(columns=[pubmed_col + '_y'], inplace=True)
            df.rename(columns={pubmed_col + '_x': pubmed_col}, inplace=True)
            df.drop_duplicates(subset=[df.columns[0], df.columns[1]], inplace=True)
            df.reset_index(drop=True, inplace=True)
        else:
            df = pd.DataFrame()
        return df
    else:
        return df

def check_ppi_confidence_(df_biogrid, level=2, by='Publication Source'):
    df = df_biogrid.copy()
    
    # Get PPIs resetting order such that protein interactions AB and BA are all listed as AB
    ppi = pd.DataFrame([set(p) for p in df[['Entrez Gene Interactor A', 'Entrez Gene Interactor B']].values])
    # Fill in for self-interacting proteins
    if ppi.empty == False:
        ppi[ppi.columns[1]] = ppi[ppi.columns[1]].fillna(ppi[ppi.columns[0]]).astype(int)
    else:
        return pd.DataFrame()
    
    # Level 0: all unique PPIs
    if level == 0:
        ppi.drop_duplicates(inplace=True)
        df = df.iloc[ppi.index].reset_index(drop=True)
        return df
    
    # Level 1: all unique PPIs with multiple instances
    elif level == 1:
        ppi = ppi[ppi.duplicated(keep='first')]
        ppi.drop_duplicates(inplace=True)
        df = df.iloc[ppi.index].reset_index(drop=True)
        return df
    
    # Level 2: all unique PPIs with multiple instances having more than 1 publication source
    elif level == 2:
        ppi = ppi[ppi.duplicated(keep=False)]
        ppi.insert(len(ppi.columns), by, df.iloc[ppi.index][by])
        group = ppi.groupby([ppi.columns[0], ppi.columns[1]])[by].apply(lambda x: x.unique()).reset_index()
        ppi_2 = group[[len(group.iloc[i][by]) > 1 for i in group.index]]
        if not ppi_2.empty:
            # Reset PPI order in df so pairs are ordered as in ppi_2 pairs for merging
            ppi_2 = ppi_2.rename(columns={ppi_2.columns[0]: 'Entrez Gene Interactor A', ppi_2.columns[1]: 'Entrez Gene Interactor B'})
            pairs = pd.DataFrame([set(p) for p in df[['Entrez Gene Interactor A', 'Entrez Gene Interactor B']].values])
            pairs[pairs.columns[1]] = pairs[pairs.columns[1]].fillna(pairs[pairs.columns[0]]).astype(int)
            df[['Entrez Gene Interactor A', 'Entrez Gene Interactor B']] = pairs[[0,1]].values
            df = df.merge(ppi_2, on=['Entrez Gene Interactor A', 'Entrez Gene Interactor B'])
            df.drop(columns=[by + '_y'], inplace=True)
            df.rename(columns={by + '_x': by}, inplace=True)
            df.drop_duplicates(subset=['Entrez Gene Interactor A', 'Entrez Gene Interactor B'], inplace=True)
            df.reset_index(drop=True, inplace=True)
        else:
            df = pd.DataFrame()
        return df
    else:
        return df

# Read hsp file as pd.read_csv(file, sep='\n', header=None)
def hsp_to_dict(hsp):
    hsp_dict = {}
    for i in range(0, hsp.shape[0]):
        if hsp.iloc[i][0][0] == '>':
            pairs = hsp.iloc[i][0]
            hsp_dict[pairs] = list()
        else:
            hsp_dict[pairs].insert(len(hsp_dict[pairs]), hsp.iloc[i][0])
            
    return hsp_dict

def hsp_dict_to_df(hsp):
    df = pd.DataFrame(hsp.keys())
    df.insert(1, 1, [ hsp[df.iloc[i][0]] for i in range(0, len(hsp)) ])
    return df

# Ecoli_Proteome proteins without PSSM results
no_pssm = [
'A0A7H2C7A0',
'P0DSH2',
'P0DSG6',
'P0AD92',
'A0A7H2C785',
'P0DPN8',
'P0DSE6',
'P0DSG0',
'A0A7H2C787',
'P0DSF6',
'C1P616',
'C1P615',
'A0A7H2C779',
'A0A7H2C794',
'P0AD72',
'P0DUM3',
'P0DSF7',
'P0DSF1',
'P0DSH8',
'A0A7H2C774',
'A0A7H2C7A7',
'P0DPM6',
'P0DPM5',
'A0A7H2C788',
'P0DSG7',
'P0DPN7',
'P0AD74',
'P0DSF9',
'A0A7H2C798',
'A0A7H2C784',
'A0A7H2C795',
'P0DPO6'
]

has_X = ['P45766', 'P76000', 'P37003', 'P33369', 'P75901', 'P39901', 'P58095']

ecoli_1 = 'BIOGRID-ORGANISM-4.4.203.tab3/BIOGRID-ORGANISM-Escherichia_coli_K12_MC4100_BW2952-4.4.203.tab3.txt'
ecoli_2 = 'BIOGRID-ORGANISM-4.4.203.tab3/BIOGRID-ORGANISM-Escherichia_coli_K12_MG1655-4.4.203.tab3.txt'
ecoli_3 = 'BIOGRID-ORGANISM-4.4.203.tab3/BIOGRID-ORGANISM-Escherichia_coli_K12_W3110-4.4.203.tab3.txt'
ecoli_4 = 'BIOGRID-ORGANISM-4.4.203.tab3/BIOGRID-ORGANISM-Escherichia_coli_K12-4.4.203.tab3.txt'

df_1 = pd.read_csv(ecoli_1, sep='\t')
df_2 = pd.read_csv(ecoli_2, sep='\t')
df_3 = pd.read_csv(ecoli_3, sep='\t')
df_4 = pd.read_csv(ecoli_4, sep='\t')
# Combine all raw files
df = pd.concat([df_1, df_2, df_3, df_4])
# Apply filters...
# Pull only E.coli-E.coli interactions and make uniform
orgs = ['Escherichia coli (K12/MC4100/BW2952)', 'Escherichia coli (K12/MG1655)', 'Escherichia coli (K12/W3110)', 'Escherichia coli (K12)',]
df = df[(df['Organism Name Interactor A'].isin(orgs)) & (df['Organism Name Interactor B'].isin(orgs))]
df['Organism Name Interactor A'] = 'Ecoli'
df['Organism Name Interactor B'] = 'Ecoli'
df['Organism ID Interactor A'] = 83333
df['Organism ID Interactor B'] = 83333

def run_cdhit(fasta_filename, new_fasta_filename, cdhit='cd-hit', threshold=0.6):
    if threshold == 1.0:
        print('\tNo sequence clustering required...')
        #return read_fasta(fasta_filename)
    elif threshold < 1.0 and threshold >= 0.7:
        words = 5
    elif threshold < 0.7 and threshold >= 0.6:
        words = 4
    elif threshold < 0.6 and threshold >= 0.5:
        words = 3
    elif threshold < 0.5 and threshold >= 0.4:
        words = 2
    #else:
    #    return read_fasta(fasta_filename)
    try:
        cmd = '%s -i %s -o %s.new -c %s -n %s'%(cdhit, fasta_filename, new_fasta_filename, threshold, words)
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        print(result.stdout)
        print(result.stderr)
        time.sleep(1)
    except Exception as e:
        print(e)
        print('\tCD-HIT not working, returning original .fasta...')
        
def copy_pssm(directory_from, directory_to):
    pssms = np.array([ i for i in os.listdir(directory_from) if '.txt' not in i ])
    to_copy = np.array([ i.replace('.txt', '') for i in os.listdir(directory_to) ])
    if pd.Series(to_copy).isin(pssms).sum() != to_copy.shape[0]:
        print("%s pssms not found to copy"%(to_copy.shape[0] - pd.Series(to_copy).isin(pssms).sum()))
        return False
    else:
        for p in to_copy:
            copy2(directory_from + p, directory_to)
        return True

# ========================== TIME SAVER ====================================
def perform_rp_traintests(train, test, results, name, k=10, dppi=False):
    t_start = time.time()
    if k == 0:
        try:
            if dppi:
                cmd = 'python rp_ppi_classifier.py -train %s -test %s -r %s -n %s'%(train, test, results, name)
            else:
                cmd = 'python rp_ppi_classifier.py -train %s -test %s -r %s -n %s'%(train, test, results, name)
            result = subprocess.run(cmd.split(), capture_output=True, text=True)
            print(result.stdout)
            print(result.stderr)
            time.sleep(1)
        except Exception as e:
            print(e)
    else:
        for i in range(0, k):
            print('=== FOLD %s ==='%i)
            try:
                if dppi:
                    cmd = 'python rp_ppi_classifier.py -train %s -test %s -r %s -n %s'%(train + '-%s.csv'%i, test + '-%s.csv'%i, results, name  + '-%s'%i)
                else:
                    cmd = 'python rp_ppi_classifier.py -train %s -test %s -r %s -n %s'%(train + '-%s.tsv'%i, test + '-%s.tsv'%i, results, name  + '-%s'%i)
                result = subprocess.run(cmd.split(), capture_output=True, text=True)
                print(result.stdout)
                print(result.stderr)
                time.sleep(1)
            except Exception as e:
                print(e)
    print('Time = %s seconds'%(round(time.time() - t_start, 2)))

def perform_rp_traintests_PM(dir_path, results, name):
    t_start = time.time()

    training = [i for i in os.listdir(dir_path) if 'train' in i]
    training.sort()
    testing = [i for i in os.listdir(dir_path) if 'test' in i]
    testing.sort()
    
    for i in range(0, len(training)):
        print('=== FOLD %s ==='%i)
        try:
            cmd = 'python rp_ppi_classifier.py -train %s -test %s -r %s -n %s'%(dir_path + training[i], dir_path + testing[i], results, name  + '-%s'%i)
            result = subprocess.run(cmd.split(), capture_output=True, text=True)
            print(result.stdout)
            print(result.stderr)
            time.sleep(1)
        except Exception as e:
            print(e)
    print('Time = %s seconds'%(round(time.time() - t_start, 2)))
    

def get_rp_cv_datasets(pred_dir, cv_dir, results_dir):
    t_start = time.time()
    pred_files = [f for f in os.listdir(pred_dir) if 'prediction' in f]
    pred_files.sort()
    
    cv_files = os.listdir(cv_dir)
    train_files = [f for f in cv_files if 'train' in f and 'test' not in f]
    train_files.sort()
    test_files = [f for f in cv_files if 'test' in f]
    test_files.sort()
    
    
    for i in range(0, len(pred_files)):
        print('=== FOLD %s ==='%i)
        try:
            # Train RP
            cmd = 'python extract_rp_features.py -l %s -p %s -r %s -m 0.75'%(cv_dir+train_files[i], pred_dir+pred_files[i], results_dir)
            result = subprocess.run(cmd.split(), capture_output=True, text=True)
            print(result.stdout)
            print(result.stderr)
            time.sleep(1)
        except Exception as e:
            print(e)
        try:
            # Test RP
            cmd = 'python extract_rp_features.py -l %s -p %s -r %s -m 0.75'%(cv_dir+test_files[i], pred_dir+pred_files[i], results_dir)
            result = subprocess.run(cmd.split(), capture_output=True, text=True)
            print(result.stdout)
            print(result.stderr)
            time.sleep(1)
        except Exception as e:
            print(e)
    print('Total time = %s seconds'%(round(time.time() - t_start, 2)))

# ==========================================================================            
def get_rp_features_park_marcotte(labels, all_predictions, results):
        try:
            cmd = 'python get_rp_features_subsets.py -l %s -p %s -r %s -m'%(labels, all_predictions, results)
            result = subprocess.run(cmd.split(), capture_output=True, text=True)
            print(result.stdout)
            print(result.stderr)
            time.sleep(1)
        except Exception as e:
            print(e)

# ===========================================================================
def get_cme_stacked_dataset(dir_path, results, name):
    to_cme = os.listdir(dir_path)
    try:
        to_cme.remove('DPPI')
        to_cme.remove('PIPR')
        to_cme.remove('CME')
        to_cme.remove('CME_AVG')
        to_cme.remove('CME_no_SPRINT')
        to_cme.remove('CME_no_DEEPFE')
        to_cme.remove('CME_no_PIPR')
        to_cme.remove('CME_no_DPPI')
    except ValueError:
        pass
    
    files_0 = os.listdir(dir_path + to_cme[0] + '/')
    files_0_train = [ f for f in files_0 if 'train' in f ]
    files_0_train.sort()
    files_0_test = [ f for f in files_0 if 'test' in f ]
    files_0_test.sort()
    
    files_1 = os.listdir(dir_path + to_cme[1] + '/')
    files_1_train = [ f for f in files_1 if 'train' in f ]
    files_1_train.sort()
    files_1_test = [ f for f in files_1 if 'test' in f ]
    files_1_test.sort()
    
    files_2 = os.listdir(dir_path + to_cme[2] + '/')
    files_2_train = [ f for f in files_2 if 'train' in f ]
    files_2_train.sort()
    files_2_test = [ f for f in files_2 if 'test' in f ]
    files_2_test.sort()

    files_3 = os.listdir(dir_path + to_cme[3] + '/')
    files_3_train = [ f for f in files_3 if 'train' in f ]
    files_3_train.sort()
    files_3_test = [ f for f in files_3 if 'test' in f ]
    files_3_test.sort()
    
    
    for i in range(0, len(files_0_train)):
        df_train_0 = pd.read_csv(dir_path + to_cme[0] + '/' + files_0_train[i], sep='\t')
        #df_train_0.fillna(value=0, inplace=True)
        #df_train_0.to_csv(dir_path + to_cme[0] + '/' + files_0_train[i], sep='\t', index=False)
        
        df_cme = df_train_0.copy()
        
        df_train_1 = pd.read_csv(dir_path + to_cme[1] + '/' + files_1_train[i], sep='\t')
        #df_train_1.fillna(value=0, inplace=True)
        #df_train_1.to_csv(dir_path + to_cme[1] + '/' + files_1_train[i], sep='\t', index=False)
        
        df_cme = df_cme.merge(df_train_1, on=[df_train_1.columns[0], df_train_1.columns[1]])
        df_cme.drop(columns=['label_x'], inplace=True)
        df_cme.rename(columns={'label_y': 'label'}, inplace=True)
        
        df_train_2 = pd.read_csv(dir_path + to_cme[2] + '/' + files_2_train[i], sep='\t')
        #df_train_2.fillna(value=0, inplace=True)
        #df_train_2.to_csv(dir_path + to_cme[2] + '/' + files_2_train[i], sep='\t', index=False)
        
        df_cme = df_cme.merge(df_train_2, on=[df_train_2.columns[0], df_train_2.columns[1]])
        df_cme.drop(columns=['label_x'], inplace=True)
        df_cme.rename(columns={'label_y': 'label'}, inplace=True)


        df_train_3 = pd.read_csv(dir_path + to_cme[3] + '/' + files_3_train[i], sep='\t')
        #df_train_3.fillna(value=0, inplace=True)
        #df_train_3.to_csv(dir_path + to_cme[3] + '/' + files_3_train[i], sep='\t', index=False)
        
        df_cme = df_cme.merge(df_train_3, on=[df_train_3.columns[0], df_train_3.columns[1]])
        df_cme.drop(columns=['label_x'], inplace=True)
        df_cme.rename(columns={'label_y': 'label'}, inplace=True)
        
        df_cme.to_csv(results + name + '_train-%s.tsv'%i, index=False, sep='\t')
        
        print('TRAIN')
        print(to_cme[0], df_train_0.isna().any().any())
        print(to_cme[1], df_train_1.isna().any().any())
        print(to_cme[2], df_train_2.isna().any().any())
        print(to_cme[3], df_train_3.isna().any().any())
        
        df_test_0 = pd.read_csv(dir_path + to_cme[0] + '/' + files_0_test[i], sep='\t')
        #df_test_0.fillna(value=0, inplace=True)
        #df_test_0.to_csv(dir_path + to_cme[0] + '/' + files_0_test[i], sep='\t', index=False)
        
        df_cme = df_test_0.copy()
        
        df_test_1 = pd.read_csv(dir_path + to_cme[1] + '/' + files_1_test[i], sep='\t')
        #df_test_1.fillna(value=0, inplace=True)
        #df_test_1.to_csv(dir_path + to_cme[1] + '/' + files_1_test[i], sep='\t', index=False)
        
        df_cme = df_cme.merge(df_test_1, on=[df_test_1.columns[0], df_test_1.columns[1]])
        df_cme.drop(columns=['label_x'], inplace=True)
        df_cme.rename(columns={'label_y': 'label'}, inplace=True)
        
        df_test_2 = pd.read_csv(dir_path + to_cme[2] + '/' + files_2_test[i], sep='\t')
        #df_test_2.fillna(value=0, inplace=True)
        #df_test_2.to_csv(dir_path + to_cme[2] + '/' + files_2_test[i], sep='\t', index=False)
        
        df_cme = df_cme.merge(df_test_2, on=[df_test_2.columns[0], df_test_2.columns[1]])
        df_cme.drop(columns=['label_x'], inplace=True)
        df_cme.rename(columns={'label_y': 'label'}, inplace=True)


        df_test_3 = pd.read_csv(dir_path + to_cme[3] + '/' + files_3_test[i], sep='\t')
        #df_test_3.fillna(value=0, inplace=True)
        #df_test_3.to_csv(dir_path + to_cme[3] + '/' + files_3_test[i], sep='\t', index=False)
        
        df_cme = df_cme.merge(df_test_3, on=[df_test_3.columns[0], df_test_3.columns[1]])
        df_cme.drop(columns=['label_x'], inplace=True)
        df_cme.rename(columns={'label_y': 'label'}, inplace=True)
        
        print('TEST')
        print(to_cme[0], df_test_0.isna().any().any())
        print(to_cme[1], df_test_1.isna().any().any())
        print(to_cme[2], df_test_2.isna().any().any())
        print(to_cme[3], df_test_3.isna().any().any())
        
        df_cme.to_csv(results + name + '_test-%s.tsv'%i, index=False, sep='\t')
        
# ================================================================================= #
def get_cme_PM_stacked_dataset(dir_path, results, name):
    
    # **** start with DEEPFE *****
    files_0 = os.listdir(dir_path)
    files_0_train = [ f for f in files_0 if 'train' in f ]
    files_0_train.sort()
    files_0_test = [ f for f in files_0 if 'test' in f ]
    files_0_test.sort()
    
    files_1 = os.listdir(dir_path.replace('DEEPFE', 'PIPR'))
    files_1_train = [ f for f in files_1 if 'train' in f ]
    files_1_train.sort()
    files_1_test = [ f for f in files_1 if 'test' in f ]
    files_1_test.sort()
    
    files_2 = os.listdir(dir_path.replace('DEEPFE', 'DPPI'))
    files_2_train = [ f for f in files_2 if 'train' in f ]
    files_2_train.sort()
    files_2_test = [ f for f in files_2 if 'test' in f ]
    files_2_test.sort()

    files_3 = os.listdir(dir_path.replace('DEEPFE', 'SPRINT'))
    files_3_train = [ f for f in files_3 if 'train' in f ]
    files_3_train.sort()
    files_3_test = [ f for f in files_3 if 'test' in f ]
    files_3_test.sort()
    
    
    for i in range(0, len(files_0_train)):
        df_train_0 = pd.read_csv(dir_path + files_0_train[i], sep='\t')
        #df_train_0.fillna(value=0, inplace=True)
        #df_train_0.to_csv(dir_path + to_cme[0] + '/' + files_0_train[i], sep='\t', index=False)
        
        df_cme = df_train_0.copy()
        
        df_train_1 = pd.read_csv(dir_path.replace('DEEPFE', 'PIPR') + files_1_train[i], sep='\t')
        #df_train_1.fillna(value=0, inplace=True)
        #df_train_1.to_csv(dir_path + to_cme[1] + '/' + files_1_train[i], sep='\t', index=False)
        
        df_cme = df_cme.merge(df_train_1, on=[df_train_1.columns[0], df_train_1.columns[1]])
        df_cme.drop(columns=['label_x'], inplace=True)
        df_cme.rename(columns={'label_y': 'label'}, inplace=True)
        
        df_train_2 = pd.read_csv(dir_path.replace('DEEPFE', 'DPPI') + files_2_train[i], sep='\t')
        #df_train_2.fillna(value=0, inplace=True)
        #df_train_2.to_csv(dir_path + to_cme[2] + '/' + files_2_train[i], sep='\t', index=False)
        
        df_cme = df_cme.merge(df_train_2, on=[df_train_2.columns[0], df_train_2.columns[1]])
        df_cme.drop(columns=['label_x'], inplace=True)
        df_cme.rename(columns={'label_y': 'label'}, inplace=True)


        df_train_3 = pd.read_csv(dir_path.replace('DEEPFE', 'SPRINT') + files_3_train[i], sep='\t')
        #df_train_3.fillna(value=0, inplace=True)
        #df_train_3.to_csv(dir_path + to_cme[3] + '/' + files_3_train[i], sep='\t', index=False)
        
        df_cme = df_cme.merge(df_train_3, on=[df_train_3.columns[0], df_train_3.columns[1]])
        df_cme.drop(columns=['label_x'], inplace=True)
        df_cme.rename(columns={'label_y': 'label'}, inplace=True)
        
        df_cme.replace(to_replace=np.nan, value=0, inplace=True)
        df_cme.to_csv(results + name + '_train-%s.tsv'%i, index=False, sep='\t')
        
        print('TRAIN')

        df_test_0 = pd.read_csv(dir_path + files_0_test[i], sep='\t')
        #df_test_0.fillna(value=0, inplace=True)
        #df_test_0.to_csv(dir_path + to_cme[0] + '/' + files_0_test[i], sep='\t', index=False)
        
        df_cme = df_test_0.copy()
        
        df_test_1 = pd.read_csv(dir_path.replace('DEEPFE', 'PIPR') + files_1_test[i], sep='\t')
        #df_test_1.fillna(value=0, inplace=True)
        #df_test_1.to_csv(dir_path + to_cme[1] + '/' + files_1_test[i], sep='\t', index=False)
        
        df_cme = df_cme.merge(df_test_1, on=[df_test_1.columns[0], df_test_1.columns[1]])
        df_cme.drop(columns=['label_x'], inplace=True)
        df_cme.rename(columns={'label_y': 'label'}, inplace=True)
        
        df_test_2 = pd.read_csv(dir_path.replace('DEEPFE', 'DPPI') + files_2_test[i], sep='\t')
        #df_test_2.fillna(value=0, inplace=True)
        #df_test_2.to_csv(dir_path + to_cme[2] + '/' + files_2_test[i], sep='\t', index=False)
        
        df_cme = df_cme.merge(df_test_2, on=[df_test_2.columns[0], df_test_2.columns[1]])
        df_cme.drop(columns=['label_x'], inplace=True)
        df_cme.rename(columns={'label_y': 'label'}, inplace=True)


        df_test_3 = pd.read_csv(dir_path.replace('DEEPFE', 'SPRINT') + files_3_test[i], sep='\t')
        #df_test_3.fillna(value=0, inplace=True)
        #df_test_3.to_csv(dir_path + to_cme[3] + '/' + files_3_test[i], sep='\t', index=False)
        
        df_cme = df_cme.merge(df_test_3, on=[df_test_3.columns[0], df_test_3.columns[1]])
        df_cme.drop(columns=['label_x'], inplace=True)
        df_cme.rename(columns={'label_y': 'label'}, inplace=True)
        
        print('TEST')

        df_cme.replace(to_replace=np.nan, value=0, inplace=True)
        df_cme.to_csv(results + name + '_test-%s.tsv'%i, index=False, sep='\t')

# ==========================================================================            
def get_predictions_average(dir_path, results, name):
    to_avg = os.listdir(dir_path)
    try:
        to_avg.remove('CME_AVG_RP')
        to_avg.remove('CME_STACKED')
        to_avg.remove('CME_SOFTVOTE_AVG')
        to_avg.remove('CME_no_SPRINT')
        to_avg.remove('CME_no_DEEPFE')
        to_avg.remove('CME_no_PIPR')
        to_avg.remove('CME_no_DPPI')
        #to_avg.remove('SPRINT')
        to_avg.remove('CME_SOFTVOTE_AVG_no_SPRINT')
        to_avg.remove('CME_SOFTVOTE_AVG_no_DPPI')
        to_avg.remove('CME_SOFTVOTE_AVG_no_DEEPFE')
        to_avg.remove('CME_SOFTVOTE_AVG_no_PIPR')
    except ValueError:
        pass
    
    files_0 = os.listdir(dir_path + to_avg[0] + '/')
    files_0_test = [ f for f in files_0 if 'test' in f or 'prediction' in f ]
    files_0_test.sort()
    files_1 = os.listdir(dir_path + to_avg[1] + '/')
    files_1_test = [ f for f in files_1 if 'test' in f or 'prediction' in f ]
    files_1_test.sort()
    files_2 = os.listdir(dir_path + to_avg[2] + '/')
    files_2_test = [ f for f in files_2 if 'test' in f or 'prediction' in f ]
    files_2_test.sort()
    files_3 = os.listdir(dir_path + to_avg[3] + '/')
    files_3_test = [ f for f in files_3 if 'test' in f or 'prediction' in f ]
    files_3_test.sort()

    for i in range(0, 10):
        #df_test_0 = pd.read_csv('/home/erixazerro/CUBIC/PPIP/RESULTS/DEEPFE_RESULTS/ECOLI_FULL/' + files_0_test[i], delim_whitespace=True, header=None)
        df_test_0 = pd.read_csv(dir_path + to_avg[0] + '/' + files_0_test[i], delim_whitespace=True, header=None)

        #df_test_1 = pd.read_csv('/home/erixazerro/CUBIC/PPIP/RESULTS/DPPI_RESULTS/ECOLI_FULL/' + files_1_test[i], delim_whitespace=True, header=None)
        df_test_1 = pd.read_csv(dir_path + to_avg[1] + '/' + files_1_test[i], delim_whitespace=True, header=None)

        #df_test_2 = pd.read_csv('/home/erixazerro/CUBIC/PPIP/RESULTS/PIPR_RESULTS/ECOLI_FULL/' + files_2_test[i], delim_whitespace=True, header=None)
        df_test_2 = pd.read_csv(dir_path + to_avg[2] + '/' + files_2_test[i], delim_whitespace=True, header=None)

        #df_test_3 = pd.read_csv('/home/erixazerro/CUBIC/PPIP/RESULTS/SPRINT_RESULTS/ECOLI_FULL/' + files_3_test[i], delim_whitespace=True, header=None)
        df_test_3 = pd.read_csv(dir_path + to_avg[3] + '/' + files_3_test[i], delim_whitespace=True, header=None)

        # Combine and average
        df_avg = df_test_0.merge(df_test_1, on=[0,1]).merge(df_test_2, on=[0,1]).merge(df_test_3, on=[0,1])
        df_avg.insert(df_avg.shape[1], 'mean', df_avg.mean(axis=1))
        
        #Save
        df_avg.to_csv(results + 'predictions_' + name + '_test-%s.tsv'%i, columns=[0,1,'mean'], header=False, index=False, sep='\t')
        
# ================================================================================= #

def get_cme_avg_dataset(dir_path, results, name):
    to_cme = os.listdir(dir_path)
    try:
        to_cme.remove('CME')
        to_cme.remove('CME_AVG')
        to_cme.remove('CME_no_SPRINT')
        to_cme.remove('CME_no_DEEPFE')
        to_cme.remove('CME_no_PIPR')
        to_cme.remove('CME_no_DPPI')
    except ValueError:
        pass
    
    files_0 = os.listdir(dir_path + to_cme[0] + '/')
    files_0_train = [ f for f in files_0 if 'train' in f ]
    files_0_train.sort()
    files_0_test = [ f for f in files_0 if 'test' in f ]
    files_0_test.sort()
    
    files_1 = os.listdir(dir_path + to_cme[1] + '/')
    files_1_train = [ f for f in files_1 if 'train' in f ]
    files_1_train.sort()
    files_1_test = [ f for f in files_1 if 'test' in f ]
    files_1_test.sort()
    
    files_2 = os.listdir(dir_path + to_cme[2] + '/')
    files_2_train = [ f for f in files_2 if 'train' in f ]
    files_2_train.sort()
    files_2_test = [ f for f in files_2 if 'test' in f ]
    files_2_test.sort()
    
    
    files_3 = os.listdir(dir_path + to_cme[3] + '/')
    files_3_train = [ f for f in files_3 if 'train' in f ]
    files_3_train.sort()
    files_3_test = [ f for f in files_3 if 'test' in f ]
    files_3_test.sort()
    
    for i in range(0, len(files_0_train)):
        df_train_0 = pd.read_csv(dir_path + to_cme[0] + '/' + files_0_train[i], sep='\t')
        #df_train_0.fillna(value=0, inplace=True)
        #df_train_0.to_csv(dir_path + to_cme[0] + '/' + files_0_train[i], sep='\t', index=False)
        
        df_cme = df_train_0.copy()
        
        df_train_1 = pd.read_csv(dir_path + to_cme[1] + '/' + files_1_train[i], sep='\t')
        #df_train_1.fillna(value=0, inplace=True)
        #df_train_1.to_csv(dir_path + to_cme[1] + '/' + files_1_train[i], sep='\t', index=False)
        
        df_cme = df_cme.merge(df_train_1, on=[df_train_1.columns[0], df_train_1.columns[1]])
        df_cme.drop(columns=['label_x'], inplace=True)
        df_cme.rename(columns={'label_y': 'label'}, inplace=True)
        
        df_train_2 = pd.read_csv(dir_path + to_cme[2] + '/' + files_2_train[i], sep='\t')
        #df_train_2.fillna(value=0, inplace=True)
        #df_train_2.to_csv(dir_path + to_cme[2] + '/' + files_2_train[i], sep='\t', index=False)
        
        df_cme = df_cme.merge(df_train_2, on=[df_train_2.columns[0], df_train_2.columns[1]])
        df_cme.drop(columns=['label_x'], inplace=True)
        df_cme.rename(columns={'label_y': 'label'}, inplace=True)
        
        
        df_train_3 = pd.read_csv(dir_path + to_cme[3] + '/' + files_3_train[i], sep='\t')
        #df_train_3.fillna(value=0, inplace=True)
        #df_train_3.to_csv(dir_path + to_cme[3] + '/' + files_3_train[i], sep='\t', index=False)
        
        df_cme = df_cme.merge(df_train_3, on=[df_train_3.columns[0], df_train_3.columns[1]])
        df_cme.drop(columns=['label_x'], inplace=True)
        df_cme.rename(columns={'label_y': 'label'}, inplace=True)
        
        df_cme_train_avg = pd.DataFrame(columns=df_train_0.columns)
        df_cme_train_avg['Protein_A'] = df_cme['Protein_A']
        df_cme_train_avg['Protein_B'] = df_cme['Protein_B']
        df_cme_train_avg['Rank_A_in_B'] = df_cme[['Rank_A_in_B_x', 'Rank_A_in_B_y']].mean(axis=1)
        df_cme_train_avg['Rank_B_in_A'] = df_cme[['Rank_B_in_A_x', 'Rank_B_in_A_y']].mean(axis=1)
        df_cme_train_avg['Score_A_in_B'] = df_cme[['Score_A_in_B_x', 'Score_A_in_B_y']].mean(axis=1)
        df_cme_train_avg['Score_B_in_A'] = df_cme[['Score_B_in_A_x', 'Score_B_in_A_y']].mean(axis=1)
        df_cme_train_avg['NaRRO'] = df_cme[['NaRRO_x', 'NaRRO_y']].mean(axis=1)
        df_cme_train_avg['ARRO'] = df_cme[['ARRO_x', 'ARRO_y']].mean(axis=1)
        df_cme_train_avg['NoRRO_A'] = df_cme[['NoRRO_A_x', 'NoRRO_A_y']].mean(axis=1)
        df_cme_train_avg['NoRRO_B'] = df_cme[['NoRRO_B_x', 'NoRRO_B_y']].mean(axis=1)
        df_cme_train_avg['NoRRO'] = df_cme[['NoRRO_x', 'NoRRO_y']].mean(axis=1)
        df_cme_train_avg['Rank_LocalCutoff_A_elbow'] = df_cme[['Rank_LocalCutoff_A_elbow_x', 'Rank_LocalCutoff_A_elbow_y']].mean(axis=1)
        df_cme_train_avg['Rank_LocalCutoff_B_elbow'] = df_cme[['Rank_LocalCutoff_B_elbow_x', 'Rank_LocalCutoff_B_elbow_y']].mean(axis=1)
        df_cme_train_avg['Score_LocalCutoff_A_elbow'] = df_cme[['Score_LocalCutoff_A_elbow_x', 'Score_LocalCutoff_A_elbow_y']].mean(axis=1)
        df_cme_train_avg['Score_LocalCutoff_B_elbow'] = df_cme[['Score_LocalCutoff_B_elbow_x', 'Score_LocalCutoff_B_elbow_y']].mean(axis=1)
        df_cme_train_avg['Rank_LocalCutoff_A_knee'] = df_cme[['Rank_LocalCutoff_A_knee_x', 'Rank_LocalCutoff_A_knee_y']].mean(axis=1)
        df_cme_train_avg['Rank_LocalCutoff_B_knee'] = df_cme[['Rank_LocalCutoff_B_knee_x', 'Rank_LocalCutoff_B_knee_y']].mean(axis=1)
        df_cme_train_avg['Score_LocalCutoff_A_knee'] = df_cme[['Score_LocalCutoff_A_knee_x', 'Score_LocalCutoff_A_knee_y']].mean(axis=1)
        df_cme_train_avg['Score_LocalCutoff_B_knee'] = df_cme[['Score_LocalCutoff_B_knee_x', 'Score_LocalCutoff_B_knee_y']].mean(axis=1)
        df_cme_train_avg['Rank_AB_AboveLocal_A_elbow'] = df_cme[['Rank_AB_AboveLocal_A_elbow_x', 'Rank_AB_AboveLocal_A_elbow_y']].mean(axis=1)
        df_cme_train_avg['Rank_BA_AboveLocal_B_elbow'] = df_cme[['Rank_BA_AboveLocal_B_elbow_x', 'Rank_BA_AboveLocal_B_elbow_y']].mean(axis=1)
        df_cme_train_avg['Rank_AB_AboveLocal_A_knee'] = df_cme[['Rank_AB_AboveLocal_A_knee_x', 'Rank_AB_AboveLocal_A_knee_y']].mean(axis=1)
        df_cme_train_avg['Rank_BA_AboveLocal_B_knee'] = df_cme[['Rank_BA_AboveLocal_B_knee_x', 'Rank_BA_AboveLocal_B_knee_y']].mean(axis=1)
        df_cme_train_avg['Above_Global_Mean'] = df_cme[['Above_Global_Mean_x', 'Above_Global_Mean_y']].mean(axis=1)
        df_cme_train_avg['Above_Global_Median'] = df_cme[['Above_Global_Median_x', 'Above_Global_Median_y']].mean(axis=1)
        df_cme_train_avg['FD_A_elbow'] = df_cme[['FD_A_elbow_x', 'FD_A_elbow_y']].mean(axis=1)
        df_cme_train_avg['FD_B_elbow'] = df_cme[['FD_B_elbow_x', 'FD_B_elbow_y']].mean(axis=1)
        df_cme_train_avg['FD_A_knee'] = df_cme[['FD_A_knee_x', 'FD_A_knee_y']].mean(axis=1)
        df_cme_train_avg['FD_B_knee'] = df_cme[['FD_B_knee_x', 'FD_B_knee_y']].mean(axis=1)
        df_cme_train_avg['label'] = df_cme['label']
        
        df_cme_train_avg.to_csv(results + name + '_train-%s.tsv'%i, index=False, sep='\t')
        
        #print('TRAIN')
        
        df_test_0 = pd.read_csv(dir_path + to_cme[0] + '/' + files_0_test[i], sep='\t')
        #df_test_0.fillna(value=0, inplace=True)
        #df_test_0.to_csv(dir_path + to_cme[0] + '/' + files_0_test[i], sep='\t', index=False)
        
        df_cme = df_test_0.copy()
        
        df_test_1 = pd.read_csv(dir_path + to_cme[1] + '/' + files_1_test[i], sep='\t')
        #df_test_1.fillna(value=0, inplace=True)
        #df_test_1.to_csv(dir_path + to_cme[1] + '/' + files_1_test[i], sep='\t', index=False)
        
        df_cme = df_cme.merge(df_test_1, on=[df_test_1.columns[0], df_test_1.columns[1]])
        df_cme.drop(columns=['label_x'], inplace=True)
        df_cme.rename(columns={'label_y': 'label'}, inplace=True)
        
        df_test_2 = pd.read_csv(dir_path + to_cme[2] + '/' + files_2_test[i], sep='\t')
        #df_test_2.fillna(value=0, inplace=True)
        #df_test_2.to_csv(dir_path + to_cme[2] + '/' + files_2_test[i], sep='\t', index=False)
        
        df_cme = df_cme.merge(df_test_2, on=[df_test_2.columns[0], df_test_2.columns[1]])
        df_cme.drop(columns=['label_x'], inplace=True)
        df_cme.rename(columns={'label_y': 'label'}, inplace=True)
        
        
        df_test_3 = pd.read_csv(dir_path + to_cme[3] + '/' + files_3_test[i], sep='\t')
        #df_test_3.fillna(value=0, inplace=True)
        #df_test_3.to_csv(dir_path + to_cme[3] + '/' + files_3_test[i], sep='\t', index=False)
        
        df_cme = df_cme.merge(df_test_3, on=[df_test_3.columns[0], df_test_3.columns[1]])
        df_cme.drop(columns=['label_x'], inplace=True)
        df_cme.rename(columns={'label_y': 'label'}, inplace=True)
        
        #print('TEST')

        df_cme_test_avg = pd.DataFrame(columns=df_test_0.columns)
        df_cme_test_avg['Protein_A'] = df_cme['Protein_A']
        df_cme_test_avg['Protein_B'] = df_cme['Protein_B']
        df_cme_test_avg['Rank_A_in_B'] = df_cme[['Rank_A_in_B_x', 'Rank_A_in_B_y']].mean(axis=1)
        df_cme_test_avg['Rank_B_in_A'] = df_cme[['Rank_B_in_A_x', 'Rank_B_in_A_y']].mean(axis=1)
        df_cme_test_avg['Score_A_in_B'] = df_cme[['Score_A_in_B_x', 'Score_A_in_B_y']].mean(axis=1)
        df_cme_test_avg['Score_B_in_A'] = df_cme[['Score_B_in_A_x', 'Score_B_in_A_y']].mean(axis=1)
        df_cme_test_avg['NaRRO'] = df_cme[['NaRRO_x', 'NaRRO_y']].mean(axis=1)
        df_cme_test_avg['ARRO'] = df_cme[['ARRO_x', 'ARRO_y']].mean(axis=1)
        df_cme_test_avg['NoRRO_A'] = df_cme[['NoRRO_A_x', 'NoRRO_A_y']].mean(axis=1)
        df_cme_test_avg['NoRRO_B'] = df_cme[['NoRRO_B_x', 'NoRRO_B_y']].mean(axis=1)
        df_cme_test_avg['NoRRO'] = df_cme[['NoRRO_x', 'NoRRO_y']].mean(axis=1)
        df_cme_test_avg['Rank_LocalCutoff_A_elbow'] = df_cme[['Rank_LocalCutoff_A_elbow_x', 'Rank_LocalCutoff_A_elbow_y']].mean(axis=1)
        df_cme_test_avg['Rank_LocalCutoff_B_elbow'] = df_cme[['Rank_LocalCutoff_B_elbow_x', 'Rank_LocalCutoff_B_elbow_y']].mean(axis=1)
        df_cme_test_avg['Score_LocalCutoff_A_elbow'] = df_cme[['Score_LocalCutoff_A_elbow_x', 'Score_LocalCutoff_A_elbow_y']].mean(axis=1)
        df_cme_test_avg['Score_LocalCutoff_B_elbow'] = df_cme[['Score_LocalCutoff_B_elbow_x', 'Score_LocalCutoff_B_elbow_y']].mean(axis=1)
        df_cme_test_avg['Rank_LocalCutoff_A_knee'] = df_cme[['Rank_LocalCutoff_A_knee_x', 'Rank_LocalCutoff_A_knee_y']].mean(axis=1)
        df_cme_test_avg['Rank_LocalCutoff_B_knee'] = df_cme[['Rank_LocalCutoff_B_knee_x', 'Rank_LocalCutoff_B_knee_y']].mean(axis=1)
        df_cme_test_avg['Score_LocalCutoff_A_knee'] = df_cme[['Score_LocalCutoff_A_knee_x', 'Score_LocalCutoff_A_knee_y']].mean(axis=1)
        df_cme_test_avg['Score_LocalCutoff_B_knee'] = df_cme[['Score_LocalCutoff_B_knee_x', 'Score_LocalCutoff_B_knee_y']].mean(axis=1)
        df_cme_test_avg['Rank_AB_AboveLocal_A_elbow'] = df_cme[['Rank_AB_AboveLocal_A_elbow_x', 'Rank_AB_AboveLocal_A_elbow_y']].mean(axis=1)
        df_cme_test_avg['Rank_BA_AboveLocal_B_elbow'] = df_cme[['Rank_BA_AboveLocal_B_elbow_x', 'Rank_BA_AboveLocal_B_elbow_y']].mean(axis=1)
        df_cme_test_avg['Rank_AB_AboveLocal_A_knee'] = df_cme[['Rank_AB_AboveLocal_A_knee_x', 'Rank_AB_AboveLocal_A_knee_y']].mean(axis=1)
        df_cme_test_avg['Rank_BA_AboveLocal_B_knee'] = df_cme[['Rank_BA_AboveLocal_B_knee_x', 'Rank_BA_AboveLocal_B_knee_y']].mean(axis=1)
        df_cme_test_avg['Above_Global_Mean'] = df_cme[['Above_Global_Mean_x', 'Above_Global_Mean_y']].mean(axis=1)
        df_cme_test_avg['Above_Global_Median'] = df_cme[['Above_Global_Median_x', 'Above_Global_Median_y']].mean(axis=1)
        df_cme_test_avg['FD_A_elbow'] = df_cme[['FD_A_elbow_x', 'FD_A_elbow_y']].mean(axis=1)
        df_cme_test_avg['FD_B_elbow'] = df_cme[['FD_B_elbow_x', 'FD_B_elbow_y']].mean(axis=1)
        df_cme_test_avg['FD_A_knee'] = df_cme[['FD_A_knee_x', 'FD_A_knee_y']].mean(axis=1)
        df_cme_test_avg['FD_B_knee'] = df_cme[['FD_B_knee_x', 'FD_B_knee_y']].mean(axis=1)
        df_cme_test_avg['label'] = df_cme['label']
        
        df_cme_test_avg.to_csv(results + name + '_test-%s.tsv'%i, index=False, sep='\t')
        
# ================================================================================

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

# ======================== GRIDSEARCHING =============================================================================
def gridsearch(train, test):
    # Grid-search params
    # LightGBM
    boosting_types = ['goss']
    learning_rates = [0.05, 0.1, 0.15]
    num_leaves = [40, 50, 60]
    n_estimators = [120, 150, 250]
    min_data_in_leafs = [40, 50, 60]
    max_depths = [7, 10, 15]
    smoothing = [0.05, 0.1, 0.15]
    l1s = [0, 0.01, 0.05]
    l2s = [0, 0.01, 0.05]
    gains = [0, 0.1]
    delta = 0.5
    #import lightgbm as lgb
    best_auPR = 0
    best = ''
    
    precs = []
    recs = []
    labs = []
    t_start = time.time()
    for s in smoothing:
        for l1 in l1s:
            for l2 in l2s:
                for g in gains:
                    for x in min_data_in_leafs:
                        for d in max_depths:
                            for n in n_estimators:
                                for boost in boosting_types:
                                    for rate in learning_rates:
                                        for leaves in num_leaves:
                                
                                            try:
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
                                                for i in range(0, 10):
                                                    print("=== FOLD %s ==="%i)
                                                    # Load Data
                                                    df_train = pd.read_csv(train + '-%s.tsv'%i, delim_whitespace=True)
                                                    df_test = pd.read_csv(test + '-%s.tsv'%i, delim_whitespace=True)
                                                    
                                                    # Define data and labels
                                                    #pairs_train = np.array(df_train[df_train.columns[0:2]])
                                                    X_train = np.array(df_train[df_train.columns[2:-1]])
                                                    y_train = np.array(df_train[df_train.columns[-1]])
                                                    pairs_test = np.array(df_test[df_test.columns[0:2]])
                                                    X_test = np.array(df_test[df_test.columns[2:-1]])
                                                    y_test = np.array(df_test[df_test.columns[-1]])
                                                    t_start = time.time()
                                                    
                                                    
                                                    '''
                                                    lgb_train = lgb.Dataset(X_train,label=y_train)
                                                    lgb_test = lgb.Dataset(X_test,label=y_test)
                                                    
                                                    params = {'metric' : 'binary_logloss',
                                                              'boosting_type' : 'dart',
                                                              'learning_rate': 0.05,
                                                              'num_leaves' : 100,
                                                              'max_depth' : 20,
                                                              'verbose' : -1,
                                                              # Reducing overfitting
                                                              'min_data_in_leaf': 300,
                                                              'feature_fraction':0.8,
                                                              'bagging_fraction':0.8,
                                                              'bagging_freq':10
                                                    }
                                                    eval_results = {}
                                                    lgbm = lgb.train(params,
                                                                     lgb_train,
                                                                     num_boost_round=200,
                                                                     valid_sets=[lgb_train, lgb_test],
                                                                     early_stopping_rounds= 20,
                                                                     #verbose_eval= 10,
                                                                     callbacks = [lgb.record_evaluation(eval_results)]
                                                                     )
                                                    plt.plot(eval_results['training']['binary_logloss'], color='blue', label='Training Data')
                                                    plt.plot(eval_results['valid_1']['binary_logloss'], color='orange', label='Validation Data')
                                                    plt.xlabel('Number of Boosting Rounds')
                                                    plt.ylabel('Binary LogLoss')
                                                    plt.title('')
                                                    plt.legend()
                                                    '''
                                                    
                                                    clf = LGBMClassifier(random_state=13052021, 
                                                                         boosting_type=boost, 
                                                                         learning_rate=rate, 
                                                                         num_leaves=leaves,
                                                                         max_depth=d, 
                                                                         min_data_in_leaf=x,
                                                                         n_estimators=n,
                                                                         #early_stopping_round=30,
                                                                         path_smooth=s,
                                                                         lambda_l1=l1,
                                                                         lambda_l2=l2,
                                                                         min_gain_to_split=g,
                                                                         )
                                                    '''
                                                    
                                                    
                                                    clf = xgb.XGBClassifier(random_state=13052021,
                                                                            booster='gbtree',
                                                                            use_label_encoder=False, 
                                                                            eval_metric=leaves,
                                                                            eta=rate
                                                                            )
                                                    '''
                                                    
                                                    '''
                                                    clf = RandomForestClassifier(random_state=13052021,
                                                                                 n_estimators=rate,
                                                                                 criterion=leaves
                                                                                 )
                                                    '''
                                                    #pipe = Pipeline([('scaler', StandardScaler()), ('rndforest', clf)])
                                                    #pipe = Pipeline([('scaler', StandardScaler()), ('xgboost', clf)])
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
                                                    
                                                    df_pred = ppi_probs.copy()
                                                    df_pred = get_matching_pairs(df_pred, df_test[['Protein_A', 'Protein_B','label']])
                                                    df_pred.drop(columns=[df_test.columns[0], df_test.columns[1]], inplace=True)
                                                    df_pred.rename(columns={3: 0, 'label': 1}, inplace=True)
                                                    df_pred_total = df_pred_total.append(df_pred)
                                                    
                                                    # Get performance metricss of binary evaluation (threshold=0.5)
                                                    tn, fp, fn, tp = metrics.confusion_matrix(y_test, pred).ravel()
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
                                                
                                                if df_pred_total[0].min() >= 0 and df_pred_total[0].max() <= 1:
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
                                                #performance = pd.DataFrame(data={'precision': pd.Series(fold_precision, dtype=float), 'recall': pd.Series(fold_recall, dtype=float), 'specificity': pd.Series(fold_specificity, dtype=float), 'f1': pd.Series(fold_f1, dtype=float), 'mcc': pd.Series(fold_mcc, dtype=float), 'auc_pr': pd.Series(np.fromiter(pr_aucs.values(), dtype=float), dtype=float), 'auc_roc': pd.Series(np.fromiter(roc_aucs.values(), dtype=float), dtype=float)}, dtype=float)
                                                #overall_curves = pd.DataFrame(data={'precision': pd.Series(precision, dtype=float), 'recall': pd.Series(recall, dtype=float), 'fpr': pd.Series(fpr, dtype=float), 'tpr': pd.Series(tpr, dtype=float)}, dtype=float)
                                                
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
                                                    
                                                if pr_auc > best_auPR:
                                                    best_auPR = pr_auc
                                                    best = "\n ====== BEST core params: boost=%s rate=%s leaves=%s n_estimators=%s min_data_in_leaf=%s max_depth=%s smooth=%s l1=%s l2=%s gain=%s ====== \n"%(boost, rate, leaves, n, x, d, s, l1, l2, g)
                                                    print(best)
                                                    print('\npr_auc=%.4f (+/- %.4f)' % (np.mean(np.fromiter(pr_aucs.values(), dtype=float)), np.std(np.fromiter(pr_aucs.values(), dtype=float))))
                                                    precs.append(precision)
                                                    recs.append(recall)
                                                    labs.append('AUC=%s boost=%s rate=%s leaves=%s trees=%s min_data=%s max_depth=%s smooth=%s l1=%s l2=%s gain=%s '%(round(best_auPR,4), boost, rate, leaves, n, x, d, s, l1, l2, g))
                                                    
                                                else:
                                                    print("Tested core params: boost %s rate %s leaves %s estimators %s min_data %s max_depth %s smooth %s l1 %s l2 %s gain %s\n"%(boost, rate, leaves, n, x, d, s, l1, l2, g))
                                            except Exception as e:
                                                print("\n *** Could not test core params: boost %s rate %s leaves %s estimators %s min_data %s max_dapth %s smooth %s l1 %s l2 %s gain %s *** \n"%(boost, rate, leaves, n, x, d, s, l1, l2, g))
                                                print(e)
                                                continue
    print('TIME = %s seconds'%(time.time() - t_start))
    
    for i in range(0, len(precs)):
        plt.plot(recs[i], precs[i], label=labs[i])
    plt.xlabel('Recall')
    plt.ylabel('Precision') 
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.title("Precision-Recall Curve")
    plt.legend(loc='best', handlelength=0, prop={'size': 6})

# Return training set,
# c1_test (both proteins in pairs are found in training set),
# c2_test (only 1 protein in pairs is found in training set),
# c3_test (no pairs contain proteins found in training set)
def park_marcotte_subsets(df, train_size=0.7):
    
    # Attempt 5 times to obtain most interactions possible in test set 3 due to randomization of train_test_split
    best_c1 = pd.DataFrame()
    best_c2 = pd.DataFrame()
    best_c3 = pd.DataFrame()
    best_train = pd.DataFrame()
    for t in range(0,5):
        # Create stratified train/test split
        pos = df[df[df.columns[-1]] == 1]
        neg = df[df[df.columns[-1]] == 0]
        train_pos, test_pos = train_test_split(pos, train_size=train_size)
        train_neg, test_neg = train_test_split(neg, train_size=train_size)
        train = train_pos.append(train_neg)
        train.reset_index(drop=True, inplace=True)
        test = test_pos.append(test_neg)
        test.reset_index(drop=True, inplace=True)
        
        # Get proteins
        train_proteins = pd.DataFrame(train[0].append(train[1]).unique())
        test_proteins = pd.DataFrame(test[0].append(test[1]).unique())
        # Get proteins found in both train and test
        proteins_both = test_proteins[test_proteins[0].isin(train_proteins[0])]
    
        # Create c1, c2, c3 sets
        c1 = test[(test[0].isin(proteins_both[0])) & (test[1].isin(proteins_both[0]))]
        c2 = test[(test[0].isin(proteins_both[0])) ^ (test[1].isin(proteins_both[0]))]
        c3 = test[(~test[0].isin(proteins_both[0])) & (~test[1].isin(proteins_both[0]))]

        if c3.shape[0] > best_c3.shape[0]:
            best_c1 = c1.copy()
            best_c2 = c2.copy()
            best_c3 = c3.copy()
            best_train = train.copy()
        
    if best_c1.empty:
        print('No c1 test set')
    else:
        best_c1 = balance_pm_test_set(best_train, best_c1, 1)
    if best_c2.empty:
        print('No c2 test set')
    else:
        best_c2 = balance_pm_test_set(best_train, best_c2, 2)
    if best_c3.empty:
        print('No c3 test set')
    else:
        best_c3 = balance_pm_test_set(best_train, best_c3, 3)
        
    return best_train, best_c1, best_c2, best_c3

def balance_pm_test_set(train, test, c_set):
    df_train = train.copy()
    df_test = test.copy()
    # Return if already balanced
    if len(df_test.value_counts(subset=[df_test.columns[-1]]).unique()) == 1 and len(df_test[df_test.columns[-1]].unique()) == 2:
        return df_test
    
    # Get all proteins for sampling
    train_proteins = df_train[df_train.columns[0]].append(df_train[df_train.columns[1]]).unique()
    test_proteins = df_test[df_test.columns[0]].append(df_test[df_test.columns[1]]).unique()
    
    if len(test_proteins) < 2:
        print('\tUnable to balance set')
        return df_test
    test_pos = df_test[df_test[df_test.columns[-1]] == 1].reset_index(drop=True)
    test_neg = df_test[df_test[df_test.columns[-1]] == 0].reset_index(drop=True)
    
    # Return balanced data if more negatives than positives
    if test_pos.shape[0] < test_neg.shape[0]:
        test_neg = test_neg[0:test_pos.shape[0]]
        df_test_balanced = test_pos.append(test_neg)
        df_test_balanced.reset_index(drop=True, inplace=True)
        return df_test_balanced
    
    # Max combinations possible
    max_combos = len(test_proteins) + (math.factorial(len(test_proteins))/(2*(math.factorial(len(test_proteins) - 2))))
    # If unable to generate enough combos to balance dataset, return
    if max_combos - df_test.shape[0] < abs(test_pos.shape[0] - test_neg.shape[0]):
        print('Not enough proteins to generate negatives and balance data.')
        return df_test
    
    print('\tGenerating negatives for c%s'%c_set)
    df_neg = test_neg.copy()
    generator = np.random.default_rng()
    while (df_neg.shape[0] < test_pos.shape[0]):
        # Generate random pairs
        df_neg = df_neg.append(pd.DataFrame(generator.choice(test_proteins, size=test_pos[test_pos.columns[:2]].shape)), ignore_index=True)
        # Remove redundant and sort AB order of PPI pairs
        df_neg = remove_redundant_pairs(df_neg)
        df_neg_rev = pd.DataFrame({0: df_neg[1], 1: df_neg[0]})
        
        # Get pairs found in existing train and test PPIs and remove from negatives
        in_sets = df_test.append(df_train).merge(df_neg)
        in_sets_rev = df_test.append(df_train).merge(df_neg_rev)
        in_sets_rev = pd.DataFrame({0: in_sets_rev[1], 1: in_sets_rev[0]})
        in_sets = in_sets.append(in_sets_rev)
        df_neg = df_neg.append(in_sets).drop_duplicates(keep=False)
        df_neg[df_neg.columns[-1]] = 0
        
        if c_set == 1:
            df_neg = df_neg[(df_neg[0].isin(train_proteins)) & (df_neg[1].isin(train_proteins))]
        elif c_set == 2:
            df_neg = df_neg[(df_neg[0].isin(train_proteins)) ^ (df_neg[1].isin(train_proteins))]
        elif c_set == 3:
            df_neg = df_neg[(~df_neg[0].isin(train_proteins)) & (~df_neg[1].isin(train_proteins))]
    
    # Trim negatives if larger than positives
    if df_neg.shape[0] > (test_pos.shape[0] - test_neg.shape[0]):
        df_neg = df_neg[0:(test_pos.shape[0] - test_neg.shape[0])]
        
    df_test_balanced = df_test.append(df_neg)
    df_test_balanced.reset_index(drop=True, inplace=True)
    
    return df_test_balanced


lgb = LGBMClassifier(random_state=13052021, 
                    boosting_type='goss', 
                    learning_rate=0.1, 
                    num_leaves=50,
                    max_depth=10, 
                    min_data_in_leaf=50,
                    n_estimators=150,
                    #early_stopping_round=30,
                    path_smooth=0.1,
                    #lambda_l1=l1,
                    #lambda_l2=l2,
                    #min_gain_to_split=g,
                    )
knn = KNeighborsClassifier(n_neighbors=10,
                        weights='uniform',
                        algorithm='auto',
                        leaf_size=30,
                        p=2,
                        )

svc = SVC(C=1.0,
          kernel='rbf',
          degree=3,
          gamma='scale',
          probability=True,
          random_state=13052021,
          )

lsvc = SVC(C=1.0,
          kernel='linear',
          degree=3,
          gamma='scale',
          probability=True,
          random_state=13052021,
          )

rf = RandomForestClassifier(random_state=13052021,
                            n_estimators=100,
                            criterion='gini',
                            )
gbc = GradientBoostingClassifier(loss='deviance',
                                 learning_rate=0.1,
                                 n_estimators=100,
                                 random_state=13052021,
                                 )

xg = xgb.XGBClassifier(random_state=13052021,
                       booster='gbtree',
                       use_label_encoder=False,
                       eta=0.3,
                       )

def run_classifier(train, test, delta=0.5, clf=None):
    t_start = time.time()
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
    for i in range(0, 10):
        print("=== FOLD %s ==="%i)
        # Load Data
        df_train = pd.read_csv(train + '-%s.tsv'%i, delim_whitespace=True)
        df_train.replace(to_replace=np.nan, value=0, inplace=True)
        df_test = pd.read_csv(test + '-%s.tsv'%i, delim_whitespace=True)
        df_test.replace(to_replace=np.nan, value=0, inplace=True)
        
        # Define data and labels
        #pairs_train = np.array(df_train[df_train.columns[0:2]])
        X_train = np.array(df_train[df_train.columns[2:-1]])
        y_train = np.array(df_train[df_train.columns[-1]])
        pairs_test = np.array(df_test[df_test.columns[0:2]])
        X_test = np.array(df_test[df_test.columns[2:-1]])
        y_test = np.array(df_test[df_test.columns[-1]])
        '''
        if clf == None:
            clf = LGBMClassifier(random_state=13052021, 
                                 boosting_type='goss', 
                                 learning_rate=0.15, 
                                 num_leaves=40,
                                 #max_depth=d, 
                                 #min_data_in_leaf=x,
                                 #n_estimators=n,
                                 #early_stopping_round=30,
                                 #path_smooth=s,
                                 #lambda_l1=l1,
                                 #lambda_l2=l2,
                                 #min_gain_to_split=g,
                                 )
        '''
        
        clf = SVC(C=0.6,
                  kernel='sigmoid',
                  #kernel='rbf',
                  gamma='scale',
                  probability=True,
                  random_state=13052021,
                  )
        '''
        clf = RandomForestClassifier(random_state=13052021,
                            n_estimators=100,
                            criterion='gini',
                            )
        
        '''
        '''
        clf = LGBMClassifier(random_state=13052021, 
                    boosting_type='goss', 
                    learning_rate=0.1, 
                    num_leaves=50,
                    max_depth=10, 
                    min_data_in_leaf=50,
                    n_estimators=150,
                    #early_stopping_round=30,
                    path_smooth=0.1,
                    #lambda_l1=l1,
                    #lambda_l2=l2,
                    #min_gain_to_split=g,
                    )
        '''
        '''
        clf = MLPClassifier(random_state=13052021,
                            hidden_layer_sizes=(100,), 
                            activation='relu',
                            solver='sgd', 
                            alpha=0.0001,
                            batch_size='auto', 
                            learning_rate='constant', 
                            learning_rate_init=0.001, 
                            power_t=0.5, 
                            max_iter=200, 
                            shuffle=True,
                            tol=0.0001,
                            momentum=0.9,
                            nesterovs_momentum=True,
                            validation_fraction=0.1,
                            beta_1=0.9, 
                            beta_2=0.999, 
                            epsilon=1e-08, 
                            early_stopping=True, 
                            n_iter_no_change=10, 
                            max_fun=15000,
                            verbose=True)
        '''
        pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
        #pipe = Pipeline([('scaler', MinMaxScaler()), ('clf', clf)])
        #pipe = Pipeline([('clf', clf)])
        
        '''
        X_train_scaled = StandardScaler().fit_transform(X_train)
        X_test_scaled = StandardScaler().fit_transform(X_test)
        clf.fit(X_train_scaled, y_train)
        plt.plot(clf.loss_curve_, label='Loss_curve')
        plt.plot(clf.validation_scores_, label='Validation_scores')
        plt.xlabel('Epochs')
        plt.legend()
        '''
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
        
        df_pred = ppi_probs.copy()
        df_pred = get_matching_pairs(df_pred, df_test[['Protein_A', 'Protein_B','label']])
        df_pred.drop(columns=[df_test.columns[0], df_test.columns[1]], inplace=True)
        df_pred.rename(columns={3: 0, 'label': 1}, inplace=True)
        df_pred_total = df_pred_total.append(df_pred)
        
        # Get performance metricss of binary evaluation (threshold=0.5)
        tn, fp, fn, tp = metrics.confusion_matrix(y_test, pred).ravel()
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
    
    if df_pred_total[0].min() >= 0 and df_pred_total[0].max() <= 1:
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
    #performance = pd.DataFrame(data={'precision': pd.Series(fold_precision, dtype=float), 'recall': pd.Series(fold_recall, dtype=float), 'specificity': pd.Series(fold_specificity, dtype=float), 'f1': pd.Series(fold_f1, dtype=float), 'mcc': pd.Series(fold_mcc, dtype=float), 'auc_pr': pd.Series(np.fromiter(pr_aucs.values(), dtype=float), dtype=float), 'auc_roc': pd.Series(np.fromiter(roc_aucs.values(), dtype=float), dtype=float)}, dtype=float)
    #overall_curves = pd.DataFrame(data={'precision': pd.Series(precision, dtype=float), 'recall': pd.Series(recall, dtype=float), 'fpr': pd.Series(fpr, dtype=float), 'tpr': pd.Series(tpr, dtype=float)}, dtype=float)
    
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

    print('TIME = %s seconds'%(time.time() - t_start))
    
    plt.figure
    plt.plot(recall, precision, color='black', label='AUC_PR = %0.4f +/- %0.4f' % (pr_auc, np.std(np.fromiter(pr_aucs.values(), dtype=float))))
    plt.fill_between(recall, precision - df_interp_precisions['std'], precision + df_interp_precisions['std'], facecolor='blue', alpha=0.25)
    plt.fill_between(recall, precision - 2*df_interp_precisions['std'], precision + 2*df_interp_precisions['std'], facecolor='blue', alpha=0.25)
    plt.xlabel('Recall')
    plt.ylabel('Precision') 
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.title("Precision-Recall Curve - %s"%str(clf.__class__).split('.')[-1].replace("'>", ""))
    plt.legend(handlelength=0)
    plt.show()
    plt.close()
    if delta == 0.5:
        plt.figure
        plt.plot(fpr, tpr, color='black', label='AUC_ROC = %0.4f +/- %0.4f' % (roc_auc, np.std(np.fromiter(roc_aucs.values(), dtype=float))))
        plt.fill_between(fpr, tpr - df_interp_tprs['std'], tpr + df_interp_tprs['std'], facecolor='blue', alpha=0.25)
        plt.fill_between(fpr, tpr - 2*df_interp_tprs['std'], tpr + 2*df_interp_tprs['std'], facecolor='blue', alpha=0.25)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.title("ROC Curve - %s"%str(clf.__class__).split('.')[-1].replace("'>", ""))
        plt.legend(handlelength=0)
        plt.show()
        plt.close()
    
    return clf, precision, recall, thresholds

# PLOTTING LGBM FEATURE IMPORTANCE
clf = LGBMClassifier(random_state=13052021,
                    boosting_type='goss', 
                    learning_rate=0.1, 
                    num_leaves=50,
                    max_depth=10, 
                    min_data_in_leaf=50,
                    n_estimators=150,
                    path_smooth=0.1,
                    )
df_train = pd.read_csv('RP/DATASETS/TOTAL/ECOLI_FULL/RP_CME_biogrid_Ecoli_interactions.tsv', delim_whitespace=True)
df_train.replace(to_replace=np.nan, value=0, inplace=True)
pairs_train = np.array(df_train[df_train.columns[0:2]])
X_train = np.array(df_train[df_train.columns[2:-1]])
y_train = np.array(df_train[df_train.columns[-1]])

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

clf.fit(X_train_scaled, y_train, feature_name=df_train.columns[2:-1].tolist())
plot_importance(clf, title='RP-CME Feature Importance', ignore_zero=True, importance_type='split', xlabel='Number of Times Used to Build Model')
plot_split_value_histogram(clf, 'Score_A_in_B')

def compile_preds(train, test, probability=0.8):
    
    # Load data
    df_train = pd.read_csv(train, delim_whitespace=True)
    df_train.replace(to_replace=np.nan, value=0, inplace=True)
    df_test = pd.read_csv(test, delim_whitespace=True)
    df_test.replace(to_replace=np.nan, value=0, inplace=True)
    
    # Define data and labels
    #pairs_train = np.array(df_train[df_train.columns[0:2]])
    X_train = np.array(df_train[df_train.columns[2:-1]])
    y_train = np.array(df_train[df_train.columns[-1]])
    pairs_test = np.array(df_test[df_test.columns[0:2]])
    X_test = np.array(df_test[df_test.columns[2:-1]])
    #y_test = np.array(df_test[df_test.columns[-1]])
    
    # No random_state
    df_preds = pd.DataFrame()
    for i in range(0, 10):
        clf = LGBMClassifier(
                            boosting_type='goss', 
                            learning_rate=0.1, 
                            num_leaves=50,
                            max_depth=10, 
                            min_data_in_leaf=50,
                            n_estimators=150,
                            path_smooth=0.1,
                            )
        pipe = Pipeline([('scaler', StandardScaler()), ('metaclf', clf)])
        pipe.fit(X_train, y_train)
        
        pred_probs = pipe.predict_proba(X_test)
        ppi_probs = pd.DataFrame(np.append(pairs_test, pred_probs, axis=1), columns=[df_test.columns[0], df_test.columns[1], 2, 3])
        ppi_probs.drop(columns=[2], inplace=True)
        
        df_preds = df_preds.append(ppi_probs[ppi_probs[ppi_probs.columns[-1]] >= probability])
        df_preds.sort_values(by=df_preds.columns[-1], ascending=False, inplace=True)
        df_preds.reset_index(drop=True, inplace=True)
'''
# PLOTTING THRESHOLD VS. PRECISON
plt.plot(threshold_05, precision_05[:len(threshold_05)], label='ratio = 1:1')
plt.plot(threshold_01, precision_01[:len(threshold_01)], label='ratio = 1:10')
plt.plot(threshold_001, precision_001[:len(threshold_001)], label='ratio = 1:100')
plt.plot(threshold_0001, precision_0001[:len(threshold_0001)], label='ratio = 1:1000')
plt.grid(b=True)
#plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('Threshold')
plt.ylabel('Precision')
plt.title('Precision vs. Threshold - %s'%title)
plt.legend(prop={'size': 8})

plt.plot(threshold_05, recall_05[:len(threshold_05)]) #, label='ratio = 1:1')
#plt.plot(threshold_01, recall_01[:len(threshold_01)], label='ratio = 1:10')
#plt.plot(threshold_001, recall_001[:len(threshold_001)], label='ratio = 1:100')
#plt.plot(threshold_0001, recall_0001[:len(threshold_0001)], label='ratio = 1:1000')
plt.grid(b=True)
#plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('Threshold')
plt.ylabel('Recall')
plt.title('Recall vs. Threshold - %s'%title)
plt.legend(prop={'size': 8})
'''

def get_top_interactors(preds, known_positives, threshold=0.9):
    df = preds.copy()
    df_top = df[df[2] >= threshold]
    df_top_new = df_top.merge(known_positives, on=[0,1], how='outer')
    df_top_new = df_top_new[df_top_new['2_y'].isna()]
    df_top_new.drop(columns=['2_y'], inplace=True)
    df_top_new.rename(columns={'2_x': 2}, inplace=True)
    df_top_new.reset_index(drop=True, inplace=True)
    return df_top_new

def merge_top_interactors(tops):
    df_top = tops[0]
    for t in tops[1:]:
        df_top = df_top.merge(t, on=[0,1])
    return df_top
'''
plt.hist(x=wong[2])
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.text(0.1, 30000,'%s <= 0.185544'%(wong[wong[2] < 0.2].shape[0]))
plt.text(0.7, 22000,'%s >= 0.8605'%(wong[wong[2] > 0.8].shape[0]))
plt.title('Distribution of Prediction Scores')
'''
'''
# Get data for RP
rp_sprint = RP_AB(sprint, labels, top.iloc[0][0], top.iloc[0][1])
rp_dppi = RP_AB(dppi, labels, top.iloc[0][0], top.iloc[0][1])
rp_deepfe = RP_AB(deepfe, labels, top.iloc[0][0], top.iloc[0][1])
rp_pipr = RP_AB(pipr, labels, top.iloc[0][0], top.iloc[0][1])

def plot_all_rp(sprint, dppi, deepfe, pipr, proteinA, proteinB, save_loc = None):
    # Get data for RP
    rp_sprint = RP_AB(sprint, labels, proteinA, proteinB)
    rp_dppi = RP_AB(dppi, labels, proteinA, proteinB)
    rp_deepfe = RP_AB(deepfe, labels, proteinA, proteinB)
    rp_pipr = RP_AB(pipr, labels, proteinA, proteinB)

    # Plot
    fig, axs = plt.subplots(2,2)
    fig.suptitle('%s - to - All\n(%s circled)'%(proteinA, proteinB))
    axs[0, 0].plot(rp_sprint.ProteinA.knee.x, rp_sprint.ProteinA.knee.y)
    axs[0, 0].vlines(rp_sprint.ProteinA.elbow.knee,  min(rp_sprint.ProteinA.elbow.y), max(rp_sprint.ProteinA.elbow.y), linestyle='--', color='m', alpha=0.5)
    axs[0, 0].vlines(rp_sprint.ProteinA.knee.knee,  min(rp_sprint.ProteinA.knee.y), max(rp_sprint.ProteinA.knee.y), linestyle='--', color='c', alpha=0.5)
    #axs[0, 0].plot(rp_sprint.ProteinB.knee.x, rp_sprint.ProteinB.knee.y)
    ppi_A = rp_sprint.ProteinA.get_ppi(rp_sprint.ProteinB.ID)
    if not ppi_A.empty:
        # Check if PPI has label and assign color for plotting
        colour = 'grey'
        if rp_sprint.ProteinB.ID in rp_sprint.ProteinA.labels.positive.values.flatten():
            colour = 'lime'
        elif rp_sprint.ProteinB.ID in rp_sprint.ProteinA.labels.negative.values.flatten():
            colour = 'red'
        # Plot ProteinB on ProteinA-to-All
        ppi_y = ppi_A.iloc[0].iloc[-1]
        ppi_x = np.where(rp_sprint.ProteinA.knee.y == ppi_y)[0][0]
        axs[0, 0].scatter(ppi_x, ppi_y, facecolors='none', edgecolors=colour, linewidth=2, label=rp_sprint.ProteinB.ID)
    axs[0, 0].set_title('SPRINT')
    
    axs[0, 1].plot(rp_dppi.ProteinA.knee.x, rp_dppi.ProteinA.knee.y)
    axs[0, 1].vlines(rp_dppi.ProteinA.elbow.knee,  min(rp_dppi.ProteinA.elbow.y), max(rp_dppi.ProteinA.elbow.y), linestyle='--', color='m', alpha=0.5)
    axs[0, 1].vlines(rp_dppi.ProteinA.knee.knee,  min(rp_dppi.ProteinA.knee.y), max(rp_dppi.ProteinA.knee.y), linestyle='--', color='c', alpha=0.5)
    ppi_A = rp_dppi.ProteinA.get_ppi(rp_dppi.ProteinB.ID)
    if not ppi_A.empty:
        # Check if PPI has label and assign color for plotting
        colour = 'grey'
        if rp_dppi.ProteinB.ID in rp_dppi.ProteinA.labels.positive.values.flatten():
            colour = 'lime'
        elif rp_dppi.ProteinB.ID in rp_dppi.ProteinA.labels.negative.values.flatten():
            colour = 'red'
        # Plot ProteinB on ProteinA-to-All
        ppi_y = ppi_A.iloc[0].iloc[-1]
        ppi_x = np.where(rp_dppi.ProteinA.knee.y == ppi_y)[0][0]
        axs[0, 1].scatter(ppi_x, ppi_y, facecolors='none', edgecolors=colour, linewidth=2, label=rp_dppi.ProteinB.ID)
    #axs[0, 1].plot(rp_dppi.ProteinB.knee.x, rp_dppi.ProteinB.knee.y)
    axs[0, 1].set_title('DPPI')
    
    axs[1, 0].plot(rp_deepfe.ProteinA.knee.x, rp_deepfe.ProteinA.knee.y)
    axs[1, 0].vlines(rp_deepfe.ProteinA.elbow.knee,  min(rp_deepfe.ProteinA.elbow.y), max(rp_deepfe.ProteinA.elbow.y), linestyle='--', color='m', alpha=0.5)
    axs[1, 0].vlines(rp_deepfe.ProteinA.knee.knee,  min(rp_deepfe.ProteinA.knee.y), max(rp_deepfe.ProteinA.knee.y), linestyle='--', color='c', alpha=0.5)
    ppi_A = rp_deepfe.ProteinA.get_ppi(rp_deepfe.ProteinB.ID)
    if not ppi_A.empty:
        # Check if PPI has label and assign color for plotting
        colour = 'grey'
        if rp_deepfe.ProteinB.ID in rp_deepfe.ProteinA.labels.positive.values.flatten():
            colour = 'lime'
        elif rp_deepfe.ProteinB.ID in rp_deepfe.ProteinA.labels.negative.values.flatten():
            colour = 'red'
        # Plot ProteinB on ProteinA-to-All
        ppi_y = ppi_A.iloc[0].iloc[-1]
        ppi_x = np.where(rp_deepfe.ProteinA.knee.y == ppi_y)[0][0]
        axs[1, 0].scatter(ppi_x, ppi_y, facecolors='none', edgecolors=colour, linewidth=2, label=rp_deepfe.ProteinB.ID)
    #axs[1, 0].plot(rp_deepfe.ProteinB.knee.x, rp_deepfe.ProteinB.knee.y)
    axs[1, 0].set_title('DEEPFE')
    
    axs[1, 1].plot(rp_pipr.ProteinA.knee.x, rp_pipr.ProteinA.knee.y)
    axs[1, 1].vlines(rp_pipr.ProteinA.elbow.knee,  min(rp_pipr.ProteinA.elbow.y), max(rp_pipr.ProteinA.elbow.y), linestyle='--', color='m', alpha=0.5)
    axs[1, 1].vlines(rp_pipr.ProteinA.knee.knee,  min(rp_pipr.ProteinA.knee.y), max(rp_pipr.ProteinA.knee.y), linestyle='--', color='c', alpha=0.5)
    ppi_A = rp_pipr.ProteinA.get_ppi(rp_pipr.ProteinB.ID)
    if not ppi_A.empty:
        # Check if PPI has label and assign color for plotting
        colour = 'grey'
        if rp_pipr.ProteinB.ID in rp_pipr.ProteinA.labels.positive.values.flatten():
            colour = 'lime'
        elif rp_pipr.ProteinB.ID in rp_pipr.ProteinA.labels.negative.values.flatten():
            colour = 'red'
        # Plot ProteinB on ProteinA-to-All
        ppi_y = ppi_A.iloc[0].iloc[-1]
        ppi_x = np.where(rp_pipr.ProteinA.knee.y == ppi_y)[0][0]
        axs[1, 1].scatter(ppi_x, ppi_y, facecolors='none', edgecolors=colour, linewidth=2, label=rp_pipr.ProteinB.ID)
    #axs[1, 1].plot(rp_pipr.ProteinB.knee.x, rp_pipr.ProteinB.knee.y)
    axs[1, 1].set_title('PIPR')
    
    for ax in axs.flat:
        ax.set(xlabel='Rank', ylabel='Score')
    #    ax.label_outer()
    fig.tight_layout()
    
    if save_loc != None:
        fig.savefig(save_loc + proteinA + '_' + proteinB + '.png')
'''