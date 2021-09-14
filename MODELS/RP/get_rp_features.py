#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description:
    Extracts Reciprocal Perspective (RP) features from all-to-all PPI predictions.
    RP objects can be created given an all-to-all .tsv file and labelled PPI .tsv file.
    
    RP_AB object supports:
        Plotting one-to-all curves for a given protein ID.
        Plotting RP one-to-all curves for a given pair of protein IDs.
    
    Input arguements:
        -l: path to labeled dataset to convert to RP dataset (.tsv)
        -p: path to directory containing all-to-all PPI prediction files
        -t: path to directory containing training PPI files (.tsv) (from cross-validations)
        -r: path to directory to save RP feature dataset
    
    Output files:
        A single .tsv file of RP features for each PPI found in given labelled pairs.
        
        NOT SUPPORTED YET:
        If given a protein ID:
            A single .png of the one-to-all curve for that protein ID.
        If given a pair of Protein IDs:
            A single .png of the RP one-to-all curve for those protein IDs.
            
    Note:
        Arguments must be input such that the index of each PPI prediction file 
        matches the index of each labelled PPI file.
        
        e.g. 
        python get_rp_features.py -l labels.tsv -p PREDICTIONS/ -t CV_TRAINED/ -r RESULTS/
        
    
@author: Eric Arezza
Last Updated: May 11, 2021
"""

import os, argparse
import numpy as np
import pandas as pd
from kneed import KneeLocator
import matplotlib.pyplot as plt
import time

describe_help = 'python get_rp_features.py -l labels.tsv -p PREDICTIONS/ -t CV_TRAINED/ -r RESULTS/'
parser = argparse.ArgumentParser(description=describe_help)
parser.add_argument('-l', '--labels', help='Path to labeled PPIs file to convert to RP dataset (.tsv)', type=str)
parser.add_argument('-p', '--predictions', help='Path to directory with all-to-all PPI prediction files', type=str)
parser.add_argument('-t', '--trained', help='Path to directory with training files used to get all-to-all predictions (for cross-validations)', type=str)
parser.add_argument('-r', '--results', help='Path to directory to save new RP dataset', type=str, default=os.getcwd()+'/')
args = parser.parse_args()


class Labels(object):
    # For sifting through labelled PPI data
    def __init__(self, df_labels):
        self.labels = df_labels.copy()
        self.positive = self.labels[self.labels[self.labels.columns[-1]] == 1].reset_index(drop=True)
        self.negative = self.labels[self.labels[self.labels.columns[-1]] == 0].reset_index(drop=True)
    def get_ppi(self, protein):
        return get_protein_ppi(self.labels, protein)
    
class OneToAll(object):
    def __init__(self, df_scores, df_labels, proteinID, sens=5, deg=7, on=True):
        self.scores = df_scores.copy()
        self.labels = Labels(df_labels.copy())
        self.ID = proteinID
        self.sensitivity=sens
        self.degree=deg
        self.online=on
        self.proteins = self.scores[self.scores.columns[0]].append(self.scores[self.scores.columns[1]]).unique()
        
        # Local metrics
        self.scores_mean = np.mean(self.scores[self.scores.columns[-1]])
        self.scores_median = np.median(self.scores[self.scores.columns[-1]])
        self.scores_std = np.std(self.scores[self.scores.columns[-1]])
        
        self.ranks = self.scores[self.scores.columns[-1]].rank(ascending=False)
        self.percentiles = self.scores[self.scores.columns[-1]].rank(pct=True)
        
        # Local knee/elbow thresholds
        self.knee = KneeLocator(self.scores.index.tolist(), self.scores[self.scores.columns[-1]], interp_method='polynomial',curve='concave', direction='decreasing', online=on, S=sens, polynomial_degree=deg)
        self.elbow = KneeLocator(self.scores.index.tolist(), self.scores[self.scores.columns[-1]], interp_method='polynomial',curve='convex', direction='decreasing', online=on, S=sens, polynomial_degree=deg)
        
        # Adjust elbow/knee to avoid infinity values in features. Take minimum PPI score above 0.
        if self.knee.y[self.knee.knee] == 0:
            self.knee.knee = np.where(self.knee.y == min(np.ma.masked_equal(self.knee.y, 0)))[0][0]
            self.knee.knee_y = min(np.ma.masked_equal(self.knee.y, 0))
        if self.elbow.y[self.elbow.knee] == 0:
            self.elbow.knee = np.where(self.elbow.y == min(np.ma.masked_equal(self.elbow.y, 0)))[0][0]
            self.elbow.knee_y = min(np.ma.masked_equal(self.elbow.y, 0))
        
    def describe(self):
        # Describe inputs
        print('PPIs for %s:'%self.ID)
        print('\t%d predicted interactions'%self.scores.shape[0])
        print('\t%d labelled interactions:'%self.labels.labels.shape[0])
        print('\t\t%d positives'%self.labels.positive.shape[0])
        print('\t\t%d negatives'%self.labels.negative.shape[0])
        
    def get_ppi(self, protein):
        return get_protein_ppi(self.scores, protein)
    
    def get_rank(self, protein):
        df = self.get_ppi(protein)
        try:
            rank = self.scores.loc[(self.scores[self.scores.columns[0]] == df.iloc[0][df.columns[0]]) & (self.scores[self.scores.columns[1]] == df.iloc[0][df.columns[1]])].index[0]
        except IndexError:
            try:
                rank = self.scores.loc[(self.scores[self.scores.columns[1]] == df.iloc[0][df.columns[0]]) & (self.scores[self.scores.columns[0]] == df.iloc[0][df.columns[1]])].index[0]
            except IndexError:
                rank = np.nan
        return rank+1
    
    def get_score(self, protein):
        df = self.get_ppi(protein)
        try:
            score = df.iloc[0].iloc[-1]
        except IndexError:
            score = np.nan
        return score
    
    def get_relative_rank(self, protein):
        return self.get_rank(protein)/len(self.ranks)
    
    def get_knee_elbow(self):
        return self.knee.y[self.knee.knee], self.elbow.y[self.elbow.elbow]
    
    def get_features(self, protein):
        # rank_order = 1/(1 + rank)
        rank_order = 1/(1 + self.get_rank(protein))
        # rank_knee = (knee - rank)/len(ranks)
        rank_knee = (self.knee.knee - self.get_rank(protein))/len(self.ranks)
        # rank_elbow = (elbow - rank)/len(ranks)
        rank_elbow = (self.elbow.knee - self.get_rank(protein))/len(self.ranks)
        # score = score
        score = self.get_ppi(protein).iloc[0].iloc[-1]
        # score_knee = (score - kneescore)/max(scores)
        score_knee = score - self.scores.iloc[self.knee.knee].iloc[-1]
        # score_elbow = (score - elbowscore)/max(scores)
        score_elbow = (score - self.scores.iloc[self.elbow.knee].iloc[-1])/max(self.scores[self.scores.columns[-1]])
        # score_mean = (score - mean(scores))/max(scores)
        score_mean = score - np.mean(self.scores[self.scores.columns[-1]])
        # score_median = (score - median(scores))/max(scores)
        score_median = score - np.median(self.scores[self.scores.columns[-1]])
        
        features = np.array([self.ID, protein, rank_order, rank_knee, rank_elbow, score, score_knee, score_elbow, score_mean, score_median], dtype=object)
        
        return features
    
    def plot(self, protein=None):
        # Check if query protein has PPI
        ppi = self.get_ppi(protein)
        if not ppi.empty:
            # Check if query protein has label and assign color for plotting
            colour = 'black'
            # Check case for self-interacting PPI
            if protein == self.ID:
                if protein in self.labels.positive[self.labels.positive[self.labels.positive.columns[0]] == self.labels.positive[self.labels.positive.columns[1]]].values.flatten():
                    colour = 'green'
                if protein in self.labels.negative[self.labels.negative[self.labels.negative.columns[0]] == self.labels.negative[self.labels.negative.columns[1]]].values.flatten():
                    colour = 'red'
            # Check otherwise
            else:
                if protein in self.labels.positive.values.flatten():
                    colour = 'green'
                if protein in self.labels.negative.values.flatten():
                    colour = 'red'
        
            # Plot query protein
            ppi_y = ppi.iloc[0].iloc[-1]
            ppi_x = np.where(self.knee.y == ppi_y)[0][0]
            plt.scatter(ppi_x, ppi_y, facecolors='none', edgecolors=colour, linewidth=2, label=protein)
        
        # Plot one-to-all
        plt.plot(self.knee.x, self.knee.y)
        plt.vlines(self.elbow.knee,  min(self.elbow.y), max(self.elbow.y), linestyle='--', color='m', label='elbow', alpha=0.7)
        plt.vlines(self.knee.knee, min(self.knee.y), max(self.knee.y), linestyle='--', color='c', label='knee', alpha=0.7)
        plt.xlabel('Rank')
        plt.ylabel('Score')
        plt.title(self.ID + ' - to - All')
        plt.grid(alpha=0.25)
        plt.legend(prop={'size':8})
        plt.show()
    

class RP_AB(object):
    # df_predictions contains 3 columns as <proteinA> <proteinB> <score>
    # df_labels contains 3 columns as <proteinA> <proteinB> <label>
    # proteinA and protein B are each a string of the protein IDs for reciprocal perspectives
    # sens, deg, and on are used for finding the elbow/knee, see kneed.KneeLocator for info
    def __init__(self, df_predictions, df_labels, proteinA, proteinB, describe_input=True, sens=5, deg=7, on=True):
        # Perform some basic checks
        if df_predictions.shape[0] == 0 or df_predictions.shape[1] != 3:
            raise ValueError('df_predictions is invalid')
        if df_labels.shape[0] == 0 or df_labels.shape[1] != 3:
            raise ValueError('df_labels is invalid')
        colA = df_predictions.columns[0]
        colB = df_predictions.columns[1]
        proteins = df_predictions[colA].append(df_predictions[colB]).unique()
        if proteinA not in proteins or proteinB not in proteins:
            raise ValueError('Both proteinA and proteinB must exist in the predictions')
        if describe_input:
            print('Input data:')
            print('\t%d proteins'%len(proteins))
            print('\t%d predicted interactions'%df_predictions.shape[0])
            print('\t%d labelled interactions'%df_labels.shape[0])
        self.sensitivity=sens
        self.degree=deg
        self.online=on
        
        self.scores = df_predictions.copy()
        self.labels = Labels(df_labels)
        
        self.global_baseline_mean = np.mean(df_predictions[df_predictions.columns[-1]])
        self.global_baseline_median = np.median(df_predictions[df_predictions.columns[-1]])
        self.global_baseline_std = np.std(df_predictions[df_predictions.columns[-1]])
        
        # Filter one-to-all interactions for proteinA/B
        df_A = get_protein_ppi(df_predictions, proteinA)
        df_B = get_protein_ppi(df_predictions, proteinB)
        # Find all known interactions for proteinA/B
        labels_A = get_protein_ppi(df_labels, proteinA)
        labels_B = get_protein_ppi(df_labels, proteinB)

        # Create attributes for each one-to-all PPIs
        self.ProteinA = OneToAll(df_A, labels_A, proteinA, sens=sens, deg=deg, on=on)
        self.ProteinB = OneToAll(df_B, labels_B, proteinB, sens=sens, deg=deg, on=on)
    
    def get_rp_features(self):
        columns = ['Protein_A', 'Protein_B', 
                   'Rank_A_in_B', 'Rank_B_in_A', 'Score_A_in_B', 'Score_B_in_A',
                   'NaRRO', 'ARRO', 'NoRRO_A', 'NoRRO_B', 'NoRRO',
                   'Rank_LocalCutoff_A_elbow', 'Rank_LocalCutoff_B_elbow',
                   'Score_LocalCutoff_A_elbow', 'Score_LocalCutoff_B_elbow',
                   'Rank_LocalCutoff_A_knee', 'Rank_LocalCutoff_B_knee',
                   'Score_LocalCutoff_A_knee', 'Score_LocalCutoff_B_knee',
                   'Rank_AB_AboveLocal_A_elbow', 'Rank_BA_AboveLocal_B_elbow', 
                   'Rank_AB_AboveLocal_A_knee', 'Rank_BA_AboveLocal_B_knee',
                   'Above_Global_Mean', 'Above_Global_Median',
                   'FD_A_elbow', 'FD_B_elbow', 'FD_A_knee', 'FD_B_knee'
                   ]
        # RP features
        rank_A_in_B = self.ProteinB.get_rank(self.ProteinA.ID)
        rank_B_in_A = self.ProteinA.get_rank(self.ProteinB.ID)
        score_A_in_B = self.ProteinB.get_score(self.ProteinA.ID)
        score_B_in_A = self.ProteinA.get_score(self.ProteinB.ID)
        narro = 1.0 / (rank_A_in_B * rank_B_in_A)
        norro_A = 1.0 / (rank_A_in_B/len(self.ProteinB.proteins))
        norro_B = 1.0 / (rank_B_in_A/len(self.ProteinA.proteins))
        norro = norro_A * norro_B
        arro = 1.0 / (rank_A_in_B/len(self.ProteinB.proteins) * rank_B_in_A/len(self.ProteinA.proteins))

        # Rank and score of local cutoff (elbow)
        rank_local_cutoff_A_elbow = self.ProteinA.elbow.knee
        rank_local_cutoff_B_elbow = self.ProteinB.elbow.knee
        score_local_cutoff_A_elbow = self.ProteinA.elbow.y[self.ProteinA.elbow.knee]
        score_local_cutoff_B_elbow = self.ProteinB.elbow.y[self.ProteinB.elbow.knee]
        # Rank and score of local cutoff (knee)
        rank_local_cutoff_A_knee = self.ProteinA.knee.knee
        rank_local_cutoff_B_knee = self.ProteinB.knee.knee
        score_local_cutoff_A_knee = self.ProteinA.knee.y[self.ProteinA.knee.knee]
        score_local_cutoff_B_knee = self.ProteinB.knee.y[self.ProteinB.knee.knee]
        
        # Binary value indicating if rank of PPI is above rank of local elbow/knee
        rank_AB_above_local_A_elbow = int(self.ProteinA.get_rank(self.ProteinB.ID) < self.ProteinA.elbow.knee)
        rank_BA_above_local_B_elbow = int(self.ProteinB.get_rank(self.ProteinA.ID) < self.ProteinB.elbow.knee)
        rank_AB_above_local_A_knee = int(self.ProteinA.get_rank(self.ProteinB.ID) < self.ProteinA.knee.knee)
        rank_BA_above_local_B_knee = int(self.ProteinB.get_rank(self.ProteinA.ID) < self.ProteinB.knee.knee)
        
        # Binary value indicating if both RP PPI scored are above global mean/median score
        above_global_mean = int(score_A_in_B > self.global_baseline_mean and score_B_in_A > self.global_baseline_mean)
        above_global_median = int(score_A_in_B > self.global_baseline_median and score_B_in_A > self.global_baseline_median)
        
        # Fold differences (negative value indicates below local cutoff, positive value indicates above local cutoff)
        fd_A_elbow = (self.ProteinA.get_score(self.ProteinB.ID) - score_local_cutoff_A_elbow) / score_local_cutoff_A_elbow
        fd_B_elbow = (self.ProteinB.get_score(self.ProteinA.ID) - score_local_cutoff_B_elbow) / score_local_cutoff_B_elbow
        fd_A_knee = (self.ProteinA.get_score(self.ProteinB.ID) - score_local_cutoff_A_knee) / score_local_cutoff_A_knee
        fd_B_knee = (self.ProteinB.get_score(self.ProteinA.ID) - score_local_cutoff_B_knee) / score_local_cutoff_B_knee
        
        return pd.DataFrame(np.array([[self.ProteinA.ID, self.ProteinB.ID,
                                       rank_A_in_B, rank_B_in_A, score_A_in_B, score_B_in_A,
                                       narro, arro, norro_A, norro_B, norro,
                                       rank_local_cutoff_A_elbow, rank_local_cutoff_B_elbow,
                                       score_local_cutoff_A_elbow, score_local_cutoff_B_elbow,
                                       rank_local_cutoff_A_knee, rank_local_cutoff_B_knee,
                                       score_local_cutoff_A_knee, score_local_cutoff_B_knee,
                                       rank_AB_above_local_A_elbow, rank_BA_above_local_B_elbow,
                                       rank_AB_above_local_A_knee, rank_BA_above_local_B_knee,
                                       above_global_mean, above_global_median,
                                       fd_A_elbow, fd_B_elbow, fd_A_knee, fd_B_knee,
                                       ]]),
                            columns=columns)
    
    def plot(self):
        # Check ProteinB in ProteinA-to-All
        ppi_A = self.ProteinA.get_ppi(self.ProteinB.ID)
        if not ppi_A.empty:
            # Check if PPI has label and assign color for plotting
            colour = 'grey'
            if self.ProteinB.ID in self.ProteinA.labels.positive.values.flatten():
                colour = 'lime'
            if self.ProteinB.ID in self.ProteinA.labels.negative.values.flatten():
                colour = 'red'
            # Plot ProteinB on ProteinA-to-All
            ppi_y = ppi_A.iloc[0].iloc[-1]
            ppi_x = np.where(self.ProteinA.knee.y == ppi_y)[0][0]
            plt.scatter(ppi_x, ppi_y, facecolors='none', edgecolors=colour, linewidth=2, label=self.ProteinB.ID)
            #plt.annotate(self.ProteinB.ID, (ppi_x, ppi_y), xytext=(ppi_x+ppi_x*0.05, ppi_y+ppi_y*0.05),
            #             arrowprops=dict(alpha=0.5, arrowstyle='->'))
        # Check ProteinA in ProteinB-to-All
        ppi_B = self.ProteinB.get_ppi(self.ProteinA.ID)
        if not ppi_B.empty:
            # Check if has label and assign color for plotting
            colour = 'black'
            if self.ProteinA.ID in self.ProteinB.labels.positive.values.flatten():
                colour = 'green'
            if self.ProteinA.ID in self.ProteinB.labels.negative.values.flatten():
                colour = 'darkred'
            # Plot ProteinA on ProteinB-to-All
            ppi_y = ppi_B.iloc[0].iloc[-1]
            ppi_x = np.where(self.ProteinB.knee.y == ppi_y)[0][0]
            plt.scatter(ppi_x, ppi_y, facecolors='none', edgecolors=colour, linewidth=2, label=self.ProteinA.ID)
            #plt.annotate(self.ProteinA.ID, (ppi_x, ppi_y), xytext=(ppi_x+ppi_x*0.05, ppi_y+ppi_y*0.05),
            #             arrowprops=dict(alpha=0.5, arrowstyle='->'))
        # Plot one-to-all
        plt.plot(self.ProteinA.knee.x, self.ProteinA.knee.y, label=self.ProteinA.ID+'-to-all', color='royalblue', alpha=0.7)
        plt.vlines(self.ProteinA.elbow.knee,  min(self.ProteinA.elbow.y), max(self.ProteinA.elbow.y), linestyle='--', color='m', alpha=0.5)
        plt.vlines(self.ProteinA.knee.knee, min(self.ProteinA.knee.y), max(self.ProteinA.knee.y), linestyle='--', color='c', alpha=0.5)
        plt.plot(self.ProteinB.knee.x, self.ProteinB.knee.y, label=self.ProteinB.ID+'-to-all', color='darkblue', alpha=0.7)
        plt.vlines(self.ProteinB.elbow.knee,  min(self.ProteinB.elbow.y), max(self.ProteinB.elbow.y), linestyle='--', color='darkmagenta', alpha=0.5)
        plt.vlines(self.ProteinB.knee.knee, min(self.ProteinB.knee.y), max(self.ProteinB.knee.y), linestyle='--', color='darkcyan', alpha=0.5)
        plt.grid(alpha=0.25)
        plt.xlabel('Rank')
        plt.ylabel('Score')
        plt.title('One - to - All PPIs')
        plt.legend(prop={'size': 8})
        plt.show()

def get_protein_ppi(df, proteinID):
    # Return df filtering for ppi containing proteinID and sort descending
    df = df[df[df.columns[0]] == proteinID].append(df[df[df.columns[1]] == proteinID])
    df.drop_duplicates(inplace=True)
    df.sort_values(by=df.columns[-1], ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df
    
def create_RP_dataset(predictions, labels):
    pred = predictions.copy()
    lab = labels.copy()
    start = time.time()
    df = pd.DataFrame()
    for i in range(0, lab.shape[0]):
        print('\r\t%s out of %s PPIs'%(i+1, lab.shape[0]), end='')
        rp = RP_AB(pred, lab, lab.iloc[i][0], lab.iloc[i][1], describe_input=False)
        df = df.append(rp.get_rp_features(), ignore_index=True)
    print('\n\tTime:', round(time.time() - start, 2), 'seconds')
    lab.rename(columns={0: 'Protein_A', 1:'Protein_B', 2:'label'}, inplace=True)
    df = df.merge(lab, on=['Protein_A', 'Protein_B'])
    return df

def filter_dataset(df_rp, df_labels):
    rp = df_rp.copy()
    rp.rename(columns={rp.columns[0]: 'Protein_A', rp.columns[1]: 'Protein_B'}, inplace=True)
    labels = df_labels.copy()
    labels.rename(columns={labels.columns[0]: 'Protein_A', labels.columns[1]: 'Protein_B', labels.columns[-1]: 'label'}, inplace=True)
    df = rp.merge(labels, on=['Protein_A', 'Protein_B'])
    rp[['Protein_A', 'Protein_B']] = rp[['Protein_B', 'Protein_A']]
    df = df.append(rp.merge(labels, on=['Protein_A', 'Protein_B']), ignore_index=True)
    df.drop_duplicates(inplace=True)
    df.sort_values(by=[labels.columns[-1]], ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# ========= Intended for processing cross-validation all-to-all predictions for Reciprocal Perspective ===========
# Reads all k-fold subsets and all-to-all tested predictions
# Removes training PPIs from all-to-all prediction for each k-fold
# Returns the single score for labelled PPIs and average score of k-folds for other all-to-all PPIs
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
    matches = df_train.merge(df, on=[df.columns[0], df.columns[1]]).drop_duplicates(subset=[df.columns[0], df.columns[1]])
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

def fasta_to_df(df):
    fasta = df.copy()
    prot = fasta.iloc[::2, :].reset_index(drop=True)
    seq = fasta.iloc[1::2, :].reset_index(drop=True)
    prot.insert(1, 1, seq)
    return prot

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
    # Get training PPI subsets to remove from all-to-all PPI predictions
    cv_files = os.listdir(path=cv_subset_dir_path)
    train_files = [ x for x in cv_files if 'train' in x and '_neg' not in x ]
    train_files.sort()
    
    # Get all-to-all PPI predictions to compile avgerage PPI scores
    prediction_files = os.listdir(path=prediction_results_dir_path)
    pred_files = [ x for x in prediction_files if 'prediction' in x ]
    pred_files.sort()
    
    if len(train_files) != len(pred_files):
        print('CV subsets and all-to-all predictions mismatch')
        return pd.DataFrame()
    
    # Compile all_to_all predictions from cross-validation
    df_cv = pd.DataFrame()
    for i in range(0, len(train_files)):
        # Read trained PPI subset
        trained = read_df(cv_subset_dir_path + train_files[i])
        # Read all-to-all PPI predictions
        predictions = pd.read_csv(prediction_results_dir_path + pred_files[i], delim_whitespace=True, header=None)
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

#rp_fast = [ RP_AB(predictions, labels, labels.iloc[i][0], labels.iloc[i][1], describe_input=False).get_rp_feature() for i in range(0, labels.shape[0]) ]
if __name__ == '__main__':
    
    if not os.path.exists(args.results):
        os.mkdir(args.results)
    
    print('Reading labels...')
    labels = pd.read_csv(args.labels, delim_whitespace=True, header=None)
    save_name = 'RP_' + args.labels.split('/')[-1]
    print('\t%s labelled PPIs'%labels.shape[0])
    
    print('Averaging k-fold predictions...')
    avg_all_preds = average_all_to_all(args.trained, args.predictions)
    all_predictions = avg_all_preds[[avg_all_preds.columns[0], avg_all_preds.columns[1], avg_all_preds.columns[-1]]]
    print('\t%s all-to-all PPIs'%all_predictions.shape[0])
    
    print('Creating RP features dataset...')
    rp = create_RP_dataset(all_predictions, labels)
    print('\t%s RP features extracted for %s PPIs'%(rp.shape[1] - 3, rp.shape[0]))
    
    rp.to_csv(args.results + save_name, sep='\t', index=False)
    print('Saved and done.')
    
    