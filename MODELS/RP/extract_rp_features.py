#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description:
    Extracts Reciprocal Perspective (RP) features from all-to-all PPI predictions.
    RP objects can be created given an all-to-all .tsv file and labelled PPI .tsv file.
    
    RP_AB object supports:
        RP_AB objects, One-to-All objects,
        Plotting one-to-all curves for a given protein ID.
        Plotting RP one-to-all curves for a given pair of protein IDs.
    
    Input arguements:
        -l: <str> path to labeled dataset to convert to RP dataset (.tsv)
        -p: <str> path to file containing all-to-all PPI predictions
        -r: <str> path to directory to save RP feature dataset
        -m: <float> Percent of available processors to use multiprocessing, default is 0 (no multiprocessing)
    
    Output files:
        A single .tsv file of RP features for each PPI found in given labelled pairs.
    
@author: Eric Arezza
Last Updated: October 13, 2021
"""

import os, argparse
import warnings
import numpy as np
import pandas as pd
from kneed import KneeLocator
import matplotlib.pyplot as plt
import time
import tqdm as tqdm
import multiprocessing

describe_help = 'python extract_rp_features.py -l labels.tsv -p predictions.tsv -r RESULTS/ -m 0.5'
parser = argparse.ArgumentParser(description=describe_help)
parser.add_argument('-l', '--labels', help='Path to labeled PPIs file to convert to RP dataset (.tsv)', type=str)
parser.add_argument('-p', '--predictions', help='Path to directory with all-to-all PPI prediction files', type=str)
parser.add_argument('-r', '--results', help='Path to directory to save new RP dataset', type=str, default=os.getcwd()+'/')
parser.add_argument('-m', '--multiprocessing', help='Percent of processors to use for multiprocessing (default 0 = no multiprocessing)', type=float, default=0)
args = parser.parse_args()


class Labels(object):
    # For sifting through labelled PPI data
    def __init__(self, df_labels):
        self.all = df_labels.copy()
        self.positive = self.all[self.all[self.all.columns[-1]] == 1].reset_index(drop=True)
        self.negative = self.all[self.all[self.all.columns[-1]] == 0].reset_index(drop=True)
    def get_ppi(self, protein):
        return get_protein_ppi(self.labels, protein)

# If no knee found, manually assign knee to last ranked PPI
class No_Knee(object):
    def __init__(self, scores):
        self.knee = scores.index.tolist()[-1]
        self.knee_y = scores.iloc[self.knee][scores.columns[-1]]
        self.y = np.array(scores.copy()[scores.columns[-1]])

# If no elbow found, manually assign elbow to first rank PPI
class No_Elbow(object):
    def __init__(self, scores):
        self.knee = scores.index.tolist()[0]
        self.knee_y = scores.iloc[self.knee][scores.columns[-1]]
        self.y = np.array(scores.copy()[scores.columns[-1]])

class OneToAll(object):
    def __init__(self, df_scores, df_labels, proteinID, sens=5, deg=7, on=True):
        self.scores = df_scores.copy()
        self.labels = Labels(df_labels.copy())
        self.proteins = self.scores[self.scores.columns[0]].append(self.scores[self.scores.columns[1]]).unique()
        self.ID = proteinID
        self.sensitivity=sens
        self.degree=deg
        self.online=on

        # # Basic local stats of scores
        self.scores_mean = np.mean(self.scores[self.scores.columns[-1]])
        self.scores_median = np.median(self.scores[self.scores.columns[-1]])
        self.scores_std = np.std(self.scores[self.scores.columns[-1]])
        
        self.ranks = self.scores[self.scores.columns[-1]].rank(ascending=False)
        self.percentiles = self.scores[self.scores.columns[-1]].rank(pct=True)
        
        # Local knee/elbow thresholds, knee is 'concave' elbow is 'convex'
        if not self.scores[self.scores.columns[-1]].any():
            # When all scores are 0, no knee/elbow possible
            self.knee = No_Knee(self.scores)
            self.elbow = No_Elbow(self.scores)
        else:
            try:
                # Adjust sensitivity until knee found to avoid None value
                for s in range(sens, -1, -1):
                    self.knee = KneeLocator(self.scores.index.tolist(), self.scores[self.scores.columns[-1]], interp_method='polynomial', curve='concave', direction='decreasing', online=on, S=s, polynomial_degree=deg)
                    if self.knee.knee:
                        self.sensitivity=s
                        break
            except:
                # If still no knee found, manually assign knee to last rank PPI
                if self.knee.knee == None:
                    self.knee = No_Knee(self.scores)
            try:
                # Adjust sensitivity until elbow found to avoid None value
                for s in range(sens, -1, -1):
                    self.elbow = KneeLocator(self.scores.index.tolist(), self.scores[self.scores.columns[-1]], interp_method='polynomial', curve='convex', direction='decreasing', online=on, S=s, polynomial_degree=deg)
                    if self.elbow.knee:
                        self.sensitivity=s
                        break
            except:
                # If still no elbow found, manually assign elbow to first rank PPI
                if self.elbow.knee == None:
                    self.elbow = No_Elbow(self.scores)
        if self.knee.knee == None:
            self.knee = No_Knee(self.scores)
        if self.elbow.knee == None:
            self.elbow = No_Elbow(self.scores)
        # Swap if poor elbow/knee detections
        if self.knee.knee < self.elbow.knee:
            temp = self.knee
            self.knee = self.elbow
            self.elbow = temp
            del temp
        
    def describe(self):
        # Describe inputs
        print('PPIs for %s:'%self.ID)
        print('\t%d predicted interactions'%self.scores.shape[0])
        print('\t%d labelled interactions:'%self.labels.all.shape[0])
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
        return rank
    
    def get_score(self, protein):
        df = self.get_ppi(protein)
        try:
            score = df.iloc[0].iloc[-1]
        except IndexError:
            score = np.nan
        return score
    
    def get_relative_rank(self, protein):
        return (self.get_rank(protein))/len(self.ranks)
    
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
        score = self.get_score(protein)
        # score_knee = (score - kneescore)/max(scores)
        score_knee = score - self.scores.iloc[self.knee.knee].iloc[-1]/max(self.scores[self.scores.columns[-1]])
        # score_elbow = (score - elbowscore)/max(scores)
        score_elbow = (score - self.scores.iloc[self.elbow.knee].iloc[-1])/max(self.scores[self.scores.columns[-1]])
        # score_mean = (score - mean(scores))/max(scores)
        score_mean = score - np.mean(self.scores[self.scores.columns[-1]])/max(self.scores[self.scores.columns[-1]])
        # score_median = (score - median(scores))/max(scores)
        score_median = score - np.median(self.scores[self.scores.columns[-1]])/max(self.scores[self.scores.columns[-1]])
        
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
                elif protein in self.labels.negative[self.labels.negative[self.labels.negative.columns[0]] == self.labels.negative[self.labels.negative.columns[1]]].values.flatten():
                    colour = 'red'
            # Check otherwise
            else:
                if protein in self.labels.positive.values.flatten():
                    colour = 'green'
                elif protein in self.labels.negative.values.flatten():
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
    def __init__(self, df_predictions, df_labels, proteinA, proteinB, sens=5, deg=7, on=True):
        # RP global attributes
        # Scores and labels
        self.scores = df_predictions.copy()
        self.labels = Labels(df_labels)
        
        # Attributes for KneeLocator
        self.sensitivity=sens
        self.degree=deg
        self.online=on

        # Basic global stats of scores
        self.global_baseline_mean = np.mean(df_predictions[df_predictions.columns[-1]])
        self.global_baseline_median = np.median(df_predictions[df_predictions.columns[-1]])
        self.global_baseline_std = np.std(df_predictions[df_predictions.columns[-1]])
        
        # RP local attributes
        # Get scores relevant to proteinA and proteinB
        self.scores_A = get_protein_ppi(df_predictions, proteinA)
        self.scores_B = get_protein_ppi(df_predictions, proteinB)
        # Get labels relevant to proteinA and proteinB
        self.labels_A = get_protein_ppi(df_labels, proteinA)
        self.labels_B = get_protein_ppi(df_labels, proteinB)
        
        # Create attributes for each one-to-all PPIs, ignore warnings from KneeLocator in choosing params
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.ProteinA = OneToAll(self.scores_A, self.labels_A, proteinA, sens=self.sensitivity, deg=self.degree, on=self.online)
            self.ProteinB = OneToAll(self.scores_B, self.labels_B, proteinB, sens=self.sensitivity, deg=self.degree, on=self.online)
    
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
        narro = 1.0 / ((rank_A_in_B+1) * (rank_B_in_A+1))
        norro_A = 1.0 / ((rank_A_in_B+1)/len(self.ProteinB.proteins))
        norro_B = 1.0 / ((rank_B_in_A+1)/len(self.ProteinA.proteins))
        norro = norro_A * norro_B
        arro = 1.0 / ((rank_A_in_B+1)/len(self.ProteinB.proteins) * (rank_B_in_A+1)/len(self.ProteinA.proteins))

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
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fd_A_elbow = (self.ProteinA.get_score(self.ProteinB.ID) - score_local_cutoff_A_elbow) / score_local_cutoff_A_elbow if not np.isinf((self.ProteinA.get_score(self.ProteinB.ID) - score_local_cutoff_A_elbow) / score_local_cutoff_A_elbow) and not np.isnan((self.ProteinA.get_score(self.ProteinB.ID) - score_local_cutoff_A_elbow) / score_local_cutoff_A_elbow) else 0
            fd_B_elbow = (self.ProteinB.get_score(self.ProteinA.ID) - score_local_cutoff_B_elbow) / score_local_cutoff_B_elbow if not np.isinf((self.ProteinB.get_score(self.ProteinA.ID) - score_local_cutoff_B_elbow) / score_local_cutoff_B_elbow) and not np.isnan((self.ProteinB.get_score(self.ProteinA.ID) - score_local_cutoff_B_elbow) / score_local_cutoff_B_elbow) else 0
            fd_A_knee = (self.ProteinA.get_score(self.ProteinB.ID) - score_local_cutoff_A_knee) / score_local_cutoff_A_knee if not np.isinf((self.ProteinA.get_score(self.ProteinB.ID) - score_local_cutoff_A_knee) / score_local_cutoff_A_knee) and not np.isnan((self.ProteinA.get_score(self.ProteinB.ID) - score_local_cutoff_A_knee) / score_local_cutoff_A_knee)  else 0
            fd_B_knee = (self.ProteinB.get_score(self.ProteinA.ID) - score_local_cutoff_B_knee) / score_local_cutoff_B_knee if not np.isinf((self.ProteinB.get_score(self.ProteinA.ID) - score_local_cutoff_B_knee) / score_local_cutoff_B_knee) and not np.isnan((self.ProteinB.get_score(self.ProteinA.ID) - score_local_cutoff_B_knee) / score_local_cutoff_B_knee) else 0
            
        rp_features = pd.DataFrame(np.array([[self.ProteinA.ID, self.ProteinB.ID,
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
        return rp_features
    
    def plot(self):
        # Check ProteinB in ProteinA-to-All
        ppi_A = self.ProteinA.get_ppi(self.ProteinB.ID)
        if not ppi_A.empty:
            # Check if PPI has label and assign color for plotting
            colour = 'grey'
            if self.ProteinB.ID in self.ProteinA.labels.positive.values.flatten():
                colour = 'lime'
            elif self.ProteinB.ID in self.ProteinA.labels.negative.values.flatten():
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
            elif self.ProteinA.ID in self.ProteinB.labels.negative.values.flatten():
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

def create_RP_dataset(predictions, labels):
    pred = predictions.copy()
    lab = labels.copy()
    start = time.time()
    df = pd.DataFrame()
    for i in tqdm.tqdm(range(0, lab.shape[0]), total=lab.shape[0]):
        rp = RP_AB(pred, lab, lab.iloc[i][0], lab.iloc[i][1])
        df = df.append(rp.get_rp_features())
        df.reset_index(drop=True, inplace=True)
    print('\n\tTime:', round(time.time() - start, 2), 'seconds')
    lab.rename(columns={0: 'Protein_A', 1:'Protein_B', 2:'label'}, inplace=True)
    df = df.merge(lab, on=['Protein_A', 'Protein_B'])
    return df

# Running parallel processes to speed up RP feature extraction
def get_rp(i):
    return RP_AB(PREDICTIONS, LABELS, LABELS.iloc[i][0], LABELS.iloc[i][1]).get_rp_features()

def create_RP_dataset_parallel(predictions, labels, processors=round(os.cpu_count())):
    start = time.time()

    with multiprocessing.Pool(processors) as pool:
        df = list(tqdm.tqdm(pool.imap(get_rp, range(0, labels.shape[0])), total=labels.shape[0]))

    df = pd.concat(df)
    df.reset_index(drop=True, inplace=True)
    LABELS.rename(columns={0: 'Protein_A', 1:'Protein_B', 2:'label'}, inplace=True)
    df = df.merge(LABELS, on=['Protein_A', 'Protein_B'])
    print('\n\tTime:', round(time.time() - start, 2), 'seconds')
    pool.close()
    pool.join()
    return df

def get_protein_ppi(df_in, proteinID):
    df = df_in.copy()
    # Return df only for PPIs containing proteinID and sort descending
    df = df[(df[df.columns[0]] == proteinID) | (df[df.columns[1]] == proteinID)]
    df.sort_values(by=df.columns[-1], ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

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

def labels_verified(labels, predictions):
    matches = get_matching_pairs(labels, predictions)
    if labels.shape[0] != matches.shape[0]:
        print('%s/%s labelled pairs found in predictions...unable to extract RP features for given labels.'%(matches.shape[0], labels.shape[0]))
        return False
    return True

def remove_redundant_pairs(df_ppi):
    df = df_ppi.copy()
    df.sort_values(by=[df.columns[0], df.columns[1]], inplace=True)
    df.reset_index(drop=True, inplace=True)
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
        df_labelled.sort_values(by=[df_labelled.columns[-1]], ascending=False, inplace=True)
        df_labelled.reset_index(drop=True, inplace=True)
        df_labelled = df_labelled.drop_duplicates(subset=[df_labelled.columns[0], df_labelled.columns[1]])
        df_out = df_labelled.reset_index(drop=True)
    else:
        df_out = df_unique.copy()
        df_out.sort_values(by=[df_out.columns[0], df_out.columns[1]], inplace=True)
        df_out.reset_index(drop=True, inplace=True)
    
    return df_out

def prep_df(df_in):
    df = remove_redundant_pairs(df_in)
    df = df[df.columns[:3]]
    return df
    
if __name__ == '__main__':
    
    if not os.path.exists(args.results):
        os.mkdir(args.results)
    start = time.time()
    print('Reading labels...')
    labels = pd.read_csv(args.labels, delim_whitespace=True, header=None)
    labels = prep_df(labels)
    
    print('Reading predictions...')
    predictions = pd.read_csv(args.predictions, delim_whitespace=True, header=None)
    predictions = prep_df(predictions)
    
    print('Verifying input data...')
    if labels_verified(labels, predictions):
        print('Creating RP features dataset...')
        if args.multiprocessing > 0:
            print('\tExecuting parallel...%s cpus out of %s'%(round(os.cpu_count()*args.multiprocessing), os.cpu_count()))
            PREDICTIONS = predictions.copy()
            LABELS = labels.copy()
            rp = create_RP_dataset_parallel(PREDICTIONS, LABELS, processors=round(os.cpu_count()*args.multiprocessing))
        else:
            rp = create_RP_dataset(predictions, labels)
        print('\t%s RP features extracted for %s PPIs'%(rp.shape[1] - 3, rp.shape[0]))
        save_name = 'RP_' + args.labels.split('/')[-1]
        rp.replace(to_replace=np.nan, value=0, inplace=True)
        rp.to_csv(args.results + save_name, sep='\t', index=False)
        print('Saved and done.')
    else:
        print('Labels missing from predictions')
        exit()
    print('\n\tTime:', round(time.time() - start, 2), 'seconds')
