#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 16:49:50 2021

@author: erixazerro
"""

import os, argparse
import numpy as np
import pandas as pd
import time

describe_help = 'python combine_hsp_files.py -f hash0.hsp hash1.hsp hash2.hsp hash3.hsp -r HSP/ -n hsps.hsp'
parser = argparse.ArgumentParser(description=describe_help)
parser.add_argument('-f', '--files', help='Path to HSP files to combine', type=str, nargs='+')
parser.add_argument('-r', '--results', help='Path to directory to save HSP file', type=str, default=os.getcwd()+'/')
parser.add_argument('-n', '--name', help='Filename of resulting hsp file', type=str, default='hsps')
args = parser.parse_args()

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

if __name__ == '__main__':
    
    t_start = time.time()
    
    if not os.path.exists(args.results):
        os.mkdir(args.results)

    # Read SPRINT hsp files
    hsps = []
    for f in args.files:
        hsps.append(pd.read_csv(f, sep='\n', header=None))

    # Convert to dictionary for pairs: hsps
    for h in range(0, len(hsps)):
        hsps[h] = hsp_to_dict(hsps[h])
    
    # Convert to DataFrames
    for h in range(0, len(hsps)):
        hsps[h] = hsp_dict_to_df(hsps[h])
    
    # Combine all
    hsp = pd.DataFrame()
    for h in range(0, len(hsps)):
        hsp = hsp.append(hsps[h], ignore_index=True)
    
    # Group all hsps found for all pairs from all files
    hsp = hsp.explode(hsp.columns[-1], ignore_index=True)
    group = hsp.groupby([hsp.columns[0]])[hsp.columns[1]].apply(lambda x: x.unique()).reset_index()
    
    # Write HSPs to file
    hsps = pd.DataFrame(group[1].to_list(), index=group[1].index)
    hsps.insert(0, 'Pair', group[0])
    f = open(args.results + args.name, 'w')
    for i in range(0, hsps.shape[0]):
        for j in range(0, hsps.shape[1]):
            if hsps.iloc[i][hsps.columns[j]] != None:
                f.write(hsps.iloc[i][hsps.columns[j]] + '\n')
    f.close()
    
    print("Time: %s"%round(time.time()-t_start, 4))
    
    