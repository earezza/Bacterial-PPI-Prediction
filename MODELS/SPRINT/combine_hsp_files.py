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
import tqdm

describe_help = 'python combine_hsp_files.py -f hash0.hsp hash1.hsp hash2.hsp hash3.hsp -r HSP/ -n hsps.hsp'
parser = argparse.ArgumentParser(description=describe_help)
parser.add_argument('-f', '--files', help='Path to HSP files to combine', type=str, nargs='+')
parser.add_argument('-r', '--results', help='Path to directory to save HSP file', type=str, default=os.getcwd()+'/')
parser.add_argument('-n', '--name', help='Filename of resulting hsp file', type=str, default='hsps')
args = parser.parse_args()

# Read hsp file as pd.read_csv(file, sep='\n', header=None)
def convert_hsp_to_df(h):
    pairs = h[h[0].str[0] == '>']
    if pairs.index[-1] == h.index[-1] - 1:
        hits = pairs.index.insert(len(pairs.index), h.index[-1] + 1)
    else:
        hits = pairs.index.insert(len(pairs.index), h.index[-1])
    hsps = np.array([ h.iloc[hits[i]+1:hits[i+1]].values.flatten().tolist() for i in tqdm.tqdm(range(0, len(pairs.index))) ], dtype=object)
    pairs.insert(1,1, hsps)
    pairs.reset_index(drop=True, inplace=True)
    return pairs
    

if __name__ == '__main__':
    
    t_start = time.time()
    
    if not os.path.exists(args.results):
        os.mkdir(args.results)

    # Read SPRINT hsp files
    hsps = []
    for f in args.files:
        print('Reading file %s...'%f)
        hsps.append(pd.read_csv(f, sep='\n', header=None))

    # Convert to DataFrames
    for h in range(0, len(hsps)):
        print('Converting to dataframe %s...'%h)
        hsps[h] = convert_hsp_to_df(hsps[h])

    # Combine all
    print('Combining HSP dataframes...'%h)
    hsp = pd.concat(hsps, ignore_index=True)
    
    # Group all hsps found for all pairs from all files
    print('Grouping HSPs...')
    hsp = hsp.explode(hsp.columns[-1], ignore_index=True)
    group = hsp.groupby([hsp.columns[0]])[hsp.columns[1]].apply(lambda x: x.unique()).reset_index()
    
    # Write HSPs to file
    print('Writing to file %s...'%args.name)
    f = open(args.results + args.name, 'w')
    for i in range(0, group.shape[0]):
        f.write(group.iloc[i][0] + '\n')
        for j in group.iloc[i][1].tolist():
            f.write(j + '\n')
    f.close()
    
    print("Time: %s"%round(time.time()-t_start, 4))
    
    
    