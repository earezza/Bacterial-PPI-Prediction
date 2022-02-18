#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description:
    
    Builds balanced protein-protein interaction (PPI) datasets from HPIDB data.
    https://hpidb.igbb.msstate.edu/index.html
    
    Requires:
        - HPIDB .mitab_plus.txt file (for extracting protein interactions)
        - CD-HIT software installed (for removing homologous proteins)
        - an internet connection (for accessing the UniProt database)
    
    Preprocessing Steps:
        1. Extract positive interactions from HPIDB:
            - Collects interactions from HPIDB file and queries UniProt database 
              for filtering un/reviewed (Swiss-Prot) protein IDs and sequences.
            
            Options:
                -f <flag> apply conservative filters to interactions
                -h <int> host interactor organism ID
                    If None, all host-pathogen interactions extracted
                -p <int> pathogen interactor organism ID (can be a list of IDs)
                    If None, all host-pathogen interactions extracted
                -u <flag> include unreviewed UniProt entries
                -c <int> confidence level of each interaction, 0 (least conservative), 1, or 2 (most conservative).
                    Level 0: include all listed interactions
                    Level 1: only interactions listed multiple times
                    Level 2 (DEFAULT): only interactions listed multiple times with different sources
        
        2. Remove homologous proteins from positive interactions using CD-HIT
            - Uses CD-HIT algorithm:
                E.g. let sequence identity threshold == 0.7
                - All sequences sorted largest to smallest
                - using first sequence as representative of cluster:
                    - find all other sequences that have >= 70% sequence identity to representative
                    - cluster all found sequences under representative
                    - use next largest sequence that had < 70% identity from representative as new representative sequence in new cluster
                    - repeat
                i.e.    
                    if threshold is 1.0, no proteins will be removed from the dataset 
                    if threshold is < 1.0, only representative proteins will be included in the dataset
                    lower thresholds will remove more proteins
            
            Options:
                -cdhit <str> Path to binary executable for cd-hit (optional if not in /usr/bin/)
                -s <float> sequence identity threshold to remove homologous proteins 
                    - valid values between [0.4 to 1.0]
                    - if 0.0, this step will be skipped (e.g. if CD-HIT not installed)
                
        3. Generate negative interactions:
            - Uses remaining proteins found in positive pairs and generates random pairs not found in positives
            - Repeats until number of negatives == number of positives
            - For inter-species PPIs, negatives generated are also inter-species
            - option allows for selecting pairs of proteins found in different subcellular locations as listed in UniProt
            
            Options:
                -d <flag> select pairs of proteins found in different subcellular locations
        
        4. Save balanced PPI dataset
            - dataset is labelled and saved under HPIDB_DATA/ as a .tsv file with no header with a .fasta file
            - options allow for additionally saving dataset as formatted for different PPI prediction models
            - option allows for creating k-fold subsets of data
            
            Options:
                -n <str> name to rename the files from the HPIDB filename (DEFAULT)
                -r <str> directory location to save the resulting datasets (default is HPIDB_DATA/)
                -m <list> choice of PPI prediction models formatting for dataset in addition to saving original data (DEFAULT)
                    pipr: labelled and saved under PIPR_DATA/ as a .tsv file with a tab-separated .fasta file
                    deepfe: saved under DEEPFE_DATA/ as positive_A.fasta, positive_B.fasta, negative_A.fasta, negative_B.fasta
                    dppi: saved under DPPI_DATA/ as data.node file of all protein IDs, a labelled data.csv of interactions, and a data/ with .fasta for each protein
                        NOTE: dppi requires PSI-BLAST to be performed to get PSSM of proteins fasta .txt files
                    sprint: labelled and saved under SPRINT_DATA/ as a space-separated .txt file with a tab-separated .fasta file
                -k <int> create k-fold subsets of data for use in cross-validation
                    5 (DEFAULT): saves data subsets under CV_SET/ including formatted data as per -m option
                    0 or 1: does not create k-fold subsets
                -a <flag> generates all-to-all PPIs, positively labelled, for proteins in the final dataset (BE MINDFUL OF HARDDRIVE/STORAGE)
                -pm <int> creates number of Park&Marcotte sets from final dataset for evaluations, default is 0)
                
@author: Eric Arezza
Last Updated: August 30 2021
"""

__all__ = ['get_hpidb_interactions',
           'separate_species_interactions',
           'check_ppi_confidence',
           'map_hpidb_to_uniprot',
           'run_cdhit',
           'remove_homology_ppi',
           'generate_negative_interactions',
           'get_protein_locations',
           'read_fasta',
           'remove_redundant_pairs',
           'save_ppi_data',
           'format_ppi_data',
           'convert_pipr',
           'convert_sprint',
           'convert_deepfe',
           'convert_dppi',
           'create_cv_subsets',
           'park_marcotte_subsets'
           ]
__version__ = '1.0'
__author__ = 'Eric Arezza'

import os
import subprocess
import argparse
import time
import re
import csv
import pandas as pd
import numpy as np
import urllib.parse
import urllib.request
import math
from io import StringIO
from sklearn.model_selection import StratifiedKFold, train_test_split
from itertools import combinations_with_replacement

describe_help = 'python preprocess_hpidb.py filename.txt -cdhit /usr/bin/cd-hit -c2 -f -s0.6 -m pipr sprint deepfe dppi -k5'
parser = argparse.ArgumentParser(description=describe_help)
parser.add_argument('file', help='Full path to HPIDB .mitab_plus.txt file', type=str)
parser.add_argument('-cdhit', help='Full path to CD-HIT binary executable (can be omitted)', type=str, nargs='?', default='cd-hit')
parser.add_argument('-c', '--confidence_level',
                    help='Confidence level of interactions, 0: include all interactions\n 1: only interactions listed more than once\n2 (default): only interactions with multiple different sources',
                    choices=(0, 1, 2), type=int, default=2)
parser.add_argument('-f', '--filter', help='Flag to apply conservative filters (desirable)', action='store_true')
parser.add_argument('-host', '--host_id', help='Organism ID of host',
                    type=int)
parser.add_argument('-pathogen', '--pathogen_id', help='Organism IDs of pathogens (can be list)',
                    type=int, nargs='+')
parser.add_argument('-u', '--unreviewed', help='Flag to include unreviewed UniProt entries (default false)', action='store_true')
parser.add_argument('-s', '--sequence_identity', help='Sequence identity threshold for removing homologous proteins (0.4 minimum, 1.0 is no removal) default 0.6',
                    type=float, default=0.6)
parser.add_argument('-d', '--diff_subcell_local', action='store_true', help='Flag to sample from proteins in seperate subcellular localizations when generating negative PPIs')
parser.add_argument('-r', '--results', help='Path to directory for saving dataset files', 
                    type=str, default=os.getcwd()+'/HPIDB_DATA/')
parser.add_argument('-n', '--name', help='Name used for saving files', type=str, nargs='?')
parser.add_argument('-m', '--models', help='Model for dataset formatting', 
                    choices=('pipr', 'sprint', 'deepfe', 'dppi'),  
                    default=[], type=str, nargs='+')
parser.add_argument('-k', '--kfolds', help='Number of K-Fold splits of data, 0 or 1 produces no subsets (default 5)', type=int, default=5)
parser.add_argument('-a', '--all_to_all', help='Flag to generate all-to-all PPIs for proteins in the final dataset', action='store_true')
parser.add_argument('-pm', '--park_marcotte', help='Number of Park & Marcotte sets to create from final datasets (default 0)', type=int, default=0)
args = parser.parse_args()

if args.name == None:
    name = args.file.split('/')[-1].split('-')[-2]
    FILENAME = name + '_' + '-'.join(args.file.split('/')[-1].split('-')[-1].replace('.', '_').split('_')[:-2])
else:
    FILENAME = args.name

# HPIDB columns used
TAB_COLS = [
    'protein_xref_1', 
    'protein_xref_2', 
    'pmid', 
    'protein_taxid_1', 
    'protein_taxid_2',
    'detection_method',
    'interaction_type']
TAB_PLUS_COLS = [
    'protein_xref_1_unique', 
    'protein_xref_2_unique', 
    'pmid', 
    'protein_taxid_1', 
    'protein_taxid_2',
    'detection_method',
    'interaction_type']

# Account for version and formatting changes based on filename
if '.mitab.txt' in args.file:
    COLS = TAB_COLS
elif '.mitab_plus.txt' in args.file:
    COLS = TAB_PLUS_COLS
ORGANISM_ID_A = 'protein_taxid_1'
ORGANISM_ID_B = 'protein_taxid_2'
PUBMED = 'pmid'

HEADER = COLS
DTYPES = {HEADER[0]: str, HEADER[1]: str, 
          HEADER[-2]: str, HEADER[-1]: str, PUBMED: str,
          ORGANISM_ID_A: str, ORGANISM_ID_B: str}
    
# Conservative filters to apply
INTERACTION_TYPES=['psi-mi:MI:0407(direct interaction)', 
                   'psi-mi:MI:0915(physical association)']
# Currently no filtering of detection methods...
DETECTION_METHODS=[
        'psi-mi:MI:0398(two hybrid pooling approach)',
        'psi-mi:MI:0071(molecular sieving)', 
        'psi-mi:MI:0018(two hybrid)',
        'psi-mi:MI:0019(coimmunoprecipitation)',
        'psi-mi:MI:0059(gst pull down)', 
        'psi-mi:MI:0096(pull down)',
        'psi-mi:MI:0411(enzyme linked immunosorbent assay)',
        'psi-mi:MI:0006(anti bait coimmunoprecipitation)',
        'psi-mi:MI:0089(protein array)',
        'psi-mi:MI:0424(protein kinase assay)',
        'psi-mi:MI:0049(filter binding)',
        'psi-mi:MI:0107(surface plasmon resonance)',
        'psi-mi:MI:0728(gal4 vp16 complementation)',
        'psi-mi:MI:0084(phage display)',
        'psi-mi:MI:0090(protein complementation assay)',
        'psi-mi:MI:0004(affinity chromatography technology)',
        'psi-mi:MI:0065(isothermal titration calorimetry)',
        'psi-mi:MI:0676(tandem affinity purification)',
        'psi-mi:MI:0007(anti tag coimmunoprecipitation)',
        'psi-mi:MI:0114(x-ray crystallography)',
        'psi-mi:MI:0405(competition binding)',
        'psi-mi:MI:0404(comigration in non denaturing gel electrophoresis)',
        'psi-mi:MI:0435(protease assay)',
        'psi-mi:MI:0415(enzymatic study)',
        'psi-mi:MI:0416(fluorescence microscopy)',
        'psi-mi:MI:0045(experimental interaction detection)',
        'psi-mi:MI:1203(split luciferase complementation)',
        'psi-mi:MI:0401(biochemical)',
        'psi-mi:MI:0412(electrophoretic mobility supershift assay)',
        'psi-mi:MI:0047(far western blotting)',
        'psi-mi:MI:0686(unspecified method)',
        'psi-mi:MI:0000(molecular interaction)',
        'psi-mi:MI:0400(affinity technology)',
        'psi-mi:MI:0397(two hybrid array)',
        'psi-mi:MI:0663(confocal microscopy)', 
        'psi-mi:MI:1313(bioid)',
        'psi-mi:MI:0588(three hybrid)',
        'psi-mi:MI:0428(imaging technique)',
        'psi-mi:MI:0030(cross-linking study)',
        'psi-mi:MI:0029(cosedimentation through density gradient)',
        'psi-mi:MI:0109(tap tag coimmunoprecipitation)',
        'psi-mi:MI:0017(classical fluorescence spectroscopy)',
        'psi-mi:MI:0025(copurification)',
        'psi-mi:MI:0077(nuclear magnetic resonance)',
        'psi-mi:MI:0254(genetic interference)',
        'psi-mi:MI:0403(colocalization)',
        'psi-mi:MI:0053(fluorescence polarization spectroscopy)',
        'psi-mi:MI:0963(interactome parallel affinity capture)',
        'psi-mi:MI:0069(mass spectrometry studies of complexes)',
        'psi-mi:MI:0808(comigration in sds page)',
        'psi-mi:MI:0990(cleavage assay)',
        'psi-mi:MI:0998(deubiquitinase assay)',
        'psi-mi:MI:0888(small angle neutron scattering)',
        'psi-mi:MI:0226(ion exchange chromatography)',
        'psi-mi:MI:0061(his pull down)',
        'psi-mi:MI:0028(cosedimentation in solution)',
        'psi-mi:MI:0410(electron tomography)',
        'psi-mi:MI:0055(fluorescent resonance energy transfer)',
        'psi-mi:MI:1356(validated two hybrid)',
        'psi-mi:MI:0826(x ray scattering)',
        'psi-mi:MI:0020(transmission electron microscopy)',
        'psi-mi:MI:0022(colocalization by immunostaining)',
        'psi-mi:MI:0515(methyltransferase assay)',
        'psi-mi:MI:0729(luminescence based mammalian interactome mapping)',
        'psi-mi:MI:1112(two hybrid prey pooling approach)',
        'psi-mi:MI:0427(Identification by mass spectrometry)',
        'psi-mi:MI:1007(glycosylase assay)',
        'psi-mi:MI:0091(chromatography technology)',
        'psi-mi:MI:0115(yeast display)', 'psi-mi:MI:0013(biophysical)',
        'psi-mi:MI:0727(lexa b52 complementation)',
        'psi-mi:MI:0889(acetylation assay)',
        'psi-mi:MI:0889(acetylase assay)',
        'psi-mi:MI:0949(gdp/gtp exchange assay)',
        'psi-mi:MI:1016(fluorescence recovery after photobleaching)',
        'psi-mi:MI:0809(bimolecular fluorescence complementation)',
        'psi-mi:MI:0228(cytoplasmic complementation assay)',
        'psi-mi:MI:1204(split firefly luciferase complementation)',
        'psi-mi:MI:1103(solution state nmr)',
        'psi-mi:MI:1354(lipase assay)', 'psi-mi:MI:0027(cosedimentation)',
        'psi-mi:MI:0040(electron microscopy)',
        'psi-mi:MI:0067(light scattering)',
        'psi-mi:MI:0984(deaminase assay)',
        'psi-mi:MI:0984(deamination assay)',
        'psi-mi:MI:0082(peptide massfingerprinting)',
        'psi-mi:MI:0051(fluorescence technology)',
        'psi-mi:MI:0413(electrophoretic mobility shift assay)',
        'psi-mi:MI:0402(chromatin immunoprecipitation assay)',
        'psi-mi:MI:0892(solid phase assay)',
        'psi-mi:MI:0023(colocalization/visualisation technologies)',
        'psi-mi:MI:0092(protein in situ array)',
        'psi-mi:MI:0075(myc tag coimmunoprecipitation)',
        'psi-mi:MI:0113(western blot)', 'psi-mi:MI:0492(in vitro)',
        'psi-mi:MI:0678(antibody array)',
        'psi-mi:MI:0010(beta galactosidase complementation)',
        'psi-mi:MI:0081(peptide array)',
        'psi-mi:MI:0231(mammalian protein protein interaction trap)',
        'psi-mi:MI:0437(protein three hybrid)',
        'psi-mi:MI:0112(ubiquitin reconstruction)',
        'psi-mi:MI:0419(gtpase assay)',
        'psi-mi:MI:0557(adp ribosylation reaction)',
        'psi-mi:MI:1022(field flow fractionation)',
        'psi-mi:MI:0104(static light scattering)',
        'psi-mi:MI:0434(phosphatase assay)',
        'psi-mi:MI:0203(dephosphorylation reaction)',
        'psi-mi:MI:0997(ubiquitinase assay)',
        'psi-mi:MI:1314(proximity-dependent biotin identification)',
        'psi-mi:MI:1104(solid state nmr)',
        'psi-mi:MI:0038(dynamic light scattering)',
        'psi-mi:MI:0054(fluorescence-activated cell sorting)',
        'psi-mi:MI:0944(mass spectrometry study of hydrogen/deuterium exchange)',
        'psi-mi:MI:0012(bioluminescence resonance energy transfer)',
        'psi-mi:MI:0016(circular dichroism)', 'psi-mi:MI:0493(in vivo)',
        'psi-mi:MI:0423(in-gel kinase assay)',
        'psi-mi:MI:0192(acetylation reaction)',
        'psi-mi:MI:0406(deacetylase assay)',
        'psi-mi:MI:0005(alanine scanning)',
        'psi-mi:MI:0058(genome based prediction)',
        'psi-mi:MI:0213(methylation reaction)',
        'psi-mi:MI:1247(microscale thermophoresis)',
        'psi-mi:MI:0813(proximity ligation assay)',
        'psi-mi:MI:1024(scanning electron microscopy)',
        'psi-mi:MI:0105(structure based prediction)',
        'psi-mi:MI:0827(x-ray tomography)',
        'psi-mi:MI:0566(sumoylation reaction)',
        'psi-mi:MI:0432(one hybrid)', 'psi-mi:MI:0440(saturation binding)',
        'psi-mi:MI:0969(bio-layer interferometry)',
        'psi-mi:MI:0807(comigration in gel electrophoresis)',
        'psi-mi:MI:0011(beta lactamase complementation)',
        'psi-mi:MI:0921(surface plasmon resonance array)',
        'psi-mi:MI:0048(filamentous phage display)',
        'psi-mi:MI:0417(footprinting)',
        'psi-mi:MI:0943(detection by mass spectrometry)',
        'psi-mi:MI:1019(protein phosphatase assay)',
        'psi-mi:MI:0276(blue native page)',
        'psi-mi:MI:2281(deamidation assay)',
        'psi-mi:MI:0021(colocalization by fluorescent probes cloning)'
    ]

# ======================= FUNCTIONS FOR STEP 1 =======================
def get_hpidb_interactions(df_file, positome_filter=True):
    df = df_file.copy()
    # Filter HPIDB data
    if positome_filter:
        print('\tApplying filters...')
        df = df[df['interaction_type'].isin(INTERACTION_TYPES)]
        df = df[df['detection_method'].isin(DETECTION_METHODS)]
    
    # Account for version and formatting changes
    df = df[COLS]
    df[COLS[0]] = df[COLS[0]].str.replace('uniprotkb:', '')
    df[COLS[1]] = df[COLS[1]].str.replace('uniprotkb:', '')
    df[COLS[0]] = df[COLS[0]].str.replace('UNIPROT_AC:', '')
    df[COLS[1]] = df[COLS[1]].str.replace('UNIPROT_AC:', '')
    
    # Leave out incomplete data
    df = df[df[COLS[0]] != '-']
    df = df[df[COLS[1]] != '-']
    df = df[df[ORGANISM_ID_A] != '-']
    df = df[df[ORGANISM_ID_B] != '-']
    df = df[df[PUBMED] != '-']
    df.dropna(subset=[COLS[0], COLS[1], PUBMED], inplace=True)
    
    df[ORGANISM_ID_A] = df[ORGANISM_ID_A].str.replace('taxid:', '')
    df[ORGANISM_ID_B] = df[ORGANISM_ID_B].str.replace('taxid:', '')
    taxidsa = [i.split('(')[0] for i in df[ORGANISM_ID_A]]
    taxidsb = [i.split('(')[0] for i in df[ORGANISM_ID_B]]
    df[ORGANISM_ID_A] = taxidsa
    df[ORGANISM_ID_B] = taxidsb
    
    df[ORGANISM_ID_A] = df[ORGANISM_ID_A].astype(int)
    df[ORGANISM_ID_B] = df[ORGANISM_ID_B].astype(int)
    df.reset_index(drop=True, inplace=True)
    
    return df

def separate_species_interactions(df_hpidb, host_id=None, pathogen_ids=None, confidence=2):
    df = df_hpidb.copy()
    ppi_type = 'inter'
    if host_id == None:
        organism_count = df[ORGANISM_ID_A].append(df[ORGANISM_ID_B]).value_counts()
        # Get main organism ID as most frequent in df for intra-species PPIs
        main_organism = organism_count.loc[organism_count == organism_count.max()].index[0]
    else:
        main_organism = host_id
    if pathogen_ids == None:
        # Get all other organism IDs for inter-species PPIs with main organism
        inter_organisms = organism_count.index.tolist()
        inter_organisms.remove(main_organism)
    else:
        inter_organisms = pathogen_ids
        
    intra_species = None
    inter_species = None
    if ppi_type == 'both' or ppi_type == 'intra':
        print('\tGetting intraspecies interactions for organismID %s'%main_organism)
        intra = df.loc[(df[ORGANISM_ID_A] == main_organism) & (df[ORGANISM_ID_B] == main_organism)]
        intra.reset_index(drop=True, inplace=True)
        intra_species = check_ppi_confidence(intra, level=confidence)
    
    if ppi_type == 'both' or ppi_type == 'inter':
        if len(inter_organisms) < 1:
            print('\tNo inter-species PPIs for %s'%main_organism)
        else:
            inter_species = []
            for organism in inter_organisms:
                print('\tGetting interspecies interactions for organismIDs %s - %s'%(main_organism, organism))
                inter = df.loc[(df[ORGANISM_ID_A] == main_organism) & (df[ORGANISM_ID_B] == organism)]
                inter = inter.append(df.loc[(df[ORGANISM_ID_A] == organism) & (df[ORGANISM_ID_B] == main_organism)], ignore_index=True)
                interspecies = check_ppi_confidence(inter, level=confidence)
                if interspecies.empty == False:
                    inter_species.append(interspecies)
                
    return intra_species, inter_species

def check_ppi_confidence(df_hpidb, level=2):
    df = df_hpidb.copy()
    
    # Get PPIs resetting order such that protein interactions AB and BA are all listed as AB
    ppi = pd.DataFrame([set(p) for p in df[[COLS[0], COLS[1]]].values])
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
        ppi.insert(len(ppi.columns), PUBMED, df.iloc[ppi.index][PUBMED])
        group = ppi.groupby([ppi.columns[0], ppi.columns[1]])[PUBMED].apply(lambda x: x.unique()).reset_index()
        ppi_2 = group[[len(group.iloc[i][PUBMED]) > 1 for i in group.index]]
        if not ppi_2.empty:
            # Reset PPI order in df so pairs are ordered as in ppi_2 pairs for merging
            ppi_2 = ppi_2.rename(columns={ppi_2.columns[0]: COLS[0], ppi_2.columns[1]: COLS[1]})
            pairs = pd.DataFrame([set(p) for p in df[[COLS[0], COLS[1]]].values])
            pairs[pairs.columns[1]] = pairs[pairs.columns[1]].fillna(pairs[pairs.columns[0]])
            df[[COLS[0], COLS[1]]] = pairs[[0,1]].values
            df = df.merge(ppi_2, on=[COLS[0], COLS[1]])
            df.drop(columns=[PUBMED + '_y'], inplace=True)
            df.rename(columns={PUBMED + '_x': PUBMED}, inplace=True)
            df.drop_duplicates(subset=[COLS[0], COLS[1]], inplace=True)
            df.reset_index(drop=True, inplace=True)
        else:
            df = pd.DataFrame()
        return df
    else:
        return df

def map_hpidb_to_uniprot(df_hpidb, include_unreviewed=False):
    df = df_hpidb.copy()

    # Query UniProt mapping
    geneIDs = df[COLS[0]].append(df[COLS[1]]).unique()
    geneIDs_query = str(geneIDs.tolist()).strip('[').strip(']').replace("'", "").replace(',', '')
    url = 'https://www.uniprot.org/uploadlists/'
    params = {
    'from': 'ACC',
    'to': 'ACC',
    'format': 'tab',
    'columns': 'id,sequence,reviewed',
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
        entrez_list = df_uniprot.columns.tolist()[-1]
        df_uniprot.rename(columns={entrez_list: 'ProtID', 'Entry': 'ProteinID'}, inplace=True)
        
        # Remove unreviewed entries
        if include_unreviewed == False:
            df_uniprot = df_uniprot[df_uniprot['Status'] == 'reviewed']
            df_uniprot.reset_index(inplace=True, drop=True)
            df_uniprot = df_uniprot.drop(columns=['Status'])
        
        # Map IDs to HPIDB dataset, remove unmapped, and rename columns
        mapped = df.copy()
        # For ProteinIDs with more than one ID
        df_uniprot['ProtID'] = df_uniprot['ProtID'].str.split(',')
        df_uniprot = df_uniprot.explode('ProtID', ignore_index=True)
        
        refdict = pd.Series(df_uniprot['ProteinID'].values, index=df_uniprot['ProtID']).to_dict()
        
        mapped[COLS[0]] = mapped[COLS[0]].map(refdict)
        mapped[COLS[1]] = mapped[COLS[1]].map(refdict)
        mapped.dropna(subset=[COLS[0], COLS[1]], inplace=True)
        mapped.rename(columns={COLS[0]:'Protein A', COLS[1]:'Protein B'}, inplace=True)
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

# ======================= FUNCTIONS FOR STEP 2 =======================
def read_fasta(filename):
     df = pd.read_csv(filename, sep='\t', header=None)
     prot = df.iloc[::2, :].reset_index(drop=True)
     seq = df.iloc[1::2, :].reset_index(drop=True)
     df = pd.concat([prot, seq], axis=1)
     df.columns = [0, 1]
     return df

def run_cdhit(fasta_filename, cdhit='cd-hit', threshold=0.6):
    if threshold == 1.0:
        print('\tNo sequence clustering required...')
        return read_fasta(fasta_filename)
    elif threshold < 1.0 and threshold >= 0.7:
        words = 5
    elif threshold < 0.7 and threshold >= 0.6:
        words = 4
    elif threshold < 0.6 and threshold >= 0.5:
        words = 3
    elif threshold < 0.5 and threshold >= 0.4:
        words = 2
    else:
        return read_fasta(fasta_filename)
    try:
        cmd = '%s -i %s -o %s.new -c %s -n %s'%(cdhit, fasta_filename, fasta_filename, threshold, words)
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        print(result.stdout)
        print(result.stderr)
        time.sleep(1)
        
        # Read new .fasta with homology reduced sequences
        df = read_fasta(fasta_filename + '.new')
        # Replace .fasta with original name
        os.remove(fasta_filename + '.new')
        os.remove(fasta_filename + '.new.clstr')
        df.to_csv(fasta_filename, sep='\n', header=None, index=False)
        
        return df
    
    except Exception as e:
        print(e)
        print('\tCD-HIT not working, returning original .fasta...')
        return read_fasta(fasta_filename)

def remove_homology_ppi(df_ppi, df_seq):
    ppi = df_ppi.copy()
    fasta = df_seq.copy()
    # Get proteins
    fasta_prots = fasta[fasta.columns[0]].str.replace('>', '').unique()
    ppi_prots = pd.Series(ppi[ppi.columns[0]].append(ppi[ppi.columns[1]]).unique())
    # Proteins in PPIs without a sequence in .fasta
    no_seq = ppi_prots[~ppi_prots.isin(fasta_prots)]
    # Remove PPIs containing proteins that have no sequence
    remove_ppi = ppi[(ppi[ppi.columns[0]].isin(no_seq.values)) | (ppi[ppi.columns[1]].isin(no_seq.values))]
    ppi.drop(index=remove_ppi.index, inplace=True)
    ppi.reset_index(drop=True, inplace=True)
    
    return ppi, fasta
    

# ======================= FUNCTIONS FOR STEP 3 =======================
def remove_redundant_pairs(df_ppi):
    df = df_ppi.copy()
    # Get only unique PPIs (using set automatically sorts AB and BA such that they will all be AB)
    pairs = pd.DataFrame([set(p) for p in df[df.columns[:2]].values])
    # Fill in for self-interacting proteins
    if len(pairs.columns) > 1:
        pairs[pairs.columns[1]] = pairs[pairs.columns[1]].fillna(pairs[pairs.columns[0]])
        pairs.drop_duplicates(inplace=True)
        pairs.reset_index(drop=True, inplace=True)
    else:
        return df
    
    return pairs

def generate_negative_interactions(df_pos, diff_locations=False):
    df = df_pos.copy()
    
    # Consider for generating negative PPIs for inter-species
    organisms = df[ORGANISM_ID_A].append(df[ORGANISM_ID_B]).unique()
    if len(organisms) == 2:
        proteins_organism_A = df[df[ORGANISM_ID_A] == organisms[0]]['Protein A'].values
        proteins_organism_A = np.unique(np.append(proteins_organism_A, df[df[ORGANISM_ID_B] == organisms[0]]['Protein B'].values))
        proteins_organism_B = df[df[ORGANISM_ID_A] == organisms[1]]['Protein A'].values
        proteins_organism_B = np.unique(np.append(proteins_organism_B, df[df[ORGANISM_ID_B] == organisms[1]]['Protein B'].values))
    
        # Consider if negatives unable to generate for interspecies when only 1 protein available from one of organisms
        if len(proteins_organism_A) < 2 or len(proteins_organism_B) < 2:
            print('\tNot enough inter-species proteins/PPIs to create negative PPIs...')
            return df, pd.DataFrame()
    
    # Remove redundant and sort AB order of PPI pairs
    df = remove_redundant_pairs(df)
    
    # Get all proteins for sampling
    sample_proteins = df[df.columns[0]].append(df[df.columns[1]]).unique()
    if len(sample_proteins) == 1:
        return df, pd.DataFrame()
    
    # Get protein location info if required
    if diff_locations:
        df_uniprot = get_protein_locations(sample_proteins)
        if df_uniprot.empty or df.shape[0] == 1:
            print('\tUnable to retrieve protein subcellular locations from UniProt...generating negatives otherwise')
            diff_locations = False
        else:
            sample_proteins = df_uniprot['Protein'].unique()
    
    # Generate negative pairs from proteins found in positives
    if df.shape[0] == 1 & (df[df.columns[0]].values != df[df.columns[1]].values)[0]:
        i = np.random.randint(0,2)
        df_neg = df.copy()
        df_neg[i] = df[df.columns[i-1]].values
        return df, df_neg
    
    df_neg = pd.DataFrame()
    generator = np.random.default_rng()
    while (df_neg.shape[0] < df.shape[0]):
        # Generate random pairs
        df_neg = df_neg.append(pd.DataFrame(generator.choice(sample_proteins, size=df.shape)), ignore_index=True)
        # Remove redundant and sort AB order of PPI pairs
        df_neg = remove_redundant_pairs(df_neg)
        df_neg_rev = pd.DataFrame({0: df_neg[1], 1: df_neg[0]})
        
        # Get pairs found in positive PPIs and remove from negatives
        in_pos = df.merge(df_neg)
        in_pos_rev = df.merge(df_neg_rev)
        in_pos_rev = pd.DataFrame({0: in_pos_rev[1], 1: in_pos_rev[0]})
        in_pos = in_pos.append(in_pos_rev)
        df_neg = df_neg.append(in_pos).drop_duplicates(keep=False)
        
        # Remove intra-species PPIs for negative inter-species pairs
        if len(organisms) == 2:
            df_orgA = df_neg[~(df_neg[df_neg.columns[0]].isin(proteins_organism_A)) & (df_neg[df_neg.columns[1]].isin(proteins_organism_A))]
            df_orgB = df_neg[~(df_neg[df_neg.columns[0]].isin(proteins_organism_B)) & (df_neg[df_neg.columns[1]].isin(proteins_organism_B))]
            df_neg = df_neg.append(df_orgA.append(df_orgB, ignore_index=True))
            df_neg = remove_redundant_pairs(df_neg)
            
        # Check if generated protein pairs have different subcellular locations
        if diff_locations:
            group = df_uniprot.groupby(['Protein'])['Locations'].apply(lambda x: x.unique())
            # Keep negative PPIs containing proteins with location info
            df_neg = df_neg[(df_neg[df_neg.columns[0]].isin(group.index)) & (df_neg[df_neg.columns[1]].isin(group.index))].reset_index(drop=True)
            # Keep negative PPIs where proteins are not found in similar locations
            df_neg = df_neg[~pd.Series([any(group.loc[df_neg[df_neg.columns[0]][i]] == group.loc[df_neg[df_neg.columns[1]][i]]) for i in df_neg.index])].reset_index(drop=True)
            df_neg = remove_redundant_pairs(df_neg)
        # Remove redundant and sort AB order of PPI pairs
        df_neg = remove_redundant_pairs(df_neg)
        
    # Trim negatives if larger than positives
    if df_neg.shape[0] > df.shape[0]:
        df_neg = df_neg[0:df.shape[0]]
    
    return df, df_neg

def get_protein_locations(proteins):
    proteins_query = str(proteins.tolist()).strip('[').strip(']').replace("'", "").replace(',', '')
    url = 'https://www.uniprot.org/uploadlists/'
    params = {
    'from': 'ACC+ID',
    'to': 'ACC',
    'format': 'tab',
    'columns': 'id,comment(SUBCELLULAR LOCATION)',
    'query': proteins_query,
    }
    for x in range(0, 3):
        try:
            print('\tGetting protein subcellular location info from UniProt...')
            # Request UniProt info for given proteins
            data = urllib.parse.urlencode(params)
            data = data.encode('utf-8')
            req = urllib.request.Request(url, data)
            with urllib.request.urlopen(req) as webpage:
                response = webpage.read().decode('utf-8')
            if response == '':
                print('\tNo UniProt response.')
            else:
                break
        except:
            pass
    if response == '':
        return pd.DataFrame()
    
    df_uniprot = pd.read_csv(StringIO(response), sep='\t', dtype=str)
    query = df_uniprot.columns.tolist()[-1]
    df_uniprot.rename(columns={query: 'Query', 'Subcellular location [CC]': 'Locations', 'Entry': 'Protein'}, inplace=True)
    # Remove proteins without location info
    df_uniprot.dropna(inplace=True)
    df_uniprot.reset_index(drop=True, inplace=True)
    df_uniprot.drop(columns=['Query'], inplace=True)
    # Format location info
    df_uniprot['Locations'] = df_uniprot['Locations'].str.replace('SUBCELLULAR LOCATION: ', '')
    df_uniprot['Locations'] = df_uniprot['Locations'].apply(lambda loc: re.sub(r'[\{[].*?[}\]]', '', loc).split('Note')[0])
    df_uniprot['Locations'] = df_uniprot['Locations'].apply(lambda loc: re.sub(r' +', '', loc))
    df_uniprot['Locations'] = df_uniprot['Locations'].apply(lambda loc: re.sub(r'[.;]', ',', loc))
    df_uniprot['Locations'] = df_uniprot['Locations'].str.split(',')
    df_uniprot['Locations'] = df_uniprot['Locations'].apply(lambda loc: list(filter(None, loc)))
    df_uniprot['Locations'] = df_uniprot['Locations'].map(tuple)
    
    return df_uniprot

# ======================= FUNCTIONS FOR STEP 4 =======================
def save_ppi_data(save_location, filename, df_pos, df_neg, df_fasta, models=[], kfolds=0, all_to_all=False, park_marcotte=0):
    pos = df_pos.copy()
    neg = df_neg.copy()
    fasta = df_fasta.copy()
    
    # Add labels
    pos.insert(pos.shape[1], pos.shape[1], np.ones(pos.shape[0], dtype=int))
    pos = pos.sort_values(by=[pos.columns[0], pos.columns[1]], ignore_index=True)
    neg.insert(neg.shape[1], neg.shape[1], np.zeros(neg.shape[0], dtype=int))
    neg = neg.sort_values(by=[neg.columns[0], neg.columns[1]], ignore_index=True)
    df = pos.append(neg, ignore_index=True)
    
    # Save in save_location
    df.to_csv(save_location + filename + '_interactions.tsv', sep='\t', header=None, index=False)
    fasta.to_csv(save_location + filename + '_sequences.fasta', sep='\n', header=None, index=False)
    
    # Format for PPI prediction methods and save
    format_ppi_data(save_location, filename, df, fasta, methods=models, k_folds=kfolds)
    
    # Save all-to-all PPIs
    if all_to_all:
        print('\tSaving all-to-all PPIs...')
        proteins = df[df.columns[0]].append(df[df.columns[1]]).unique()
        df_all = pd.DataFrame(list(combinations_with_replacement(proteins, 2)))
        df_all.insert(df_all.shape[1], df_all.shape[1], np.ones(df_all.shape[0], dtype=int))
        df_all = df_all.sort_values(by=[df_all.columns[0], df_all.columns[1]], ignore_index=True)
        filename = filename + '_all'
        df_all.to_csv(save_location + filename + '_interactions.tsv', sep='\t', header=None, index=False)
        format_ppi_data(save_location, filename, df_all, fasta, methods=models, k_folds=0)
    
    if park_marcotte > 0:
        pm_save_location = save_location + 'PARK_MARCOTTE/'
        if not os.path.exists(save_location + 'PARK_MARCOTTE/'):
                os.mkdir(save_location + 'PARK_MARCOTTE/')
                
    if park_marcotte > 0:
        pm_test_c1 = pd.DataFrame()
        pm_test_c2 = pd.DataFrame()
        pm_test_c3 = pd.DataFrame()
        pm_train = pd.DataFrame()
    # Create and save Park & Marcotte sets
    for i in range(park_marcotte):
        
        print('\nSaving Park & Marcotte set %s...'%i)
        train, test_c1, test_c2, test_c3 = park_marcotte_subsets(df, train_size=0.7)
        c1_fasta = fasta[fasta[fasta.columns[0]].str.replace('>', '').isin(test_c1[test_c1.columns[0]].append(test_c1[test_c1.columns[1]]).unique())]
        c2_fasta = fasta[fasta[fasta.columns[0]].str.replace('>', '').isin(test_c2[test_c2.columns[0]].append(test_c2[test_c2.columns[1]]).unique())]
        c3_fasta = fasta[fasta[fasta.columns[0]].str.replace('>', '').isin(test_c3[test_c3.columns[0]].append(test_c3[test_c3.columns[1]]).unique())]
        train.reset_index(drop=True, inplace=True)
        test_c1.reset_index(drop=True, inplace=True)
        test_c2.reset_index(drop=True, inplace=True)
        test_c3.reset_index(drop=True, inplace=True)
        c1_fasta.reset_index(drop=True, inplace=True)
        c2_fasta.reset_index(drop=True, inplace=True)
        c3_fasta.reset_index(drop=True, inplace=True)
        
        # Save in save_location
        train.to_csv(pm_save_location + filename + '_PM%s_train'%i + '_interactions.tsv', sep='\t', header=None, index=False)
        fasta.to_csv(pm_save_location + filename + '_PM%s_train'%i + '_sequences.fasta', sep='\n', header=None, index=False)
        test_c1.to_csv(pm_save_location + filename + '_PM%s_test_c1'%i + '_interactions.tsv', sep='\t', header=None, index=False)
        test_c2.to_csv(pm_save_location + filename + '_PM%s_test_c2'%i + '_interactions.tsv', sep='\t', header=None, index=False)
        test_c3.to_csv(pm_save_location + filename + '_PM%s_test_c3'%i + '_interactions.tsv', sep='\t', header=None, index=False)
        c1_fasta.to_csv(pm_save_location + filename + '_PM%s_test_c1'%i + '_sequences.fasta', sep='\n', header=None, index=False)
        c2_fasta.to_csv(pm_save_location + filename + '_PM%s_test_c2'%i + '_sequences.fasta', sep='\n', header=None, index=False)
        c3_fasta.to_csv(pm_save_location + filename + '_PM%s_test_c3'%i + '_sequences.fasta', sep='\n', header=None, index=False)
        
        # Save formatted for PPI prediction methods
        print('\tSaving PM train set %s...'%i)
        format_ppi_data(pm_save_location, filename + '_PM%s_train'%i, train, fasta, methods=models, k_folds=0)
        print('\tSaving PM C1 test set %s...'%i)
        format_ppi_data(pm_save_location, filename + '_PM%s_test_c1'%i, test_c1, c1_fasta, methods=models, k_folds=0)
        print('\tSaving PM C2 test set %s...'%i)
        format_ppi_data(pm_save_location, filename + '_PM%s_test_c2'%i, test_c2, c2_fasta, methods=models, k_folds=0)
        print('\tSaving PM C3 test set %s...'%i)
        format_ppi_data(pm_save_location, filename + '_PM%s_test_c3'%i, test_c3, c3_fasta, methods=models, k_folds=0)
        
        pm_test_c1 = pm_test_c1.append(test_c1)
        pm_test_c2 = pm_test_c2.append(test_c2)
        pm_test_c3 = pm_test_c3.append(test_c3)
        pm_train = pm_train.append(train)
        
    if park_marcotte > 0:
        pm_train = remove_redundant_pairs(pm_train)
        pm_test_c1 = remove_redundant_pairs(pm_test_c1)
        pm_test_c2 = remove_redundant_pairs(pm_test_c2)
        pm_test_c3 = remove_redundant_pairs(pm_test_c3)
        pm_train.to_csv(pm_save_location + filename + '_PM_total_train_interactions.tsv', sep='\t', header=None, index=False)
        pm_test_c1.to_csv(pm_save_location + filename + '_PM_total_test_c1'%i + '_interactions.tsv', sep='\t', header=None, index=False)
        pm_test_c2.to_csv(pm_save_location + filename + '_PM_total_test_c2'%i + '_interactions.tsv', sep='\t', header=None, index=False)
        pm_test_c3.to_csv(pm_save_location + filename + '_PM_total_test_c3'%i + '_interactions.tsv', sep='\t', header=None, index=False)


def format_ppi_data(location, filename, df_ppi, df_fasta, methods=[], k_folds=0):
    ppi = df_ppi.copy()
    fasta = df_fasta.copy()

    create_cv_subsets(location, filename, ppi, fasta, k_splits=k_folds)
    # Format data as per model input
    for m in methods:
        if m.lower() == 'pipr':
            file, df, seq = convert_pipr(location, filename, ppi, fasta, save=True)
            create_cv_subsets(location, file, df, seq, k_splits=k_folds)
        if m.lower() == 'sprint':
            file, df, seq = convert_sprint(location, filename, ppi, fasta, save=True)
            create_cv_subsets(location, file, df, seq, k_splits=k_folds)
        if m.lower() == 'deepfe':
            file, df, seq = convert_deepfe(location, filename, ppi, fasta, save=True)
            create_cv_subsets(location, file, df, seq, k_splits=k_folds)
        if m.lower() == 'dppi':
            file, df, seq = convert_dppi(location, filename, ppi, fasta, save=True)
            create_cv_subsets(location, file, df, seq, k_splits=k_folds)
        if m.lower() not in methods:
            print('\t%s data formatting is not available\n'%m)
        
def convert_pipr(save_location, file, df_ppi, df_fasta, save=False):
    if save:
        if not os.path.exists(save_location + 'PIPR_DATA/'):
            os.mkdir(save_location + 'PIPR_DATA/')
    ppi_pipr = df_ppi.copy()
    fasta_pipr = df_fasta.copy()
    filename = file + '_PIPR'
    print("\tFormatting dataset for PIPR...")
    ppi_pipr.columns = ['v1', 'v2', 'label']
    if save:
        ppi_pipr.to_csv(save_location + 'PIPR_DATA/' + filename + '_interactions.tsv', sep='\t', index=False)
        fasta_pipr[fasta_pipr.columns[0]] = fasta_pipr[fasta_pipr.columns[0]].str.replace('>', '')
        fasta_pipr.to_csv(save_location + 'PIPR_DATA/' + filename + '_sequences.fasta', sep='\t', index=False, header=False)
    
    return filename, ppi_pipr, fasta_pipr
    
def convert_sprint(save_location, file, df_ppi, df_fasta, save=False):
    if save:
        if not os.path.exists(save_location + 'SPRINT_DATA/'):
            os.mkdir(save_location + 'SPRINT_DATA/')
    ppi_sprint = df_ppi.copy()
    fasta_sprint = df_fasta.copy()
    filename = file + '_SPRINT'
    print("\tFormatting dataset for SPRINT...")
    pos = ppi_sprint[ppi_sprint[ppi_sprint.columns[-1]] == 1]
    neg = ppi_sprint[ppi_sprint[ppi_sprint.columns[-1]] == 0]
    if save:
        pos.to_csv(save_location + 'SPRINT_DATA/' + filename + '_pos_interactions.txt', columns=list(pos.columns[:2]), sep=' ', index=False, header=False)
        if neg.empty != True:
            neg.to_csv(save_location + 'SPRINT_DATA/' + filename + '_neg_interactions.txt', columns=list(neg.columns[:2]), sep=' ', index=False, header=False)
        fasta_sprint.to_csv(save_location + 'SPRINT_DATA/' + filename + '_sequences.fasta', sep='\n', index=False, header=False)
    
    return filename, ppi_sprint, fasta_sprint
        
def convert_deepfe(save_location, file, df_ppi, df_fasta, save=False):
    if save:
        if not os.path.exists(save_location + 'DEEPFE_DATA/'):
            os.mkdir(save_location + 'DEEPFE_DATA/')
        if not os.path.exists(save_location + 'DEEPFE_DATA/' + file + '_DEEPFE/'):
            os.mkdir(save_location + 'DEEPFE_DATA/' + file + '_DEEPFE/')
    ppi_deepfe = df_ppi.copy()
    fasta_deepfe = df_fasta.copy()
    filename = file + '_DEEPFE'
    print("\tFormatting dataset for DEEPFE...")
    pos = ppi_deepfe[ppi_deepfe[ppi_deepfe.columns[-1]] == 1]
    neg = ppi_deepfe[ppi_deepfe[ppi_deepfe.columns[-1]] == 0]
    posA = '>' + pos[pos.columns[0]]
    posB = '>' + pos[pos.columns[1]]
    negA = '>' + neg[neg.columns[0]]
    negB = '>' + neg[neg.columns[1]]

    # Map proteins to .fasta sequence format
    fasta_deepfe[fasta_deepfe.columns[-1]] = fasta_deepfe[fasta_deepfe.columns[0]] + '\n' + fasta_deepfe[fasta_deepfe.columns[-1]]
    refdictseq = pd.Series(fasta_deepfe[fasta_deepfe.columns[-1]].values, index=fasta_deepfe[fasta_deepfe.columns[0]]).to_dict()
    posA = posA.map(refdictseq)
    posB = posB.map(refdictseq)
    negA = negA.map(refdictseq)
    negB = negB.map(refdictseq)

    # Write to files
    if save:
        posA.to_csv(save_location + 'DEEPFE_DATA/' + filename + '/' + filename + '_pos_ProteinA.fasta', sep='\n', index=False, header=False, quoting=csv.QUOTE_NONE, escapechar=" ")
        posB.to_csv(save_location + 'DEEPFE_DATA/' + filename + '/' + filename + '_pos_ProteinB.fasta', sep='\n', index=False, header=False, quoting=csv.QUOTE_NONE, escapechar=" ")
        if negA.empty != True and negB.empty != True:
            negA.to_csv(save_location + 'DEEPFE_DATA/' + filename + '/' + filename + '_neg_ProteinA.fasta', sep='\n', index=False, header=False, quoting=csv.QUOTE_NONE, escapechar=" ")
            negB.to_csv(save_location + 'DEEPFE_DATA/' + filename + '/' + filename + '_neg_ProteinB.fasta', sep='\n', index=False, header=False, quoting=csv.QUOTE_NONE, escapechar=" ")

    pos = pd.DataFrame(data={0: posA, 1: posB, 2: np.ones(posA.shape, dtype=int)})
    neg = pd.DataFrame(data={0: negA, 1: negB, 2: np.zeros(negA.shape, dtype=int)})
    df_deepfe = pos.append(neg, ignore_index=True)
    df_fasta_deepfe = df_fasta.copy()
        
    return filename, df_deepfe, df_fasta_deepfe

def convert_dppi(save_location, file, df_ppi, df_fasta, save=False):
    if save:
        if not os.path.exists(save_location + 'DPPI_DATA/'):
            os.mkdir(save_location + 'DPPI_DATA/')
    ppi_dppi = df_ppi.copy()
    fasta_dppi = df_fasta.copy()
    filename = file + '_DPPI'
    print("\tFormatting dataset for DPPI...")
    if save:
        # DATA.csv
        ppi_dppi.to_csv(save_location + 'DPPI_DATA/' + filename + '.csv', index=False, header=None)
        # DATA.node
        proteins = pd.DataFrame(ppi_dppi[ppi_dppi.columns[0]].append(ppi_dppi[ppi_dppi.columns[1]]).reset_index(drop=True).unique())
        proteins.to_csv(save_location + 'DPPI_DATA/' + filename + '.node', sep='\n', index=False, header=False)
        # DATA/ protein fasta files ***NOTE: BLAST still required for all protein fasta files to get PSSM (replace .txt)***
        if not os.path.exists(save_location + 'DPPI_DATA/' + filename + '/'):
            os.mkdir(save_location + 'DPPI_DATA/' + filename + '/')
        fasta_dppi[fasta_dppi.columns[-1]] = fasta_dppi[fasta_dppi.columns[0]] + '\n' + fasta_dppi[fasta_dppi.columns[-1]]
        fasta_dppi[fasta_dppi.columns[0]] = fasta_dppi[fasta_dppi.columns[0]].str.replace('>', '')
        for p in range(0, fasta_dppi.shape[0]):
            with open(save_location + 'DPPI_DATA/' + filename + '/' + str(fasta_dppi[fasta_dppi.columns[0]][p]) + '.txt', 'w') as f:
                f.write(fasta_dppi[fasta_dppi.columns[-1]][p])
    else:
        fasta_dppi[fasta_dppi.columns[-1]] = fasta_dppi[fasta_dppi.columns[0]] + '\n' + fasta_dppi[fasta_dppi.columns[-1]]
        fasta_dppi[fasta_dppi.columns[0]] = fasta_dppi[fasta_dppi.columns[0]].str.replace('>', '')
    
    return filename, ppi_dppi, fasta_dppi
    
def create_cv_subsets(save_location, filename, df_ppi, df_fasta, k_splits=5):
    if k_splits == 0 or k_splits == 1 or k_splits > df_ppi.shape[0]:
        #print('No CV subsets...k_splits = %s, PPIs = %s'%(k_splits, df_ppi.shape[0]))
        return
    
    if not os.path.exists(save_location + 'CV_SET/'):
        os.mkdir(save_location + 'CV_SET/')
    
    df = df_ppi.copy()
    
    # Consider formatting
    sep = '\t'
    keep_labels = True
    separate_pos_neg = False
    header = None
    extension = '.tsv'
    if 'SPRINT' in filename:
        keep_labels = False
        separate_pos_neg =True
        sep = ' '
        extension = '.txt'
    elif 'DEEPFE' in filename:
        keep_labels = False
        separate_pos_neg =True
        sep = '\n'
        extension = '.fasta'
    elif 'PIPR' in filename:
        header = list(df.columns)
    elif 'DPPI' in filename:
        sep = ','
        extension = '.csv'
    
    kf = StratifiedKFold(n_splits=k_splits)
    fold = 0
    for train_index, test_index in kf.split(df[df.columns[:2]], df[df.columns[-1]]):
        train, test = df.iloc[train_index].reindex(), df.iloc[test_index].reindex()
        
        
        pos_train, pos_test = train[train[train.columns[-1]] == 1], test[test[test.columns[-1]] == 1]
        neg_train, neg_test = train[train[train.columns[-1]] == 0], test[test[test.columns[-1]] == 0]
        
        if keep_labels:
            cols = list(df.columns)
        else:
            cols = list(df.columns[:2])
            
        if not os.path.exists(save_location + 'CV_SET/' + filename + '/'):
            os.mkdir(save_location + 'CV_SET/' + filename + '/')
        
        if separate_pos_neg:
            if 'DEEPFE' in filename:
                if not os.path.exists(save_location + 'CV_SET/' + filename + '/' + filename + '_train-' + str(fold) + '/'):
                    os.mkdir(save_location + 'CV_SET/' + filename + '/' + filename + '_train-' + str(fold) + '/')
                if not os.path.exists(save_location + 'CV_SET/' + filename + '/' + filename + '_test-' + str(fold) + '/'):
                    os.mkdir(save_location + 'CV_SET/' + filename + '/' + filename + '_test-' + str(fold) + '/')
                pos_train.to_csv(save_location + 'CV_SET/' + filename + '/' + filename + '_train-' + str(fold) + '/' + filename + 'pos_ProteinA_train-' + str(fold) + extension, columns=[cols[0]], sep=sep, header=header, index=False, quoting=csv.QUOTE_NONE, escapechar=" ")
                pos_train.to_csv(save_location + 'CV_SET/' + filename + '/' + filename + '_train-' + str(fold) + '/' + filename + 'pos_ProteinB_train-' + str(fold) + extension, columns=[cols[1]], sep=sep, header=header, index=False, quoting=csv.QUOTE_NONE, escapechar=" ")
                pos_test.to_csv(save_location + 'CV_SET/' + filename + '/' + filename + '_test-' + str(fold) + '/' + filename + 'pos_ProteinA_test-' + str(fold) + extension, columns=[cols[0]], sep=sep, header=header, index=False, quoting=csv.QUOTE_NONE, escapechar=" ")
                pos_test.to_csv(save_location + 'CV_SET/' + filename + '/' + filename + '_test-' + str(fold) + '/' + filename + 'pos_ProteinB_test-' + str(fold) + extension, columns=[cols[1]], sep=sep, header=header, index=False, quoting=csv.QUOTE_NONE, escapechar=" ")
                neg_train.to_csv(save_location + 'CV_SET/' + filename + '/' + filename + '_train-' + str(fold) +  '/' +  filename + 'neg_ProteinA_train-' + str(fold) + extension, columns=[cols[0]], sep=sep, header=header, index=False, quoting=csv.QUOTE_NONE, escapechar=" ")
                neg_train.to_csv(save_location + 'CV_SET/' + filename + '/' + filename + '_train-' + str(fold) +  '/' +  filename + 'neg_ProteinB_train-' + str(fold) + extension, columns=[cols[1]], sep=sep, header=header, index=False, quoting=csv.QUOTE_NONE, escapechar=" ")
                neg_test.to_csv(save_location + 'CV_SET/' + filename + '/' + filename + '_test-' + str(fold) +  '/' +  filename + 'neg_ProteinA_test-' + str(fold) + extension, columns=[cols[0]], sep=sep, header=header, index=False, quoting=csv.QUOTE_NONE, escapechar=" ")
                neg_test.to_csv(save_location + 'CV_SET/' + filename + '/' + filename + '_test-' + str(fold) +  '/' +  filename + 'neg_ProteinB_test-' + str(fold) + extension, columns=[cols[1]], sep=sep, header=header, index=False, quoting=csv.QUOTE_NONE, escapechar=" ")
            else:
                pos_train.to_csv(save_location + 'CV_SET/' + filename + '/' + filename + '_pos_train-' + str(fold) + extension, columns=cols, sep=sep, header=header, index=False)
                pos_test.to_csv(save_location + 'CV_SET/' + filename + '/' + filename + '_pos_test-' + str(fold) + extension, columns=cols, sep=sep, header=header, index=False)
                neg_train.to_csv(save_location + 'CV_SET/' + filename + '/' +  filename + '_neg_train-' + str(fold) + extension, columns=cols, sep=sep, header=header, index=False)
                neg_test.to_csv(save_location + 'CV_SET/' + filename + '/' +  filename + '_neg_test-' + str(fold) + extension, columns=cols, sep=sep, header=header, index=False)
        else:
            df_train = pos_train.append(neg_train, ignore_index=True)
            df_test = pos_test.append(neg_test, ignore_index=True)
            df_train.to_csv(save_location + 'CV_SET/' + filename + '/' + filename + '_train-' + str(fold) + extension, columns=cols, sep=sep, header=header, index=False)
            df_test.to_csv(save_location + 'CV_SET/' + filename + '/' + filename + '_test-' + str(fold) + extension, columns=cols, sep=sep, header=header, index=False)
            if 'DPPI' in filename:
                prot_train = pd.DataFrame(df_train[df_train.columns[0]].append(df_train[df_train.columns[1]]).unique())
                prot_test = pd.DataFrame(df_test[df_test.columns[0]].append(df_test[df_test.columns[1]]).unique())
                prot_train.to_csv(save_location + 'CV_SET/' + filename + '/' + filename + '_train-' + str(fold) + '.node', header=None, index=False)
                prot_test.to_csv(save_location + 'CV_SET/' + filename + '/' + filename + '_test-' + str(fold) + '.node', header=None, index=False)
        fold += 1
    print("\tCross-validation subsets created!")

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
        
        # Remove redundant and sort AB order of PPI pairs
        df_neg = remove_redundant_pairs(df_neg)
        
    # Trim negatives if larger than positives
    if df_neg.shape[0] > (test_pos.shape[0] - test_neg.shape[0]):
        df_neg = df_neg[0:(test_pos.shape[0] - test_neg.shape[0])]
        
    df_test_balanced = df_test.append(df_neg)
    df_test_balanced.reset_index(drop=True, inplace=True)
    
    return df_test_balanced

if __name__ == "__main__":
    # Display args
    print('\nPreprocessing HPIDB with the following args:\n', args)
    start = time.time()
    
    if not os.path.exists(args.results):
        os.mkdir(args.results)
    
    print('\nReading', args.file)
    df = pd.read_csv(args.file, sep='\t', usecols=HEADER, dtype=DTYPES)
    print('\t%s PPIs'%df.shape[0])
    
    print('\nCleaning HPIDB data...')
    df_pos = get_hpidb_interactions(df, positome_filter=args.filter)
    print('\t%s PPIs'%df_pos.shape[0])
    
    print('\nOrganizing species-specific interactions...')
    df_intra, df_inter = separate_species_interactions(df_pos, host_id=args.host_id, pathogen_ids=args.pathogen_id, confidence=args.confidence_level)
    
    '''
    # Get intra-species PPIs
    if args.type == 'both' or args.type == 'intra':
        print('\n===== Working on intra-species interactions... =====')
        print('\nMapping HPIDB entries to UniProt database...')
        if df_intra.empty:
            print('\tNo intra-species data obtained...')
        else:
            try:
                print('\t%s PPIs'%df_intra.shape[0])
                df_intra_mapped, df_intra_fasta_mapped = map_hpidb_to_uniprot(df_intra, include_unreviewed=args.unreviewed)
                if df_intra_mapped.empty or df_intra_fasta_mapped.empty:
                    print('\tNo intra-species data obtained...')
                else:
                    print('\t%s mapped PPIs'%df_intra_mapped.shape[0])
                    organisms = df_intra_mapped[ORGANISM_ID_A].append(df_intra_mapped[ORGANISM_ID_B]).unique()
                    filename = FILENAME + '_ID_' + str(organisms[0])
                    
                    # Save for CD-HIT to read and run
                    df_intra_mapped.to_csv(args.results + filename + '_interactions.tsv', columns=['Protein A', 'Protein B'], sep='\t', header=None, index=False)
                    df_intra_fasta_mapped.to_csv(args.results + filename + '_sequences.fasta', sep='\n', header=None, index=False)
                    
                    print('\nRunning CD-HIT...')
                    df_intra_fasta_reduced = run_cdhit(args.results + filename + '_sequences.fasta', cdhit=args.cdhit, threshold=args.sequence_identity)
                    df_intra_pos, df_intra_fasta_final = remove_homology_ppi(df_intra_mapped, df_intra_fasta_reduced)
                    print('\t%s positive PPIs'%df_intra_pos.shape[0])
                    
                    print('\nGenerating negative PPIs...')
                    print('\t%s proteins available'%df_intra_pos[df_intra_pos.columns[0]].append(df_intra_pos[df_intra_pos.columns[1]]).unique().shape[0])
                    df_intra_pos, df_intra_neg = generate_negative_interactions(df_intra_pos, diff_locations=args.diff_subcell_local)
                    if df_intra_neg.shape[0] == 0:
                        print('\tNo negatives generated...')
                    else:
                        print('\nSaving PPI dataset...%s'%filename)
                        df_intra_pos = df_intra_pos[df_intra_pos.columns[:2]]
                        df_intra_pos.columns = df_intra_neg.columns
                        df_intra_neg = df_intra_neg[df_intra_neg.columns[:2]]
                        save_ppi_data(args.results, filename, df_intra_pos, df_intra_neg, df_intra_fasta_final, models=args.models, kfolds=args.kfolds, all_to_all=args.all_to_all, park_marcotte=args.park_marcotte)
                        print('\nTime %s seconds...'%round(time.time() - start, 2))
            except Exception as e:
                print(e)
                exit()
    '''
    # Get inter-species PPIs
    if df_inter != None:
        print('\n===== Working on inter-species interactions... =====')
        for df in range(0, len(df_inter)):
            time.sleep(1)
            if df_inter[df].empty:
                continue
            try:
                df_current = df_inter[df]
                print('\t%s PPIs'%df_current.shape[0])
                organisms = df_current[ORGANISM_ID_A].append(df_current[ORGANISM_ID_B]).unique()
                filename = FILENAME + '_ID_' + '-'.join(organisms.astype(str).tolist())
                print('\n----- %s -----'%filename)
                print('\nMapping HPIDB entries to UniProt database...')
                try:
                    df_inter_temp, df_inter_fasta_temp = map_hpidb_to_uniprot(df_current, include_unreviewed=args.unreviewed)
                    print('\t%s mapped PPIs'%df_inter_temp.shape[0])
                except Exception as e:
                    print(e)
                    continue
                
                if df_inter_temp.empty or df_inter_fasta_temp.empty:
                    continue
            
                df_inter_temp.to_csv(args.results + filename + '_interactions.tsv', columns=['Protein A', 'Protein B'], sep='\t', header=None, index=False)
                df_inter_fasta_temp.to_csv(args.results + filename + '_sequences.fasta', sep='\n', header=None, index=False)
                
                print('\nRunning CD-HIT...')
                df_inter_fasta_reduced = run_cdhit(args.results + filename + '_sequences.fasta', cdhit=args.cdhit, threshold=args.sequence_identity)
                df_inter_pos, df_inter_fasta_final = remove_homology_ppi(df_inter_temp, df_inter_fasta_reduced)
                print('\t%s positive PPIs'%df_inter_pos.shape[0])
                
                print('\nGenerating negative PPIs...')
                print('\t%s proteins available'%df_inter_pos[df_inter_pos.columns[0]].append(df_inter_pos[df_inter_pos.columns[1]]).unique().shape[0])
                df_inter_pos, df_inter_neg = generate_negative_interactions(df_inter_pos, diff_locations=args.diff_subcell_local)
                if df_inter_neg.shape[0] == 0:
                        print('\tNo negatives generated...')
                else:
                    # Remove sequences not in interactions
                    seq = df_inter_fasta_final.copy()
                    seq[seq.columns[0]] = seq[seq.columns[0]].str.replace('>', '')
                    proteins = df_inter_pos[df_inter_pos.columns[0]].append(df_inter_pos[df_inter_pos.columns[1]]).unique()
                    seq = seq[seq[seq.columns[0]].isin(proteins)]
                    seq[seq.columns[0]] = '>' + seq[seq.columns[0]]
                    seq.reset_index(drop=True, inplace=True)
                    print('\nSaving PPI dataset...%s'%filename)
                    df_inter_pos = df_inter_pos[df_inter_pos.columns[:2]]
                    df_inter_pos.columns = df_inter_neg.columns
                    df_inter_neg = df_inter_neg[df_inter_neg.columns[:2]]
                    save_ppi_data(args.results, filename, df_inter_pos, df_inter_neg, df_inter_fasta_final, models=args.models, kfolds=args.kfolds, all_to_all=args.all_to_all, park_marcotte=args.park_marcotte)
                    print('\nTime %s seconds...'%round(time.time() - start, 2))
            except Exception as e:
                print('**********\n', e, '\n')
                os.remove(args.results + filename + '_interactions.tsv')
                os.remove(args.results + filename + '_sequences.fasta')
                continue
    
    print('\nCompleted in %s seconds.'%round(time.time() - start, 2))