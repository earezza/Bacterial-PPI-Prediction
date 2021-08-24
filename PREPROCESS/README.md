# BioGRID Dataset Preprocessing

After downloading and unzipping a [BioGRID release](https://downloads.thebiogrid.org/BioGRID/Release-Archive/) file, run the following for example:  

python preprocess_biogrid.py biogrid_filename.txt -t intra -c2 -f -s0.6 -m pipr sprint deepfe dppi -k5  
  
Environment used:

    python 3.7  
    scikit-learn  
    pandas 0.24.2  
    numpy 1.16.4  
    urllib3 1.26.4  
    matplotlib 3.1.0  
    kneed 0.7.0  
___

    Description:
    
    Builds balanced protein-protein interaction (PPI) datasets from BioGRID (.tab2 or .tab3) data.
    
    Requires:
        - BioGRID .tab.txt file (for extracting protein interactions)
        - CD-HIT software installed (for removing homologous proteins)
        - an internet connection (for accessing the UniProt database)
    
    Preprocessing Steps:
        1. Extract positive interactions from BioGRID:
            - Collects interactions from BioGRID file using UniProt database 
              for mapping reviewed (Swiss-Prot) entrez gene IDs to protein IDs and sequences.
            
            Options:
                -f <flag> apply conservative filters to interactions
                -u <flag> include unreviewed UniProt entries
                -t <str> type of interactions to extract
                    'intra': extract only intraspecies interactions within BioGRID file
                    'inter': extract only interspecies interactions within BioGRID file
                    'both' (DEFAULT): extract both intra- and interspecies interactions within BioGRID file
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
            - dataset is labelled and saved under BIOGRID_DATA/ as a .tsv file with no header with a .fasta file
            - options allow for additionally saving dataset as formatted for different PPI prediction models
            - option allows for creating k-fold subsets of data
            
            Options:
                -n <str> name to rename the files from the BioGRID organism name in the file (DEFAULT)
                -r <str> directory location to save the resulting datasets (default is BIOGRID_DATA/)
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
               
#### General output file format  
Sequence data as .fasta files, example:  
>\>ProteinID_1  
SEQUENCE1  
\>ProteinID_2  
SEQUENCE2  
...  
  
Interaction data as .tsv files, example:  
>ProteinA  ProteinB  
ProteinC  ProteinD  
...  
     
This will also generate new directories for the chosen models/formats and format the provided data to the corresponding directory.  
*Currently only supports formatting for [PIPR](https://github.com/muhaochen/seq_ppi), [DeepFE-PPI](https://github.com/xal2019/DeepFE-PPI), [DPPI](https://github.com/hashemifar/DPPI), and [SPRINT](https://github.com/lucian-ilie/SPRINT) protein prediction models.*  
*Note: DPPI will still require a PSI-BLAST search for PSSMs using the protein files created and the get_profiles.sh script.*  
  
#### To get PSSM profiles for DPPI, download BLAST and a BLAST database then run:  

./get_profiles.sh directory_with_sequences/ database_name  
  
*Note: This will add the PSSM files to the same directory as the .fasta files. For use in DPPI, the directory should only contain PSSM files with filenames of the protein IDs and no extension.*  
___  
### References:  

Obtaining protein sequence's PSSM profile data for DPPI can be derived using [BLAST+](https://blast.ncbi.nlm.nih.gov/Blast.cgi?CMD=Web&PAGE_TYPE=BlastDocs&DOC_TYPE=Download)  
- Altschul S, Gish W, Miller W, Myers E, Lipman D: Basic local alignment search tool. J Mol Biol 1990, 215(3):403–410.  
- Altschul S, Madden T, Schäffer A, Zhang J, Zhang Z, Miller W, Lipman D: Gapped BLAST and PSI-BLAST: a new generation of protein database search programs. Nucleic Acids Res 1997, 25(17):3389–3402. 10.1093/nar/25.17.3389  
- Camacho, C., Coulouris, G., Avagyan, V. et al. BLAST+: architecture and applications. BMC Bioinformatics 10, 421 (2009). https://doi.org/10.1186/1471-2105-10-421  
  
Positive interactions are filtered based on conservative approach prescribed by [POSITOME](http://bioinf.sce.carleton.ca/POSITOME/)  
- K. Dick, F. Dehne, A. Golshani and J. R. Green, "Positome: A method for improving protein-protein interaction quality and prediction accuracy," 2017 IEEE Conference on Computational Intelligence in Bioinformatics and Computational Biology (CIBCB), Manchester, 2017, pp. 1-8, doi: 10.1109/CIBCB.2017.8058545.  
  
Sequence similarity is filtered using [CDHIT](http://weizhong-lab.ucsd.edu/cd-hit/)  
- Weizhong Li & Adam Godzik. Cd-hit: a fast program for clustering and comparing large sets of protein or nucleotide sequences. Bioinformatics (2006) 22:1658-1659  
- Limin Fu, Beifang Niu, Zhengwei Zhu, Sitao Wu and Weizhong Li, CD-HIT: accelerated for clustering the next generation sequencing data. Bioinformatics, (2012), 28 (23): 3150-3152. doi: 10.1093/bioinformatics/bts565  
  
