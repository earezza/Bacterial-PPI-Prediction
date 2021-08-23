**[DPPI](https://github.com/hashemifar/DPPI)**  
Hashemifar S, Neyshabur B, Khan AA, Xu J. Predicting protein-protein interactions through sequence-based deep learning. Bioinformatics. 2018 Sep 1;34(17):i802-i810. doi: 10.1093/bioinformatics/bty573. PMID: 30423091; PMCID: PMC6129267. 
 
___
## Usage:  

e.g.  
> **th dppi.lua -train dataTrain -test dataTest -device 1**  

- First arg supplies name for training data, second arg supplies name for test data  
- Each dataset should contain the following using the _same name_  
e.g.  
1. A **dataTrain.node** file containing a list of all protein IDs  
> PROTEIN1  
> PROTEIN2  
> PROTEIN3  
> ...  
2. A **dataTrain.csv** file containing the interactions with labels  
> PROTEIN1,PROTEIN2,1  
> PROTEIN1,PROTEIN1,1  
> PROTEIN3,PROTEIN2,1  
> PROTEIN1,PROTEIN3,0  
> PROTEIN2,PROTEIN2,0  
> ...  
3. A  **dataTrain/** directory containing files named as the protein IDs found in the .node file without any extension.  
Each file's contents contain the PSSM results table of a [BLAST](https://blast.ncbi.nlm.nih.gov/Blast.cgi?PAGE_TYPE=BlastDocs&DOC_TYPE=Download) search for that protein against a database  
      
    1. This can be obtained by first downloading the [BLAST+ executable](https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/) and a [BLAST database](https://ftp.ncbi.nlm.nih.gov/blast/db/) and unzipping it, then formatting the db  
     e.g.  
    > makeblastdb -in uniprot_sprot.fasta -out uniprot_sprot -dbtype prot -title swissprot   
    2. Then make a separate directory named Proteins/ containing the .fasta file for each protein in dataTrain  
    e.g.  
    > head Proteins/PROTEIN1  
    > \>PROTEIN1  
    > SEQUENCE1  
    > head Proteins/PROTEIN2  
    > \>PROTEIN2  
    > SEQUENCE2  
    > ...etc...  
    3. Then either run the get_profiles.sh provided  
    > ./get_profiles.sh  
    
   or perform the BLAST search otherwise  
   e.g.  
    > psiblast -db swissprot -evalue 0.001 -query Proteins/PROTEIN1 -out_ascii_pssm dataTrain/PROTEIN1 -out dataTrain/PROTEIN1-output_file -num_iterations 3  
   
   and format to obtain just the PSSM tables.  
   e.g. values should all be tab-spaced
   > -1	-1	-2	-3	-1	0 ...  
   > -1	5	0	-2	-3	1 ...  
   > 1	-1	1	0	-1	0 ...  
   > -1	0	5	1	-3	0 ...  
   > ... ... ...  
   
Repeat for test set.  

<i>Note: if train data and test data args are the same, a (default) 5-fold cross-validation will be performed on the provided data.</i>  

### Requirements:  
lua  
cuda  
torch  
