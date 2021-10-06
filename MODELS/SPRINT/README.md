**[SPRINT](https://github.com/lucian-ilie/SPRINT)**  
Y. Li, L. Ilie, SPRINT: Ultrafast protein-protein interaction prediction of the entire human interactome, BMC Bioinformatics 18 (2017) 485.   
Y. Li, L. Ilie, Predicting Protein–Protein Interactions Using SPRINT, In Protein-Protein Interaction Networks (pp. 1-11). Humana, New York, NY.  
  
___
## Usage:  

e.g.  
> **python sprint.py -p protein_sequences.fasta -h HSP/file.hsp -f interaction_data.tsv -k5**  

Description:  

    Python wrapper to run SPRINT as a subproccess.  
    Uses all the same arguements as SPRINT with additional options.  
    -f option is file with labelled PPIs for running cross-validation  
    Makes running cross-validation easier.  
    Includes evaluation of predictions using ROC and precision-recall curves.  
    
Separate from the python wrapper,  

If running ./bin/compute_HSPs takes very long, create a binary for Src/compute_HSPs_checkpoints:  
To compile, change to SPRINT/ directory and run **make compute_HSPs_serial_checkpoints** or **make compute_HSPs_parallel_checkpoints**.  
    
This allows you to compute HSPs for each hashtable separately by running **./bin/compute_HSPs_checkpoints** with an additional arguement **-hashtable**. Be sure to save each HSP file under a different name.  

e.g.  
> ./bin/compute_HSPs_checkpoints -p proteins.fasta -h hsp_1 -hashtable 1  
> ./bin/compute_HSPs_checkpoints -p proteins.fasta -h hsp_2 -hashtable 2  
> ./bin/compute_HSPs_checkpoints -p proteins.fasta -h hsp_3 -hashtable 3  
> ./bin/compute_HSPs_checkpoints -p proteins.fasta -h hsp_4 -hashtable 4  

Then you can combine each HSP file from all 4 hashtable runs using **combine_hsp_files.py**  
e.g.  
> python combine_hsp_files.py -f hsp_1 hsp_2 hsp_3 hsp_4 -r HSP/ -n hsps.hsp  
