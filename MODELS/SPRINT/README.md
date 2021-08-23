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
