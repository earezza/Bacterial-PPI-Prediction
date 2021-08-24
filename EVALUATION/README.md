Environment used:

    python 3.7  
    scikit-learn  
    pandas 0.24.2  
    numpy 1.16.4  
    urllib3 1.26.4  
    matplotlib 3.1.0  
    kneed 0.7.0  
___

### Description:  

    Evaluates the performance of PPI predictions by plotting ROC and Precision-Recall curves.
    Plots prevalence-corrected curves for hypothetically imbalanced data.
    
    Requires files to have no header and be whitespace-separated (.tsv).
    
### Usage:  

    python evaluate_ppi.py -s SCORES/ -l labels.tsv -d 0.5 -r RESULTS/
    
    Input arguements:
        -s <str> Can be either:
            - a directory path where tested PPI prediction k-fold subset files exist (file names must contain the word 'fold' and a number)
            - a file path for the tested PPI predictions
        -l <str> is the file path to the labelled PPIs
        -d <float> is the hypothetical imbalance ratio of positives/all PPIs, default is 0.5
            e.g.
            - balanced data would be 0.5 (number of positives == number of negatives)
            - imbalanced data where 1 positive for every 100 negatives would be 1/101, so 0.0099
        -r <str> is a directory path for saving the results, default is current directory
        -n <str> name for plot titles and files, default is result directory name 
