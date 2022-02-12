Environment used:

    python 3.7  
    scikit-learn  
    pandas 0.24.2  
    numpy 1.16.4  
    urllib3 1.26.4  
    matplotlib 3.1.0  
    kneed 0.7.0  
    scipy 1.3.0
    statsmodels 0.12.2
    
___

### evaluate_ppi.py Description:  

    Evaluates the performance of PPI predictions by plotting ROC and Precision-Recall curves.  
    Plots prevalence-corrected curves for hypothetically imbalanced data.  
    Writes to file with evaluated metric results.  
    
    Requires files to have no header and be whitespace-separated (.tsv).  
    
### evaluate_ppi.py Usage:  
    eg.
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

### compare_performace.py Description:  
    Compares the performance of PPI prediction results between methods.
    Performs comparisons by significance/hypothesis testing using ANOVA first, then two-tailed t-tests.
    Writes to file with comparison results.
    
    Requires files in PPI results to have no header and be whitespace-separated (.tsv).
    
### compare_performance.py Usage:
    eg.
    python compare_performance.py -s SCORES_1/ SCORES_2/ SCORES_3/ -l labels_1.tsv labels_2.tsv labels_3.tsv -d 0.5 -r RESULTS/
    python compare_performance.py -s SCORES_1.tsv SCORES_2.tsv SCORES_3.tsv -l labels.tsv -d 0.5 -r RESULTS/ -m auc_pr -t paired
    
    Input arguements:
        -s list of <str> Can be either:
            - a directory path where tested PPI prediction k-fold subset files exist (file names must contain the word 'prediction' and a number)
            - a file path for the tested PPI predictions
        -l list of <str> is the file path to the labelled PPIs
            - order of provided list corresponds to order of provided scores list
            - e.g. SCORES_1 predictions will be evaluated against labels_1.tsv, then SCORES_2 with labels_2.tsv
            - if only one file of labels is provided, all scores will be evaluated against those labels
        -d <float> is the hypothetical imbalance ratio of positives/all PPIs, default is 0.5
            e.g.
            - balanced data would be 0.5 (number of positives == number of negatives)
            - imbalanced data where 1 positive for every 100 negatives would be 1/101, so 0.0099
        -r <str> is a directory path for saving the results, default is current directory + 'COMPARISONS/'
        -n <str> name for saving files, default is result directory name
        -m <str> metric used in significance tests to compare performance, default is area under precision-recall curve
        -t <str> type of two-tailed t-test performed (paired or independent), default is independent
