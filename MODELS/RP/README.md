## Modules used for implementation of:  
**[Reciprocal Perspective for Improved Protein-Protein Interaction Prediction](https://github.com/hashemifar/DPPI)**  
Dick, K., Green, J.R. Reciprocal Perspective for Improved Protein-Protein Interaction Prediction. Sci Rep 8, 11694 (2018). https://doi.org/10.1038/s41598-018-30044-1  
  
___
### Description:  
After running a prediction model and obtaining all-to-all PPI prediction results, the **get_rp_features.py** module can be used to extract RP features.  
Cross-validated prediction results (preferred) can be used by averaging scores for all-to-all PPIs for less biased input into extracting RP features.  
A labelled PPI dataset will then have RP features for use in any machine learning model.  
  
Then, **rp_ppi_classifier.py** can be run using the RP dataset to make predictions based on a previous model's results or the combined RP features from multiple models' predictions.  
