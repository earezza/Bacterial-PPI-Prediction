**[DeepFE-PPI](https://github.com/xal2019/DeepFE-PPI)**  
Yao, Y., Du, X., Diao, Y., & Zhu, H. (2019). An integration of deep learning with feature embedding for protein-protein interaction prediction. PeerJ, 7, e7126. https://doi.org/10.7717/peerj.7126  
 
___
## Usage:  

e.g.  
> **python deepfe_res2vec.py dataTrain/ dataTest/**  

1. First arg supplies directory to training data, second arg supplies directory to test data  

2. Each directory should contain .fasta files named with 'ProteinA' or 'ProteinB' and 'positive' or 'negative':  
e.g.  
> ls dataTrain/  
> negative_ProteinA.fasta  positive_ProteinA.fasta negative_ProteinB.fasta  positive_ProteinB.fasta  

These are formatted such that the .fasta lines of protein IDs and sequences line up one-to-one with protein As and protein Bs  
e.g.  
> head positive_ProteinA.fasta  
> \>PROTEINA1  
> SEQUENCEA1  
> \>PROTEINA2  
> SEQUENCEA2  
> \>PROTEINA3  
> SEQUENCEA3  

> head positive_ProteinB.fasta  
> \>PROTEINB1  
> SEQUENCEB1  
> \>PROTEINB2  
> SEQUENCEB2  
> \>PROTEINB3  
> SEQUENCEB3  

Therefore, PROTEINA1 and PROTEINB1 are a positive interaction, PROTEINA2 and PROTEINB2 are positive, etc...thus, the number of lines for each A - B file must be equal.  

<i>Note 1: if train data and test data args are the same, a 5-fold cross-validation will be performed on the provided data.</i> 

### Requirements:  
python3.5.2  
Numpy 1.14.1  
Gensim 3.4.0  
HDF5 and h5py  
Pickle  
Scikit-learn 0.19  
Tensorflow 1.2.0  
keras 1.2.0  
