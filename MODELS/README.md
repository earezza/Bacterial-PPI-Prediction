### Models used for binary protein interaction prediction:
1. **[PIPR](https://github.com/muhaochen/seq_ppi)**  
Muhao Chen, Chelsea J -T Ju, Guangyu Zhou, Xuelu Chen, Tianran Zhang, Kai-Wei Chang, Carlo Zaniolo, Wei Wang, Multifaceted protein–protein interaction prediction based on Siamese residual RCNN, Bioinformatics, Volume 35, Issue 14, July 2019, Pages i305–i314, https://doi.org/10.1093/bioinformatics/btz328  
e.g.  
CUDA_VISIBLE_DEVICES=0 python pipr_rcnn.py all_sequences.fasta dataTrain.tsv dataTest.tsv  

2. **[DeepFE-PPI](https://github.com/xal2019/DeepFE-PPI)**  
Yao, Y., Du, X., Diao, Y., & Zhu, H. (2019). An integration of deep learning with feature embedding for protein-protein interaction prediction. PeerJ, 7, e7126. https://doi.org/10.7717/peerj.7126  
e.g.  
python deepfe_res2vec.py dataTrain/ dataTest/  

3. **[DPPI](https://github.com/hashemifar/DPPI)**  
Hashemifar S, Neyshabur B, Khan AA, Xu J. Predicting protein-protein interactions through sequence-based deep learning. Bioinformatics. 2018 Sep 1;34(17):i802-i810. doi: 10.1093/bioinformatics/bty573. PMID: 30423091; PMCID: PMC6129267.  
e.g.  
th dppi.lua -train dataTrain -test dataTest -device 1  

4. **[SPRINT](https://github.com/lucian-ilie/SPRINT)**  
Li, Y., Ilie, L. SPRINT: ultrafast protein-protein interaction prediction of the entire human interactome. BMC Bioinformatics 18, 485 (2017). https://doi.org/10.1186/s12859-017-1871-x  
e.g.  
bin/compute_HSPs -p sequences.fasta -h hsp_filename  
bin/predict_interactions -p sequences.fasta -h HSP/hsp_filename -tr positive_interactions.tsv -e -o score_results.txt
