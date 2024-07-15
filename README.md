# MSI2DA-Net

Experimental codes for paper "Importance-aware Subgraph Convolutional Networks Based on Multi-source Information Fusion for Cross-domain Mechanical Fault Diagnosis".

<div align=center>
<img src="https://github.com/Polimi-YuYue/MSI2DA-Net/blob/main/Overall%20Framework.png" width="700px">
</div>

# Abstract

Graph convolutional networks (GCNs) as the emerging neural networks have shown great success in Prognostics and Health Management because they can not only extract node features but can also mine relationship between nodes in the graph data. However, the most existing GCNs-based methods are still limited by graph quality, variable working conditions, and limited data, making them difficult to obtain remarkable performance. Therefore, it is proposed in this paper a two stage importance-aware subgraph convolutional network based on multi-source sensors named $I_2SGCNs$ to address the above-mentioned limitations. In the real-world scenarios, it is found that the diagnostic performance of the most existing GCNs is commonly bounded by the graph quality because it is hard to get high quality through a single sensor. Therefore, we leveraged multi-source sensors to construct graphs that contain more fault-based information of mechanical equipment. Then, we discovered that unsupervised domain adaptation (UDA) methods only use single stage to achieve cross-domain fault diagnosis and ignore more refined feature extraction, which can make the representations contained in the features inadequate. Hence, it is proposed the two-stage fault diagnosis in the whole framework to achieve UDA. In the first stage, the multiple-instance learning is adopted to obtain the importance factor of each sensor towards preliminary fault diagnosis. In the second stage, it is proposed 
 to achieve refined cross-domain fault diagnosis. Moreover, we observed that deficient and limited data may cause label bias and biased training, leading to reduced generalization capacity of the proposed method. Therefore, we constructed the feature-based graph and importance-based graph to jointly mine more effective relationship and then presented a subgraph learning strategy, which not only enriches sufficient and complementary features but also regularizes the training. Comprehensive experiments conducted on four case studies demonstrate the effectiveness and superiority of the proposed method for cross-domain fault diagnosis, which outperforms the state-of-the art methods.


# Paper

A two-stage importance-aware subgraph convolutional network based on multi-source sensors for cross-domain fault diagnosis

a. Yue Yu, a. Youqian He, a. Hamid Reza Karimi, b. Len Gelman, c. Ahmet Enis Cetin

a Department of Mechanical Engineering, Politecnico di Milano, via La Masa 1, Milan 20156, Italy
b School of Computing and Engineering, University of Huddersfield, Queensgate, Huddersfield, HD1 3DH, UK
c Department of Electrical and Computer Engineering, University of Illinois Chicago, Chicago, USA

https://www.sciencedirect.com/science/article/pii/S0893608024004428

# Citation

@article{YU2024106518,
title = {A two-stage importance-aware subgraph convolutional network based on multi-source sensors for cross-domain fault diagnosis},
journal = {Neural Networks},
pages = {106518},
year = {2024},
issn = {0893-6080},
doi = {https://doi.org/10.1016/j.neunet.2024.106518},
url = {https://www.sciencedirect.com/science/article/pii/S0893608024004428},
author = {Yue Yu and Youqian He and Hamid Reza Karimi and Len Gelman and Ahmet Enis Cetin},}
