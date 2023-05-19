% All Copyrights Reserved
% Please Cite our paper P.-Y. Chen and A. O. Hero, ��Multilayer Spectral Graph Clustering via Convex Layer Aggregation: Theory and Algorithms,�� IEEE Transactions on Signal and Information Processing over Networks, 2017
% if you use our codes for any purpose
% Aug. 8 2017

Description:

This dataset contains the coauthors of Prof. Jure Leskovec or Prof. Andrew Ng at Stanford University
from year 1995 to year 2014.
We partition this 20-year co-authorship into 4 different 5-year intervals and hence create
a 4-layer multilayer graph. For each layer, there is an edge between two researchers if they co-authored at least one paper in the 5-year interval. For every edge in each layer, we adopt the temporal collaboration strength as the edge weight proposed in [2,3]. We manually label each researcher by either ``Leskovec's collaborator'' or ``Ng's collaborator'' based on the collaboration frequency and use the labels as the ground-truth cluster assignment.

++++++++++++++++++++++++++++

Matlab format: 

1. Leskovec_Ng_author_name.mat: author names
2. LN_true: ground-truth labels
3. LN_XXXX_OOOO: collaboration strength matrix among all authors from year XXXX to OOOO

Plain format:
1. multi_graph_matrix_X collaboration strength matrix among all authors for layer X




