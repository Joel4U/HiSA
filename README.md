# HiSA
A Named Entity Recognition (NER) model that utilizes soft head word embeddings of the spans, guided by syntactic priors and distance priors to direct the attention heads in performing inter-span attention.

Performance of the single-layer attention version (without attention prior parameter decay and without updating the neighbor span set).
Achieving performance comparable to the 3-layer Span-CNN.

**On CoNLL 2003 EN with bert-base-cased** 
| Seed | P. (Precision) | R. (Recall) | F1 |
| :---| :---  | :---  | :---  |
| 11  | 91.96 | 90.88 | 91.42 |
| 35  | 92.05 | 91.47 | 91.76 |
| 39  | 92.03 | 91.75 | 91.89 |
