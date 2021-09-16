# Bayesian Nonnegative Matrix Factorization

This folder contains the source code for the paper "Bayesian Nonnegative Matrix Factorization"


#### -- Project Status: [Under Review]

## Description
The purpose of this project is to develop a new Bayesian Nonnegative Matrix Factorization based on inverted beta implementation and the usage of the variational learning. The Matrix Factorization model can be applied on various application. On this repo, we share with the readers an example of how to run it on the 1M movielens dataset[1]. The demo.py script returns the ratings as well as the evaluation metrics. 

### Methods Used
* Variational Inferential
* Machine Learning
* Data Visualization

### Technologies
* Python
* Pandas, jupyter

## Folder Structure
Depending upon the selected options when creating the project, the generated structure will look similar to the below:

```
├── README.md                
├── data
│   ├── __init__.py
│   └── CF.py				 <- Data preprocessing method for Collaborative Filtering.
│
├── models					 
│   ├── IBG_BNMF.py  			 <- Bayesian non negative matrix factorization approach.
│   └── online_IBG_BNMF.py       <- online Bayesian non negative matrix factorization approach.
│   
├── src                      <- Examples to run with this implementation
│   └── demmo.py   		     <- Example python package - place shared code in such a package
│       
└── utils                    
    └── distributions   
        ├── gamma.py    
        ├── IB_vector.py    
        ├── io              
        └── inverted_beta.py
```

## Installation 

To use this project, first clone the repo on your device using the command below:

		git init

		git clone https://github.com/Oumaymaa/IBG-NMF.git

Download raw data from https://files.grouplens.org/datasets/movielens/ml-1m.zip and unzip it under [./data]


## Getting Started

1. Data processing/transformation scripts are being kept under [./data]
2. Demo script is available under [./examples]
3. Results are stored under [./results/1M]

## Contact
If you have any questions, please feel free to contact the authors on oumayma.dalhoumi@concordia.ca

## References
[1] F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History
and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4,
Article 19 (December 2015), 19 pages. DOI=http://dx.doi.org/10.1145/2827872
