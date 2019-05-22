import numpy as np
from sklearn.metrics import label_ranking_average_precision_score

# In our use case 1 means that the image is relevant (same label as the query image) 
# And 0 means that the image is irrelevant
y_true = np.array([[1, 1, 0, 0]])

# For each train image we compute the relevance score

''' Example 1 '''
y_score = np.array([[28, 10, 1, 0.5]])
label_ranking_average_precision_score(y_true, y_score)
# In this first example, the two relevant items have the highest score,
# the scoring function returns 1.0

''' Example 2'''
y_score = np.array([[28, 10, 10, 0.5]])
label_ranking_average_precision_score(y_true, y_score)
# returns 0.83333333333333326

''' Example 3'''
y_score = np.array([[28, 10, 28, 0.5]])
label_ranking_average_precision_score(y_true, y_score)
# returns 0.58333333333333326

''' Example 4'''
y_score = np.array([[10, 10, 28, 28]])
label_ranking_average_precision_score(y_true, y_score)
# returns 0.5