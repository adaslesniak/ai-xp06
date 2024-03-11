## Estimate randomness of random forest

I've read that random forest randomly subsamples features for different trees. 
That pure randomnes leads to overrepresentation of some features and underepresentation
of others. Firstly it may lead to underrepresentation of crucial features, secondly
it's reducing diversity (some features are overrepresented) what is against the spirit
of random forest.

# Goal
Goal of this experiment is to create alternative implementation of random forest,
that would use more uniform distribution of features among trees in the forest 
and check how that would affect results.

# Results
Unclear, definitely RandomForestClassifier isn't as simple as described in lectures. 
My naive approach is just terrible compared to it, but my naive approach with 
the trick of more uniform distribution vs without the trick - difference is huge.

========== IRIS ==========
classic >>  cv accuracy: 0.95 +/- 0.03,  Accuracy: 1.00, Precision: 1.00, Recall: 1.00, F1 Score: 1.00
uniform >>  cv accuracy: 0.91 +/- 0.09,  Accuracy: 1.00, Precision: 1.00, Recall: 1.00, F1 Score: 1.00
naive   >>  cv accuracy: 0.69 +/- 0.31,  Accuracy: 0.56, Precision: 0.46, Recall: 0.54, F1 Score: 0.45
========== Wine Quality ==========
classic >>  cv accuracy: 0.59 +/- 0.03,  Accuracy: 0.59, Precision: 0.29, Recall: 0.26, F1 Score: 0.26
uniform >>  cv accuracy: 0.55 +/- 0.06,  Accuracy: 0.56, Precision: 0.29, Recall: 0.24, F1 Score: 0.24
naive   >>  cv accuracy: 0.34 +/- 0.05,  Accuracy: 0.32, Precision: 0.17, Recall: 0.23, F1 Score: 0.13
