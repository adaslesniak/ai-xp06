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
