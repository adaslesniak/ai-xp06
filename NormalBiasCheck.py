import FeatureDistribution as distribution_data
import numpy as np
 
# This script was created to check for something I call "normal bias" in random forest algorithm by sci-kit library.
# Normal bias is a bias caused by normal distribution if we would randomly sample features for each tree.

def analyze_usages(dataset):
    usages, importances, deviations = dataset
    print("Feature Importance:  ", importances)
    avg_use = np.round(np.average(usages, axis=0), 0)
    print("Average use:         ", avg_use)
    for usage in usages:
        print("Feature Distribution:", usage)
    avg_deviation = np.round(np.average(deviations, axis=0), 1)
    print("Average deviation:   ", avg_deviation)
    print("")


analyze_usages(distribution_data.iris())

analyze_usages(distribution_data.wine_quality())


# SUMMARY: Sci-Kit library algorithm for random forest does not pick features randomly
#    those features are definitely and clearly weighted - most probably by feature importance
#    their distribution is neither uniform, nor normal. Instead of random feature selection
#    there is some focus mechanism that picks most important data features and overrepresents them.