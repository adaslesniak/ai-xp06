import FeatureDistribution as distribution_data
import numpy as np

# Post check after NormalBiasCheck
# I'm here trying to find out if even after adjusting for feature importance and 
# focus algorithm in sci-kit random forest algorithm for picking features to given trees
# we still may found some resiudual normal bias (comming from normal distribution of random sampling)

def analyze_adjusted_bias(dataset):
    usages, importances, deviations = dataset
    print("importances: ", np.round(importances, 1))
    print("deviations:  ", np.round(deviations, 1))
    weighted_average_deviation = np.sum(importances * deviations) / np.sum(importances)
    print(" ---> [" + str(len(usages[0])) + " features]  variablity:", weighted_average_deviation)

analyze_adjusted_bias(distribution_data.iris())
analyze_adjusted_bias(distribution_data.wine_quality())
analyze_adjusted_bias(distribution_data.breast_cancer())
analyze_adjusted_bias(distribution_data.human_activity())

#TODO summarize the results and conclusions