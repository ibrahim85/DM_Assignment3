import numpy as np
import pandas as pd
import matplotlib
import math
import csv

from math import isnan

def impute_neighbors(row, n_neighbors=3):
    """
    Uses KNN for data imputation. Accepts only categorical values and ignores
    rows with any null values. Returns the k-nearest instances as a dataframe
    """
    neighbors = []
    
    for i, instance in data.iterrows():
        # Iterate over all the rows in the dataframe

        if row.name == i or instance.isnull().any():
            # If we're looking at the same row we passed or a row
            # with null values we pass over this instance
            continue
        else:
            # Otherwise measure the distance
            dist = 0
            for attr, _ in row.iteritems():
                # Distance is 1 if the two items are not equal, 0 otherwise
                if row[attr] != instance[attr]:
                    dist += 1
                    
            # Append the distance and instance to our list
            neighbors.append((dist, instance))
        
    # Sort the list by distances and store only the instances
    knn = [tup[1] for tup in sorted(neighbors, key=lambda t: t[0])]
    
    # Return the KNN as a dataframe
    return pd.DataFrame(knn[:n_neighbors], columns=row.index)


def knn_impute(data, k_neighbors=3):
    """
    Imputes missing data using the nearest non-null neighbors
    """
    # Iterate over rows
    for i, row in data.iterrows():

        # Find rows that contain nulls
        if row.isnull().any():

            # Find K nearest neighbors
            knn = impute_neighbors(row)

            # Find the cell with the null value and fill it
            for i, v in row.iteritems():
                if isinstance(v, float) and isnan(v):
                    # Fill that with the voted upon value
                    val = knn[i].mode().values[0]
                    data.set_value(row.name, i, val)


if __name__ == "__main__":
    
    data = pd.read_csv('breast-cancer.csv', quotechar="'", na_values='?')

    knn_impute(data)

    data.to_csv(path_or_buf='breast-cancer-imputed.csv', 
            quotechar="'", 
            na_rep='?', 
            quoting=csv.QUOTE_ALL, 
            index=False)
