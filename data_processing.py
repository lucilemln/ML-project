import numpy as np


def replaceMissingValuesMean(X):
    #compute the mean of the column
    mean = np.nanmean(X, axis = 0)

    #replace all the NaN values by the mean
    X = np.where(np.isnan(X), mean, X)

    return X

def masking(X, features_name, features_list):
    """takes as input the matrix of the features and the list of the features to keep
    and returns the matrix with the features to keep"""
    #create a mask of the features to keep
    mask = np.isin(features_name, features_list)

    #apply the mask to the matrix
    X = X[:, mask]
    return X


def compute_dist_array(x_featured_clean_1, column_index=None, values_to_remove = []):
    """takes as input the matrix with missing values and the column index, 
        removes the missing values, the 7,9,77,99
      and returns the array of the distribution of the values in the column
      use this function to replace missing values in the test data"""
    
    column_to_change = x_featured_clean_1[:, column_index]
    
    #remove all missing values and nans
    for i in values_to_remove:
        column_to_change = column_to_change[column_to_change != i]

    #compute the distribution of the values in the column
    dist = np.bincount(column_to_change.astype(int))/len(column_to_change)

    return dist

def replace_nan_mean_column(X):
    """takes as input the matrix with missing values and the column index, 
      and returns the array of the distribution of the values in the column
      use this function to replace missing values in the test data"""

    #compute the mean of the column
    mean = np.nanmean(X, axis = 0)

    #replace all the NaN values by the mean
    X = np.where(np.isnan(X), mean, X)

    return X


def feature_processing_test(x_test_featured):
    """replace all the values equal to 7, 9, 77 and 99 by NaN"""
    x_test_featured[x_test_featured == 7] = np.nan
    x_test_featured[x_test_featured == 9] = np.nan
    x_test_featured[x_test_featured == 77] = np.nan
    x_test_featured[x_test_featured == 99] = np.nan

    ###################################################

    #2 PHYSHLTH : in days, scale = 0-30. change 88 to 0

    x_test_featured[x_test_featured[:, 1] == 88, 1] = 0

    ###################################################

    #3 MENTHLTH : in days, scale = 0-30. Change 88 to 0
    #change 88 to 0
    x_test_featured[x_test_featured[:, 2] == 88, 2] = 0

    ###################################################

    #4 HLTHPLN1 : health care access : change 1 (yes) to 0 and 2 (no) to 1, and remove 9 and 7

    x_test_featured[x_test_featured[:, 3] == 1, 3] = 0
    x_test_featured[x_test_featured[:, 3] == 2, 3] = 1

    ###################################################

    #5 MEDCOST : binary variable if medical care was delayed due to cost, change 1 (yes) to 0 and 2 (no) to 1, and remove 9 and 7
    x_test_featured[x_test_featured[:, 4] == 1, 4] = 0
    x_test_featured[x_test_featured[:, 4] == 2, 4] = 1


    ###################################################

    #6 : TOLDHI2 high blood cholesterol binary variable : change 1 (yes) to 0 and 2 (no) to 1, and remove 9 and 7
    x_test_featured[x_test_featured[:, 5] == 1, 5] = 0
    x_test_featured[x_test_featured[:, 5] == 2, 5] = 1

    ###################################################

    #7 CVDSTRK3 ever had a stroke binary variable : change 1 (yes) to 0 and 2 (no) to 1, and remove 9 and 7
    x_test_featured[x_test_featured[:, 6] == 1, 6] = 0
    x_test_featured[x_test_featured[:, 6] == 2, 6] = 1


    ###################################################

    #8 : DIABETE3 : ever had diabete : yes (1) or during pregnancy (2) to 0, pre-diabete or borderline (4) to 1, no (3) to 2, and remove 9 and 7
    x_test_featured[x_test_featured[:, 7] == 1, 7] = 0
    x_test_featured[x_test_featured[:, 7] == 2, 7] = 0
    x_test_featured[x_test_featured[:, 7] == 4, 7] = 1
    x_test_featured[x_test_featured[:, 7] == 3, 7] = 2


    ###################################################

    #9 : SEX : male (1) to 0 and female (2) to 1
    x_test_featured[x_test_featured[:, 8] == 1, 8] = 0
    x_test_featured[x_test_featured[:, 8] == 2, 8] = 1

    ###################################################

    #12 : DIFFWALK : binary :Do you have serious difficulty walking or climbing stairs? 1 = yes, 2 = no

    #change 1 (yes) to 0 and 2 (no) to 1

    x_test_featured[x_test_featured[:, 11] == 1, 11] = 0
    x_test_featured[x_test_featured[:, 11] == 2, 11] = 1

    ###################################################

    #13 : SMOKE100 : binary variable : Have you smoked at least 100 cigarettes in your entire life? 1 = yes, 2 = no, 7 = don't know, 9 = missing values

    #change 1 (yes) to 0 and 2 (no) to 1

    x_test_featured[x_test_featured[:, 12] == 1, 12] = 0
    x_test_featured[x_test_featured[:, 12] == 2, 12] = 1

    ###################################################

    #14 : _RFHYPE5 : Adults who have been told they have high blood pressure by a doctor. Yes (2) to 0, no (1) to 1, 7 = don't know, 9 = missing values

    #change 1 (yes) to 0 and 2 (no) to 1

    x_test_featured[x_test_featured[:, 13] == 1, 13] = 1
    x_test_featured[x_test_featured[:, 13] == 2, 13] = 0

    ###################################################

    #15 : _CHOLCHK : Cholesterol check within past five years. No = {2,3} to 0, Yes = {1}, remove missing values

    #change 2 and 3 to 0 

    x_test_featured[x_test_featured[:, 14] == 2, 14] = 0
    x_test_featured[x_test_featured[:, 14] == 3, 14] = 0

    ###################################################

    #17 : _BMI5 : Body mass index (kg/m**2) : divide by 100
    x_test_featured[:, 16] = x_test_featured[:, 16] / 100

    ###################################################

    #18 : _RFDRHV5 : Heavy drinkers, binary. yes (2) to 0 and no (1) to 1

    x_test_featured[x_test_featured[:, 17] == 2, 17] = 0


    ###################################################

    #19 : _FRTLT1 : Consume Fruit 1 or more times per day : change 1 = yes to 0, 2 = less than 1 time per day to 1, 9 = missing values
    x_test_featured[x_test_featured[:, 18] == 1, 18] = 0
    x_test_featured[x_test_featured[:, 18] == 2, 18] = 1


    ###################################################

    #20 : _VEGLT1 : Consume Vegetables 1 or more times per day : change 1 = yes to 0, 2 = less than 1 time per day to 1, 9 = missing values
    x_test_featured[x_test_featured[:, 19] == 1, 19] = 0
    x_test_featured[x_test_featured[:, 19] == 2, 19] = 1


    ###################################################

    #21 : _TOTINDA : Adults who reported doing physical activity or exercise during the past 30 days other than their regular job. 
    #change 1 = yes to 0, 2 = less than 1 time per day to 1, 9 = missing values

    x_test_featured[x_test_featured[:, 20] == 1, 20] = 0
    x_test_featured[x_test_featured[:, 20] == 2, 20] = 1


    return x_test_featured