def feature_processing(x_featured_clean_1, y_clean, ids):
    """takes as input the matrix with no missing values and the output vector (also with rows removed) and returns the matrix with the features processed)"""

    #1 GenHealth : ordinal variable, 1 = very good physical condition -> 5 = very bad physical condition. 7/9 = missing values that need to be dropped

    #remove all missing values
    y_clean = y_clean[x_featured_clean_1[:, 0] != 7] 
    ids = ids[x_featured_clean_1[:, 0] != 7]
    x_featured_clean_1 = x_featured_clean_1[x_featured_clean_1[:, 0] != 7]
    y_clean = y_clean[x_featured_clean_1[:, 0] != 9]
    ids = ids[x_featured_clean_1[:, 0] != 9]
    x_featured_clean_1 = x_featured_clean_1[x_featured_clean_1[:, 0] != 9]

    ###################################################

    #2 PHYSHLTH : in days, scale = 0-30. 77/99 = missing values that need to be dropped
    #change 88 to 0
    x_featured_clean_1[x_featured_clean_1[:, 1] == 88, 1] = 0

    #remove 77 and 99
    y_clean = y_clean[x_featured_clean_1[:, 1] != 77]
    ids = ids[x_featured_clean_1[:, 1] != 77]
    x_featured_clean_1 = x_featured_clean_1[x_featured_clean_1[:, 1] != 77]
    y_clean = y_clean[x_featured_clean_1[:, 1] != 99]
    ids = ids[x_featured_clean_1[:, 1] != 99]
    x_featured_clean_1 = x_featured_clean_1[x_featured_clean_1[:, 1] != 99]

    ###################################################
    #3 MENTHLTH same as PHYSHLTH
    #change 88 to 0
    x_featured_clean_1[x_featured_clean_1[:, 2] == 88, 2] = 0

    #remove 77 and 99
    y_clean = y_clean[x_featured_clean_1[:, 2] != 77]
    ids = ids[x_featured_clean_1[:, 2] != 77]
    x_featured_clean_1 = x_featured_clean_1[x_featured_clean_1[:, 2] != 77]
    y_clean = y_clean[x_featured_clean_1[:, 2] != 99]
    ids = ids[x_featured_clean_1[:, 2] != 99]
    x_featured_clean_1 = x_featured_clean_1[x_featured_clean_1[:, 2] != 99]

    ###################################################

    #4 HLTHPLN1 : health care access : change 1 (yes) to 0 and 2 (no) to 1, and remove 9 and 7
    x_featured_clean_1[x_featured_clean_1[:, 3] == 1, 3] = 0
    x_featured_clean_1[x_featured_clean_1[:, 3] == 2, 3] = 1
    y_clean = y_clean[x_featured_clean_1[:, 3] != 9]
    ids = ids[x_featured_clean_1[:, 3] != 9]
    x_featured_clean_1 = x_featured_clean_1[x_featured_clean_1[:, 3] != 9]
    y_clean = y_clean[x_featured_clean_1[:, 3] != 7]
    ids = ids[x_featured_clean_1[:, 3] != 7]
    x_featured_clean_1 = x_featured_clean_1[x_featured_clean_1[:, 3] != 7]

    ###################################################

    #5 MEDCOST : binary variable if medical care was delayed due to cost, change 1 (yes) to 0 and 2 (no) to 1, and remove 9 and 7
    x_featured_clean_1[x_featured_clean_1[:, 4] == 1, 4] = 0
    x_featured_clean_1[x_featured_clean_1[:, 4] == 2, 4] = 1
    y_clean = y_clean[x_featured_clean_1[:, 4] != 9]
    ids = ids[x_featured_clean_1[:, 4] != 9]
    x_featured_clean_1 = x_featured_clean_1[x_featured_clean_1[:, 4] != 9]
    y_clean = y_clean[x_featured_clean_1[:, 4] != 7]
    ids = ids[x_featured_clean_1[:, 4] != 7]
    x_featured_clean_1 = x_featured_clean_1[x_featured_clean_1[:, 4] != 7]

    ###################################################

    #6 : TOLDHI2 high blood cholesterol binary variable : change 1 (yes) to 0 and 2 (no) to 1, and remove 9 and 7
    x_featured_clean_1[x_featured_clean_1[:, 5] == 1, 5] = 0
    x_featured_clean_1[x_featured_clean_1[:, 5] == 2, 5] = 1
    y_clean = y_clean[x_featured_clean_1[:, 5] != 9]
    ids = ids[x_featured_clean_1[:, 5] != 9]
    x_featured_clean_1 = x_featured_clean_1[x_featured_clean_1[:, 5] != 9]
    y_clean = y_clean[x_featured_clean_1[:, 5] != 7]
    ids = ids[x_featured_clean_1[:, 5] != 7]
    x_featured_clean_1 = x_featured_clean_1[x_featured_clean_1[:, 5] != 7]

    ###################################################

    #7 CVDSTRK3 ever had a stroke binary variable : change 1 (yes) to 0 and 2 (no) to 1, and remove 9 and 7
    x_featured_clean_1[x_featured_clean_1[:, 6] == 1, 6] = 0
    x_featured_clean_1[x_featured_clean_1[:, 6] == 2, 6] = 1
    y_clean = y_clean[x_featured_clean_1[:, 6] != 9]
    ids = ids[x_featured_clean_1[:, 6] != 9]
    x_featured_clean_1 = x_featured_clean_1[x_featured_clean_1[:, 6] != 9]
    y_clean = y_clean[x_featured_clean_1[:, 6] != 7]
    ids = ids[x_featured_clean_1[:, 6] != 7]
    x_featured_clean_1 = x_featured_clean_1[x_featured_clean_1[:, 6] != 7]

    ###################################################

    #8 : DIABETE3 : ever had diabete : yes (1) or during pregnancy (2) to 0, pre-diabete or borderline (4) to 1, no (3) to 2, and remove 9 and 7
    x_featured_clean_1[x_featured_clean_1[:, 7] == 1, 7] = 0
    x_featured_clean_1[x_featured_clean_1[:, 7] == 2, 7] = 0
    x_featured_clean_1[x_featured_clean_1[:, 7] == 4, 7] = 1
    x_featured_clean_1[x_featured_clean_1[:, 7] == 3, 7] = 2
    y_clean = y_clean[x_featured_clean_1[:, 7] != 9]
    ids = ids[x_featured_clean_1[:, 7] != 9]
    x_featured_clean_1 = x_featured_clean_1[x_featured_clean_1[:, 7] != 9]
    y_clean = y_clean[x_featured_clean_1[:, 7] != 7]
    ids = ids[x_featured_clean_1[:, 7] != 7]
    x_featured_clean_1 = x_featured_clean_1[x_featured_clean_1[:, 7] != 7]

    ###################################################

    #9 : SEX : male (1) to 0 and female (2) to 1
    x_featured_clean_1[x_featured_clean_1[:, 8] == 1, 8] = 0
    x_featured_clean_1[x_featured_clean_1[:, 8] == 2, 8] = 1

    ###################################################

    #10 : EDUCA ordinal variable : 1 = never attended school -> 6 = college graduate or above. 9 = missing values that need to be dropped
    #remove all missing values
    y_clean = y_clean[x_featured_clean_1[:, 9] != 9]
    ids = ids[x_featured_clean_1[:, 9] != 9]
    x_featured_clean_1 = x_featured_clean_1[x_featured_clean_1[:, 9] != 9]

    ###################################################

    #11 : INCOME2 remove 77 and 99 (already ordinal)
    #remove 77 and 99
    y_clean = y_clean[x_featured_clean_1[:, 10] != 77]
    ids = ids[x_featured_clean_1[:, 10] != 77]
    x_featured_clean_1 = x_featured_clean_1[x_featured_clean_1[:, 10] != 77]
    y_clean = y_clean[x_featured_clean_1[:, 10] != 99]
    ids = ids[x_featured_clean_1[:, 10] != 99]
    x_featured_clean_1 = x_featured_clean_1[x_featured_clean_1[:, 10] != 99]

    ###################################################

    #12 : DIFFWALK : binary :Do you have serious difficulty walking or climbing stairs? 1 = yes, 2 = no, 7 = don't know, 9 = missing values
    #remove all missing values
    y_clean = y_clean[x_featured_clean_1[:, 11] != 7]
    ids = ids[x_featured_clean_1[:, 11] != 7]
    x_featured_clean_1 = x_featured_clean_1[x_featured_clean_1[:, 11] != 7]
    y_clean = y_clean[x_featured_clean_1[:, 11] != 9]
    ids = ids[x_featured_clean_1[:, 11] != 9]
    x_featured_clean_1 = x_featured_clean_1[x_featured_clean_1[:, 11] != 9]

    #change 1 (yes) to 0 and 2 (no) to 1

    x_featured_clean_1[x_featured_clean_1[:, 11] == 1, 11] = 0
    x_featured_clean_1[x_featured_clean_1[:, 11] == 2, 11] = 1

    ###################################################

    #13 : SMOKE100 : binary variable : Have you smoked at least 100 cigarettes in your entire life? 1 = yes, 2 = no, 7 = don't know, 9 = missing values
    #remove all missing values
    y_clean = y_clean[x_featured_clean_1[:, 12] != 7]
    ids = ids[x_featured_clean_1[:, 12] != 7]
    x_featured_clean_1 = x_featured_clean_1[x_featured_clean_1[:, 12] != 7]
    y_clean = y_clean[x_featured_clean_1[:, 12] != 9]
    ids = ids[x_featured_clean_1[:, 12] != 9]
    x_featured_clean_1 = x_featured_clean_1[x_featured_clean_1[:, 12] != 9]

    #change 1 (yes) to 0 and 2 (no) to 1

    x_featured_clean_1[x_featured_clean_1[:, 12] == 1, 12] = 0
    x_featured_clean_1[x_featured_clean_1[:, 12] == 2, 12] = 1

    ###################################################

    #14 : _RFHYPE5 : Adults who have been told they have high blood pressure by a doctor. Yes (2) to 0, no (1) to 1, 7 = don't know, 9 = missing values

    #remove all missing values
    y_clean = y_clean[x_featured_clean_1[:, 13] != 7]
    ids = ids[x_featured_clean_1[:, 13] != 7]
    x_featured_clean_1 = x_featured_clean_1[x_featured_clean_1[:, 13] != 7]
    y_clean = y_clean[x_featured_clean_1[:, 13] != 9]
    ids = ids[x_featured_clean_1[:, 13] != 9]
    x_featured_clean_1 = x_featured_clean_1[x_featured_clean_1[:, 13] != 9]

    #change 1 (yes) to 0 and 2 (no) to 1

    x_featured_clean_1[x_featured_clean_1[:, 13] == 1, 13] = 1
    x_featured_clean_1[x_featured_clean_1[:, 13] == 2, 13] = 0

    ###################################################

    #15 : _CHOLCHK : Cholesterol check within past five years. No = {2,3} to 0, Yes = {1}, remove missing values
    #remove all missing values
    y_clean = y_clean[x_featured_clean_1[:, 14] != 7]
    ids = ids[x_featured_clean_1[:, 14] != 7]
    x_featured_clean_1 = x_featured_clean_1[x_featured_clean_1[:, 14] != 7]
    y_clean = y_clean[x_featured_clean_1[:, 14] != 9]
    ids = ids[x_featured_clean_1[:, 14] != 9]
    x_featured_clean_1 = x_featured_clean_1[x_featured_clean_1[:, 14] != 9]

    #change 2 and 3 to 0 

    x_featured_clean_1[x_featured_clean_1[:, 14] == 2, 14] = 0
    x_featured_clean_1[x_featured_clean_1[:, 14] == 3, 14] = 0

    ###################################################


    #16 : _AGEG5YR : ordinal varible to represent the age groupe -> remove 14 as missing values
    #remove all missing values

    y_clean = y_clean[x_featured_clean_1[:, 15] != 14]
    ids = ids[x_featured_clean_1[:, 15] != 14]
    x_featured_clean_1 = x_featured_clean_1[x_featured_clean_1[:, 15] != 14]

    ###################################################

    #17 : _BMI5 : Body mass index (kg/m**2) : divide by 100
    x_featured_clean_1[:, 16] = x_featured_clean_1[:, 16] / 100

    ###################################################

    #18 : _RFDRHV5 : Heavy drinkers, binary. yes (2) to 0 and no (1) to 1, remove missing values
    x_featured_clean_1[x_featured_clean_1[:, 17] == 2, 17] = 0

    #remove all missing values

    y_clean = y_clean[x_featured_clean_1[:, 17] != 9]
    ids = ids[x_featured_clean_1[:, 17] != 9]
    x_featured_clean_1 = x_featured_clean_1[x_featured_clean_1[:, 17] != 9]

    ###################################################

    #19 : _FRTLT1 : Consume Fruit 1 or more times per day : change 1 = yes to 0, 2 = less than 1 time per day to 1, 9 = missing values
    x_featured_clean_1[x_featured_clean_1[:, 18] == 1, 18] = 0
    x_featured_clean_1[x_featured_clean_1[:, 18] == 2, 18] = 1

    #remove all missing values
    y_clean = y_clean[x_featured_clean_1[:, 18] != 9]
    ids = ids[x_featured_clean_1[:, 18] != 9]
    x_featured_clean_1 = x_featured_clean_1[x_featured_clean_1[:, 18] != 9]

    ###################################################

    #20 : _VEGLT1 : Consume Vegetables 1 or more times per day : change 1 = yes to 0, 2 = less than 1 time per day to 1, 9 = missing values
    x_featured_clean_1[x_featured_clean_1[:, 19] == 1, 19] = 0
    x_featured_clean_1[x_featured_clean_1[:, 19] == 2, 19] = 1

    #remove all missing values
    y_clean = y_clean[x_featured_clean_1[:, 19] != 9]
    ids = ids[x_featured_clean_1[:, 19] != 9]
    x_featured_clean_1 = x_featured_clean_1[x_featured_clean_1[:, 19] != 9]

    ###################################################

    #21 : _TOTINDA : Adults who reported doing physical activity or exercise during the past 30 days other than their regular job. 
    #change 1 = yes to 0, 2 = less than 1 time per day to 1, 9 = missing values

    x_featured_clean_1[x_featured_clean_1[:, 20] == 1, 20] = 0
    x_featured_clean_1[x_featured_clean_1[:, 20] == 2, 20] = 1

    #remove all missing values
    y_clean = y_clean[x_featured_clean_1[:, 20] != 9]
    ids = ids[x_featured_clean_1[:, 20] != 9]
    x_featured_clean_1 = x_featured_clean_1[x_featured_clean_1[:, 20] != 9]

    return x_featured_clean_1, y_clean, ids