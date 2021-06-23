# Loading Libraries...

#Pandas Library
import pandas as pd

#Numpy Library
import numpy as np

#SVR classifier
from sklearn.svm import SVR

#KNN for distances
from sklearn.neighbors import NearestNeighbors

#Scale data
from sklearn import preprocessing

#Splitting the data
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

#Shuffle data
from sklearn.utils import shuffle

#Bayesian Optimization
from skopt import BayesSearchCV

#Metrics
# mean_absolute_error(y_true, y_pred)
# mean_squared_error(y_true, y_pred)
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error

#Time
import time

#Math Functions
import math


def svr_subsample_bayes_train(trainD,testD,sig,ep,y_label_name,kernel_type,num_neighbors,rs=45):

    """
    Implementation of algorithm #2 based on Nearest neighbors methods for support vector machines, 
    Camelo, S. A.,González-Lima, M. D.,Quiroz, A. J. Adapted for SVR.

    :param trainD: Train data
    :param testD: Test data
    :param sig: Percentage of the subsampleT (0.01 or  0.1)
    :param ep: Percentage of R set (usually 0.1)
    :param y_label_name: Column name of the target variable
    :param kernel_type: Type of kernel ('linear', 'poly', 'rbf')
    :param num_neighbors: Number of neighbors to look for (5 when sig=0.01 or 3 when sig=0.1)
    :param rs: random_state seed for the reproducibility of the experiment
    :return: The best trained SVR Model
    """

    print("####### BAYESIAN_OPT MODE #######")
    #Start time 
    start = time.time()

    #1. Select a random subsample T(0) of size δn from the training data set D of size n.
    subsampleT = trainD.sample(frac=sig,random_state=rs)

    #2. Initialize S(0) = D\T (0) and j = 0.
    subsample_index=list(subsampleT.index)
    SetS=trainD.drop(subsample_index,axis=0)

    #3. Solve the SVR problem for T(0) and identify the support vectors V(0)...
    # We use bayesian optomization for hyperparameters

    svr_0 = BayesSearchCV(
    SVR(gamma='scale'),
    {
        'C': (1e-2, 1e+3,'log-uniform'),
        'degree': (2, 5),
        'epsilon': (1e-2, 1e+2,'log-uniform'),  # integer valued parameter
        'kernel': [kernel_type]  # categorical parameter
    },
    cv=3,
    n_jobs=3,
    verbose=3
    )

    #Train the model
    startTrain0 = time.time()
    svr_0.fit(subsampleT.loc[:,subsampleT.columns!=y_label_name], subsampleT.loc[:,y_label_name])
    endTrain0 = time.time()
    print("++++++ TRAIN TIME (0) "+str(endTrain0 - startTrain0)+" +++++++++++")

    print("####### BEST MODEL PARAMS INITIAL STEP (0) #####")
    print(svr_0.best_params_)
    print("Coefficient of determination R^2 of the prediction: "+"\n")
    e0=svr_0.score(testD.loc[:,testD.columns!=y_label_name], testD.loc[:,y_label_name])
    print(e0)
    pred0 = svr_0.predict(testD.loc[:,testD.columns!=y_label_name])
    # mean_absolute_error(y_true, y_pred)
    # mean_squared_error(y_true, y_pred)
    print("MAE: ",mean_absolute_error(testD.loc[:,y_label_name], pred0))
    print("RMSE: ",math.sqrt(mean_squared_error(testD.loc[:,y_label_name], pred0)))
    print("####### BEST MODEL PARAMS INITIAL STEP (0) #####")

    #4. Consider only the new support vectors V(0)
    #We take support vector indices
    support_vec_index=list(svr_0.best_estimator_.support_)
    #We build the support vectors dataframe
    support_matrix=subsampleT.iloc[support_vec_index]

    #For the previously chosen value of k, find the k-nearest-neighbors in S(0) 
    # of each support vector in V(0).
    nbrs = NearestNeighbors(n_neighbors=num_neighbors, algorithm='auto').fit(SetS.loc[:,SetS.columns!=y_label_name].to_numpy()) #Should I include the y?

    #finding the indices 
    arr_knn=nbrs.kneighbors(support_matrix.loc[:,support_matrix.columns!=y_label_name].to_numpy(), return_distance=False)
    knn = [item for sublist in arr_knn for item in sublist]
    #Remove repeated vectors
    knn_indices=list(dict.fromkeys(knn))

    #set of nearest neighbors
    knn_data=SetS.iloc[knn_indices]

    #5. Redefine the subsample as T( j+1) and define R and s+1

    #R
    sR=SetS.drop(list(knn_data.index),axis=0)
    SetR=sR.sample(frac=ep,random_state=rs)

    #T+1=knn_data+support_matrix+setR
    subsampleT1=pd.concat([knn_data, support_matrix, SetR], axis=0).reset_index(drop=True)
    #Shuffle it
    subsampleT1 = shuffle(subsampleT1).reset_index(drop=True)

    #S+1. S+1= S\(knn U setR)
    knn_setR_union=list(SetR.index)+list(knn_data.index)
    SetS1=SetS.drop(knn_setR_union,axis=0)


    #Solve the SVR problem for T+1 svr_1

    svr_1 = BayesSearchCV(
    SVR(gamma='scale'),
    {
        'C': (1e-2, 1e+3,'log-uniform'),
        'degree': (2, 5),
        'epsilon': (1e-2, 1e+2,'log-uniform'),  # integer valued parameter
        'kernel': [kernel_type]  # categorical parameter
    },
    cv=3,
    n_jobs=3,
    verbose=3
    )

    startTrain1 = time.time()
    svr_1.fit(subsampleT1.loc[:,subsampleT1.columns!=y_label_name], subsampleT1.loc[:,y_label_name])
    endTrain1 = time.time()
    print("++++++ TRAIN TIME (1) "+str(endTrain1 - startTrain1)+" +++++++++++")
    

    print("####### BEST MODEL PARAMS STEP 1 #####")
    print(svr_1.best_params_)
    print("Coefficient of determination R^2 of the prediction: "+"\n")
    e1=svr_1.score(testD.loc[:,testD.columns!=y_label_name], testD.loc[:,y_label_name])
    print(e1)
    diff=e0-e1
    pred1 = svr_1.predict(testD.loc[:,testD.columns!=y_label_name])
    # mean_absolute_error(y_true, y_pred)
    # mean_squared_error(y_true, y_pred)
    print("MAE: ",mean_absolute_error(testD.loc[:,y_label_name], pred1))
    print("RMSE: ",math.sqrt(mean_squared_error(testD.loc[:,y_label_name], pred1)))
    print("DIFF with previous: ", diff)
    print("####### BEST MODEL PARAMS STEP 1 #####")

    #If there's no significant improvement
    if diff >= 0.05:
        print("Iteration process...")
        last_df=subsampleT1.copy()
        SetR_previous=SetR.copy()
        SetS_next=SetS1.copy()
        eprev=e1
        stopFlag=True
        count=0
        svLast=svr_1
        while stopFlag:
            indx_sv=list(svLast.best_estimator_.support_)
            sv_df=last_df.iloc[indx_sv]
            #consider only the new support vectors
            sv_intersect=pd.merge(sv_df, SetR_previous, how='inner')
            nbrs2 = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(SetS_next.loc[:,SetS_next.columns!=y_label_name].to_numpy())
            arr_knn2=nbrs2.kneighbors(sv_intersect.loc[:,sv_intersect.columns!=y_label_name].to_numpy(), return_distance=False)
            knn2=[item for sublist in arr_knn2 for item in sublist]
            knn_indices2=list(dict.fromkeys(knn2))
            knn_data2=SetS_next.iloc[knn_indices2]
            sR2=SetS_next.drop(list(knn_data2.index),axis=0)
            SetR2=sR2.sample(frac=ep,random_state=rs)
            SetR_previous=SetR2.copy()
            last_df=pd.concat([knn_data2, sv_intersect, SetR2], axis=0).reset_index(drop=True)
            #Shuffle it
            last_df = shuffle(last_df).reset_index(drop=True)
            #knn U setR
            knn2_setR2_union=list(SetR2.index)+list(knn_data2.index)
            SetS_next=SetS_next.drop(knn2_setR2_union,axis=0)
            svr_iter = BayesSearchCV(
            SVR(gamma='scale'),
            {
                'C': (1e-2, 1e+3,'log-uniform'),
                'degree': (2, 5),
                'epsilon': (1e-2, 1e+2,'log-uniform'),  # integer valued parameter
                'kernel': [kernel_type]  # categorical parameter
            },
    		cv=3,
    		n_jobs=3,
    		verbose=3
            )

            #Train the model
            startTrain_iter = time.time()
            svr_iter.fit(last_df.loc[:,last_df.columns!=y_label_name], last_df.loc[:,y_label_name])
            endTrain_iter = time.time()
            print("++++++ TRAIN TIME (ITER) "+str(endTrain_iter - startTrain_iter)+" +++++++++++")

            print("####### BEST MODEL PARAMS ITER " + str(count)+" #####")
            print(svr_iter.best_params_)
            print("Coefficient of determination R^2 of the prediction: "+"\n")
            e_iter=svr_iter.score(testD.loc[:,testD.columns!=y_label_name], testD.loc[:,y_label_name])
            print(e_iter)
            diff2=eprev-e_iter
            pred_iter = svr_iter.predict(testD.loc[:,testD.columns!=y_label_name])
            # mean_absolute_error(y_true, y_pred)
            # mean_squared_error(y_true, y_pred)
            print("MAE: ",mean_absolute_error(testD.loc[:,y_label_name], pred_iter))
            print("RMSE: ",math.sqrt(mean_squared_error(testD.loc[:,y_label_name], pred_iter)))
            print("DIFF with previous: ", diff2)
            print("####### BEST MODEL PARAMS ITER " + str(count)+" #####")
            count=count+1
            eprev=e_iter
            if diff2 < 0.05:
                stopFlag=False
                svLast=svr_iter
                end = time.time()
                print(str("!!!! TOTAL ELAPSED TIME: ")+ (str(end - start))+" !!!!")
        
        #return the best estimator        
        return svLast

    else:

        #End time
        end = time.time()
        print(str("!!!! TOTAL ELAPSED TIME: ")+ (str(end - start))+" !!!!")

        #return the best estimator
        return svr_1
