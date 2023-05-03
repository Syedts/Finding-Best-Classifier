# Importing 
import pandas
import numpy as np
import time 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC 
from sklearn.metrics import confusion_matrix , roc_auc_score , roc_curve, auc
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV



#Importing the DataSet
dataset = pandas.read_csv('Skin_NonSkin.csv')

dataset.columns

dataset['Class'] = dataset['Class'].replace({1: 0, 2: 1})

X = dataset[['B','G', 'R']]
Y = dataset['Class']
print(dataset.head())

# Select a random subset of 5000 rows from the dataframe
dataset_small = dataset.sample(n=5000, random_state=42)

# Extract the features and target variables from the smaller dataframe
X_small = dataset_small[['B', 'G', 'R']]
Y_small = dataset_small['Class']

# We are going to do a 70 - 30 split, 30 to test
validation_size = 0.3
seed = 400    
X_train, X_Test_Data, Y_train, Y_Test_Data = train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Parameter grid for GaussianNB
parameters_for_gnb = {'var_smoothing': [0.001,0.01, 0.1, 1, 10, 100]}

# Creating a GaussianNB model
gnb = GaussianNB()

# Going with a grid search with cross-validation
gnbgridsearch_startTime = time.time()
gnbgrid_search = GridSearchCV(gnb, parameters_for_gnb, scoring='accuracy', cv=5)
gnbgrid_search.fit(X_train, Y_train)
gnbgridsearch_endTime = time.time()
gnbgridsearch_final_time = gnbgridsearch_endTime - gnbgridsearch_startTime
# Print the best parameter found
#print("\nBest parameter for GaussianNB: ", gnbgrid_search.best_params_)
#print("Best cross-validated accuracy: ", gnbgrid_search.best_score_)

best_params = gnbgrid_search.best_params_.get('var_smoothing')



gnb_new= GaussianNB(var_smoothing=best_params)
gnb_start_time_train = time.time()
gnb_cv_scores = cross_val_score(gnb_new, X, Y, cv=5)
gnb_new.fit(X_train, Y_train)
gnb_end_time_train = time.time()
gnb_final_time_train = gnb_end_time_train - gnb_start_time_train

gnb_start_time_test = time.time()
y_pred_gnb = gnb_new.predict(X_Test_Data)
gnb_end_time_test = time.time()
gnb_final_time_test = gnb_end_time_test - gnb_start_time_test


fpr_gnb, tpr_gnb, thresholds_roc = roc_curve(Y_Test_Data, y_pred_gnb )
roc_auc_gnb = auc(fpr_gnb,tpr_gnb)
gnb_cm = confusion_matrix(Y_Test_Data,y_pred_gnb)
#print("Gaussian Naive Bayes Confusion Matrix: \n",gnb_cm)


gridsearch_lr_startTime = time.time()
lr_init = LogisticRegression()
parameters_for_lr = {'C': [0.01, 0.1, 1, 10, 100],
                    'solver':['newton-cg', 'lbfgs', 'liblinear','sag','saga'],
                    'penalty':['l1','l2']                  
                    }
grid_search_lr = GridSearchCV(lr_init,parameters_for_lr, scoring=["accuracy","precision"],cv=5, refit="precision",n_jobs=8)

grid_search_lr.fit(X_train, Y_train)

gridsearch_lr_endTime = time.time()
gridsearch_lr_final_time = gridsearch_lr_endTime - gridsearch_lr_startTime
#print("\nBest parameter for Logistic Regression: ", grid_search_lr.best_params_)

best_C_LR = grid_search_lr.best_params_.get('C')
best_penalty_LR = grid_search_lr.best_params_.get('penalty')
best_solver_LR = grid_search_lr.best_params_.get('solver')
#print("Best cross-validated Score: ", grid_search_lr.best_score_)



final_lr = LogisticRegression(penalty=best_penalty_LR,C=best_C_LR,solver=best_solver_LR)
lr_startTime_train = time.time()
final_lr.fit(X_train,Y_train)
lr_endTime_train = time.time()
lr_final_time_train = lr_endTime_train - lr_startTime_train


lr_startTime_test = time.time()
y_pred_lr = final_lr.predict(X_Test_Data)
lr_endTime_test = time.time()
lr_final_time_test = lr_endTime_test - lr_startTime_test

fpr_lr, tpr_lr, thresholds_roc = roc_curve(Y_Test_Data, y_pred_lr )
roc_auc_lr = auc(fpr_lr,tpr_lr)
lr_cm = confusion_matrix(Y_Test_Data,y_pred_lr)


                                                # SVM # 
# Going with Linear Kernal because our dataset is huge , so it is optimal to use the Linear Kernel
# Had to reduce the data to 8000 points ( random ) because 25000 was too much and it took too long 
X_small_train, X_small_Test_Data, Y_small_train, Y_small_Test_Data = train_test_split(X_small, Y_small, test_size=validation_size, random_state=seed)
parameters_for_svm = { 'gamma': [0.001,0.01, 0.1, 1, 10, 100]}
svm_init = SVC(kernel='linear')


svmgridsearch_startTime = time.time()
svm_gridsearch = GridSearchCV(svm_init,parameters_for_svm,scoring='accuracy',cv=5,n_jobs=9) 
# cv we are using the K Fold cross validation and folding it 5 times and n_jobs makes 9 threads to work on it parallelly to reduce computation time
svm_gridsearch.fit(X_small_train, Y_small_train)
svmgridsearch_endTime = time.time()
svmgridsearch_final_time = svmgridsearch_endTime - svmgridsearch_startTime



best_gamma_svm = svm_gridsearch.best_params_.get('gamma')


final_svm = SVC(kernel='linear', gamma=best_gamma_svm)
svm_startTime_train = time.time()
final_svm.fit(X_small_train,Y_small_train)
svm_endTime_train = time.time()
svm_final_time_train = svm_endTime_train - svm_startTime_train

svm_startTime_test = time.time()
y_small_pred_svm = final_svm.predict(X_small_Test_Data)
svm_endTime_test = time.time()
svm_final_time_test  = svm_endTime_test  - svm_startTime_test 

fpr_svm, tpr_svm, thresholds_roc = roc_curve(Y_small_Test_Data, y_small_pred_svm )
roc_auc_svm = auc(fpr_svm,tpr_svm)
svm_cm = confusion_matrix(Y_small_Test_Data,y_small_pred_svm)


# Function to plot the ROC graph
def plot_roc_curve(ax, fpr, tpr, roc_auc, title):
    ax.plot(fpr, tpr, label="ROC AUC = {:0.2f} %".format(roc_auc*100), lw=3, alpha=0.7)
    ax.plot([0, 1], [0, 1], 'r', linestyle="--", lw=2)
    ax.set_xlabel("False Positive Rate", fontsize=14)
    ax.set_ylabel("True Positive Rate", fontsize=14)
    ax.set_title(title)
    ax.legend(loc='best')


fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))

# Plot the ROC curve for GNB classifier 
plot_roc_curve(axs[0], fpr_gnb, tpr_gnb, roc_auc_gnb,"Gaussian Naive Bayes - ROC Curve")

# Logistic Regression ROC plot
plot_roc_curve(axs[1], fpr_lr, tpr_lr, roc_auc_lr,"Logistic Regression - ROC Curve")

# SVM ROC Plot
plot_roc_curve(axs[2], fpr_svm, tpr_svm, roc_auc_svm,"SVM Linear Kernal - ROC Curve")



fig.tight_layout()
print("===================================================================================================\n")
print("===================================================================================================\n")
print("\nExecution time for Cross Validating and finding the best parameter for Gaussian Naive Bayes Classification is: {}".format(gnbgridsearch_final_time))
print(f"Time spent on Training --- Gaussian Naive Bayes : {gnb_final_time_train:.4f} seconds")
print(f"Time spent on Testing  --- Gaussian Naive Bayes : {gnb_final_time_test:.4f} seconds")
print("\nGaussian Naive Bayes Cross-Validation Scores: \n", gnb_cv_scores)
print("\nAverage Cross-Validation Accuracy: {:.2f}".format(gnb_cv_scores.mean()))
print("\nGaussian Naive Bayes Confusion Matrix: \n",gnb_cm)
print("===================================================================================================\n")
print("===================================================================================================\n")
print("\nExecution time for Cross Validating and finding the best parameter for Logistic Regression is:",gridsearch_lr_final_time, "Seconds")
print("\nBest parameter for Logistic Regression: ", grid_search_lr.best_params_)
print("\nBest cross-validated Score: ", grid_search_lr.best_score_)
print(f"Time spent on Training --- Logistic Regression : {lr_final_time_train:.4f} seconds")
print(f"Time spent on Testing  --- Logistic Regression : {lr_final_time_test:.4f} seconds")
print("\nLogistic Regression Confusion Matrix: \n",lr_cm)
print("===================================================================================================\n")
print("===================================================================================================\n")
print("\nExecution time for Cross Validating and finding the best parameter for SVM is:",svmgridsearch_final_time,"Seconds")
print("\nBest parameter for SVM: ", svm_gridsearch.best_params_)
print("\nBest cross-validated Score: ", svm_gridsearch.best_score_)
print(f"Time spent on Training --- SVM : {svm_final_time_train:.4f} seconds")
print(f"Time spent on Testing  --- SVM : {svm_final_time_test:.4f} seconds")
print("\nSVM Confusion Matrix: \n",svm_cm)
print("===================================================================================================\n")
print("GRID SEARCH RESULTS FOR GNB: \n",gnbgrid_search.cv_results_)
print("\n===================================================================================================\n")
print("GRID SEARCH RESULTS FOR LOGISTIC REGRESSION: \n",grid_search_lr.cv_results_)
print("\n===================================================================================================\n")
print("GRID SEARCH RESULTS FOR SVM : \n",svm_gridsearch.cv_results_)
print("\n===================================================================================================\n")
print("\n========================================= DONE =====================================================\n")
plt.show()