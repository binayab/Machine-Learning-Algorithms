# Machine-Learning-Algorithms

my_cross_val(method,X,r,k) performs k-fold cross-validation on the data (X,r) (X is a N x d matrix where the rows are the samples and columns are the features, and r is a N-dimensional vector of class labels) using method, and returns the error rate in each fold. 
Here, the code, reports the error rates in each fold as well as the mean and standard deviation of error rates across folds for three methods: LinearSVC, SVC and Logistic Regression, applied to three classification datasets: Boston50, Boston75 and Digits
