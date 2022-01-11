# Machine-Learning-Algorithms
**Datasets**
Boston50, Boston75 and Digits datasets from sklearn

_from sklearn.datasets import load_boston
from sklearn.datasets import load_digits_

**Pre-requisites**
This implementation requires Python 3.8.2 and Conda 4.9.2

**Algorithms: **
1)	**K-fold Cross-Validation:**
my_cross_val(method,X,r,k) performs k-fold cross-validation on the data (X,r).
(X is a N x d matrix where the rows are the samples and columns are the features, and r is a N-dimensional vector of class labels) using method and returns the error rate in each fold. 
**Output**: The code reports the error rates in each fold as well as the mean and standard deviation of error rates across folds for three methods: LinearSVC, SVC and Logistic Regression, applied to three classification datasets: Boston50, Boston75 and Digits
**Running the script: **        
in terminal: > $ python3 my_cross_val.py   #The results are printed on the terminal
Output Example: 
	Error rates for LinearSVC with Boston50:
Fold 1: 0.1568627450980392
Fold 2: 0.43137254901960786
Fold 3: 0.0
Fold 4: 0.2941176470588235
Fold 5: 0.3921568627450981
Fold 6: 0.21568627450980393
Fold 7: 0.12
Fold 8: 0.18000000000000005
Fold 9: 0.040000000000000036
Fold 10: 0.040000000000000036
Mean: 0.18701960784313726
Standard Deviation: 0.140619210633104
--------------------------------------------------
Fold 1: 0.11764705882352944
Fold 2: 0.21568627450980393
Fold 3: 0.039215686274509776
Fold 4: 0.13725490196078427
Fold 5: 0.196078431372549
Fold 6: 0.2941176470588235
Fold 7: 0.42000000000000004
Fold 8: 0.16000000000000003
Fold 9: 0.040000000000000036
Fold 10: 0.09999999999999998
Mean: 0.17200000000000001
Standard Deviation: 0.11110810914822054
--------------------------------------------------
Fold 6: 0.5882352941176471
Fold 7: 0.16000000000000003
Fold 8: 0.14
Fold 9: 0.040000000000000036
Fold 10: 0.040000000000000036
Mean: 0.2595686274509804
Standard Deviation: 0.20494107516712562
--------------------------------------------------
Error rates for SVC with Digits:
Fold 1: 0.033333333333333326
Fold 2: 0.0
Fold 3: 0.050000000000000044
Fold 4: 0.005555555555555536
Fold 5: 0.005555555555555536
Fold 6: 0.011111111111111072
Fold 7: 0.0
Fold 8: 0.005586592178770999
Fold 9: 0.03351955307262566
Fold 10: 0.03351955307262566
Mean: 0.017818125387957785
Standard Deviation: 0.017028714753132383
--------------------------------------------------
Error rates for LogisticRegression with Boston50:
Fold 1: 0.07843137254901966
Fold 2: 0.11764705882352944
Fold 3: 0.0
Fold 4: 0.1568627450980392
Fold 5: 0.1568627450980392
Fold 6: 0.13725490196078427
Fold 7: 0.12
Fold 8: 0.16000000000000003
Fold 9: 0.06000000000000005
Fold 10: 0.040000000000000036
Mean: 0.1027058823529412
Standard Deviation: 0.052685935184295166
--------------------------------------------------
Error rates for LogisticRegression with Boston75:
Fold 1: 0.07843137254901966
Fold 2: 0.11764705882352944
Fold 3: 0.0
Fold 4: 0.1568627450980392
Fold 5: 0.1568627450980392
Fold 6: 0.13725490196078427
Fold 7: 0.12
Fold 8: 0.16000000000000003
Fold 9: 0.06000000000000005
Fold 10: 0.040000000000000036
Mean: 0.1027058823529412
Standard Deviation: 0.052685935184295166
--------------------------------------------------
Error rates for LogisticRegression with Digits:
Fold 1: 0.09444444444444444
Fold 2: 0.03888888888888886
Fold 3: 0.1166666666666667
Fold 4: 0.03888888888888886
Fold 5: 0.05555555555555558
Fold 6: 0.03888888888888886
Fold 7: 0.050000000000000044
Fold 8: 0.07262569832402233
Fold 9: 0.1061452513966481
Fold 10: 0.06145251396648044
Mean: 0.06735567970204842
Standard Deviation: 0.02757564691954651
--------------------------------------------------

**2)	Principal Component Analysis:**
**myPCA(X,k)** function takes two inputs: (1) the original data from the Digits Dataset, and (2) an interger k = 2 indicating how many principle components used for projection. It returns (1) the projection matrix W ∈ R d×2 and (2) the estimated mean of the digits data.

**Projected_data_points** function takes three inputs (1) the original data X ∈ R N×d, (2) the projection matrix learnt from myPCA and (3) the estimated mean µ ∈ R d×1 returned by myPCA. Then the projected data is plotted. 

**Running the script:     **    
In terminal: $ python3 Projected_data_points.py  #the png file is saved as myPCA.png
**
**Output example:** myPCA.png
 


	

