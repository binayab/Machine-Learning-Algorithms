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

Output Example: **error_rates.txt**


**2)	Principal Component Analysis:**
**myPCA(X,k)** function takes two inputs: (1) the original data from the Digits Dataset, and (2) an interger k = 2 indicating how many principle components used for projection. It returns (1) the projection matrix W ∈ R d×2 and (2) the estimated mean of the digits data.

**Projected_data_points** function takes three inputs (1) the original data X ∈ R N×d, (2) the projection matrix learnt from myPCA and (3) the estimated mean µ ∈ R d×1 returned by myPCA. Then the projected data is plotted. 

**Running the script:    
In terminal: $ python3 Projected_data_points.py  #the png file is saved as myPCA.png


Output example: **myPCA.png**
 


	

