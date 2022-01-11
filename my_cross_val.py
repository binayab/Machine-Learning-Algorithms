import sklearn as sk
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

#calculating accuracy as error rate = 1- accuracy
def acc_score(r_test, r_prediction):
    score = 0.0
    for i in range(0,len(r_test)):
        if(r_test[i] == r_prediction[i]):
            score += 1
    return score / len(r_test) 

# Three different models
def Method(method):
    if(method == "LinearSVC"):
        Method = LinearSVC(max_iter=2000)
    elif(method == "SVC"):
        Method = SVC(gamma='scale', C=10)
    elif(method == "LogisticRegression"):
        Method = LogisticRegression(penalty='l2', solver='lbfgs', multi_class='multinomial', max_iter=5000)
    else:
        print("Try Again!!")
    return Method

#printing the results as F1-F10, Mean and SD
def result(results):
    for i in range(0, len(results)):
        if( i < 10 ):
            print("Fold {}: {}".format(i+1, results[i]))
        elif( i == 10 ):
            print("Mean: {}".format(results[i]))
        elif( i == 11 ):
            print("Standard Deviation: {}".format(results[i]))
    print('--------------------------------------------------')

# ************************** Working on Datasets *****************************************
def Boston50(dataset):
    resp_var = dataset.target

    per50 = np.percentile(resp_var, 50)

    b50 = []
    for i in resp_var:
        if(i >= per50):
            b50.append(1)
        else:
            b50.append(0)

    return np.array(b50)


def Boston75(dataset):
    resp_var = dataset.target

    per75 = np.percentile(resp_var, 75)

    b75 = []
    for i in resp_var:
        if(i >= per75):
            b75.append(1)
        else:
            b75.append(0)

    return np.array(b75)

def q4_dataset():
    from sklearn.datasets import load_boston
    dataset = load_boston()
    b50 = dataset
    b75 = dataset

    b50_resp = Boston50(dataset)
    b75_resp = Boston75(dataset)

    b50["target"] = b50_resp
    b75["target"] = b75_resp

    from sklearn.datasets import load_digits
    digits = load_digits()

    boston = [
                {"Name": "Boston50", "Dataset": b50},
                {"Name": "Boston75", "Dataset": b75},
                {"Name": "Digits", "Dataset": digits},
                ]

    return boston
# ***************************************************************************************

# ************************ Working on my_cross_val() ************************************
def my_cross_val(method, X, r, k):
    
    X_split = []

    r_split = np.array_split(r, k)
    
    total = 0
    for i in r_split:
        nf = len(i)
        X_split.append(X[total:(total + nf)])
        total += nf

    error_rate = []
    
    for j in range(0,k):
        model = Method(method)
        
        X_test = X_split[j]
        r_test = r_split[j]

        X_train = []
        r_train = []

        for l in range(0,k):
            if( l != j):
                X_train.extend(X_split[l])
                r_train.extend(r_split[l])

        model.fit(np.asarray(X_train), np.asarray(r_train))
        r_prediction = model.predict(np.asarray(X_test))

        acc = acc_score(r_test,r_prediction)
        error_rate.append(1 - acc)

        X_test, r_test, X_train, r_train = [], [], [], []        
    avg = np.mean(error_rate)
    stdev = np.std(error_rate)
    
    error_rate.append(avg)
    error_rate.append(stdev)

    result(error_rate)

    return error_rate
# ******************************************************************************************
if __name__ == "__main__":
    datasets = q4_dataset()
    methods = ["LinearSVC", "SVC", "LogisticRegression"]
    for i in methods:
        for j in datasets:
            if(j["Name"] == "Boston50"):
                b50 = j["Dataset"] 
            elif(j["Name"] == "Boston75"):
                b75 = j["Dataset"] 
            elif(j["Name"] == "Digits"):
                digits = j["Dataset"] 
            print("Error rates for {} with {}:".format(i, j["Name"]))
            results = my_cross_val(i,j["Dataset"].data,j["Dataset"].target,10)
