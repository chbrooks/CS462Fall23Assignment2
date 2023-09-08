from sklearn.datasets import load_iris
from collections import Counter
from sklearn.model_selection import train_test_split, cross_val_score


def ZeroR(data) :
    c = Counter(data)
    return c.most_common(1)

## return a random element from the dataset
## you implement this.
def RandR(data) :
    pass

## load in the iris dataset.
features, classifications = load_iris(return_X_y=True)

pairs = zip(features, classifications)

## use ZeroR and RandR to predict classifications.
ZeroRScore = 0
RandRScore = 0

for item in pairs :
    pred1 = ZeroR(classifications)
    print(pred1)
    pred2 = ZeroR(classifications)
    if pred1 == item[1] :
        ZeroRScore += 1
    if pred2 == item[1] :
        RandRScore += 1

print("ZeroR accuracy: %f RandR accuracy" % (pred1 / len(pairs), pred2 / len(pairs)))

### Let's split the training and test set.

X_train, X_test, y_train, y_test =  train_test_split(features, classifications,
                                                     test_size=0.2)

## Now you do it by hand.



## Randomly shuffle the features and classifications (keep them aligned with each other!)
## and then assign 80% to X_train/y_train and 20% to X_test/y_test

## your code goes here.

## five-fold cross-validation.
## If we use a built-in sklearn estimator, we can get this for free:

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
print(cross_val_score(gnb, features, classifications, cv=5))

## But it's good practice to do this once by hand.
## Write a loop that does five-fold cross validation. Break your dataset into five bins and
## (five times) train on four of the bins and then test on the fifth.
## Average the accuracy across all five iterations.






