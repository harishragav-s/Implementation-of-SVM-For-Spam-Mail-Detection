# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and Preprocess Data: Load the spam dataset, label "spam" as 1 and "ham" as 0, and clean the text data.
2. Split Data: Divide the dataset into training and testing sets.
3. Train SVM Model: Train an SVM model with a linear kernel on the training set.
4. Evaluate Model: Test the model on the test set and evaluate using accuracy, precision, recall, and a confusion matrix.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: HARISH RAGAV S
RegisterNumber:  212222110013
*/

```


```
import chardet
file='spam.csv'
with open(file,'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result
import pandas as pd 
data=pd.read_csv("spam.csv",encoding="Windows-1252")
data.head()
data.info()
data.isnull().sum()
x=data["v2"].values
y=data["v1"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
## Output:
### OUTPUT 1: ![image](https://github.com/user-attachments/assets/a737d693-b178-40b9-b3b4-c9a1d2d04cb9)

### OUTPUT 2:  ![image](https://github.com/user-attachments/assets/ed3eb8c6-3c73-47bf-9157-8ad687e38329)

### OUTPUT 3:  ![image](https://github.com/user-attachments/assets/1b5140d2-cf52-4545-8aad-a34188031356)

### OUTPUT 4:   ![image](https://github.com/user-attachments/assets/4775af96-c825-4c5c-b075-b0703f37ed4f)

### OUTPUT 5:   ![image](https://github.com/user-attachments/assets/14432860-613a-4c0b-a8cd-442c28d446bf)

### OUTPUT 6:   ![image](https://github.com/user-attachments/assets/60bec89e-c266-4fe8-b9de-96e621b94ec7)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
