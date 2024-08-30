# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1: Start.

Step 2: Import the standard libraries in python required for finding Gradient Design.

Step  3: Read the dataset file and check any null value using .isnull() method.

Step  4: Declare the default variables with respective values for linear regression.

Step 5: Calculate the loss using Mean Square Error.

Step 7: Predict the value of y.

Step 8: Plot the graph respect to hours and scores using .scatterplot() method for Linear Regression.

Step 9: Plot the graph respect to loss and iterations using .plot() method for Gradient Descent.

Step 10: End

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: LAVANYA S
RegisterNumber: 212223230112
*/

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(x1,y,learning_rate=0.01,num_iters=1000):
  x=np.c_[np.ones(len(x1)),x1]
  theta=np.zeros(x.shape[1]).reshape(-1,1)
  for c in range(num_iters):
    predictions=(x).dot(theta).reshape(-1,1)
    errors=(predictions-y).reshape(-1,1)
    theta-=learning_rate*(1/len(x1))*x.T.dot(errors)
    return theta
data=pd.read_csv('/content/50_Startups.csv',header=None)
x=(data.iloc[1:, :-2].values)
x1=x.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
x1_scaled=scaler.fit_transform(x1)
y1_scaled=scaler.fit_transform(y)
theta=linear_regression(x1_scaled,y1_scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value:{pre}")

```

## Output:
DATA.HEAD()

![Screenshot 2024-08-29 183415](https://github.com/user-attachments/assets/763022ca-616b-4141-942a-109ee33e0446)


prediction
[[-0.42925552]]

Predicted valeue: [[192932.45813544]]



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
