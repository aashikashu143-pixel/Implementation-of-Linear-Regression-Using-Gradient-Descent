# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required library and read the dataframe.
2. Write a function computeCost to generate the cost function.
3. Perform iterations og gradient steps with learning rate. 
4. Plot the Cost function using Gradient Descent and generate the required graph.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: AASHIK>A
RegisterNumber: 25012808 
*/
```
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("C:/Users/acer/Downloads/50_Startups.csv")
x = data["R&D Spend"].values
y = data["Profit"].values
x = (x - np.mean(x)) / np.std(x)
w = 0.0   
b = 0.0      
alpha = 0.01   
epochs = 100
n = len(x)

losses = []
for i in range(epochs):
    y_hat = w * x + b

    loss = np.mean((y_hat - y) ** 2)
    losses.append(loss)


    dw = (2/n) * np.sum((y_hat - y) * x)
    db = (2/n) * np.sum(y_hat - y)

    w = w - alpha * dw
    b = b - alpha * db
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(losses)
plt.xlabel("Iterations")
plt.ylabel("Loss (MSE)")
plt.title("Loss vs Iterations")

plt.subplot(1, 2, 2)
plt.scatter(x, y, label="Data")
plt.plot(x, w * x + b, label="Regression Line")
plt.xlabel("R&D Spend (scaled)")
plt.ylabel("Profit")
plt.title("Linear Regression using Gradient Descent")
plt.legend()

plt.tight_layout()
plt.show()


print("Final Weight (w):", w)
print("Final Bias (b):", b)
```

## Output:
<img width="602" height="581" alt="542724794-b55dac67-bde8-44a6-a54e-b1322a66f7cf" src="https://github.com/user-attachments/assets/eb7c2f6f-b9ca-43da-bdfe-cd4583848766" />

<img width="582" height="583" alt="542724627-61b6d32d-ada6-4e0c-aeb8-496ae670b8d5" src="https://github.com/user-attachments/assets/0f8af3de-da37-4cc0-947c-6e5d0cd91c3f" />

<img width="393" height="50" alt="542724489-df1b1027-c789-4ad0-ad28-ca771eb20f2b" src="https://github.com/user-attachments/assets/6d0dbb60-3f76-4348-a5b5-3b90f47cd246" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
