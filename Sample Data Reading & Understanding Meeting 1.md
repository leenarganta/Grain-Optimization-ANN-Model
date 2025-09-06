
### Columns Overview:
##### Inputs (Features/Independent Variables):
- X1(Mpa) -> pressure 
- X2(Celsius) -> Temperature
- X3(Seconds) -> Time

#### Outputs (Targets/Dependent Variables):
- Y1(%) -> Quality Metric 1 (% yield?) 
	- goal: maximize!
- Y2(%) -> Quality Metric 2 
	- goal: minimize
- Y3(-) -> Quality Score (normalized) 
	- goal: minimize
- Y4(%) -> Another Desirable Quality
	- goal: maximize!
- Y5(%) -> Defect/impurity rate
	- goal: minimize
---

| Column Name    | Description                | Matches Problem Statement? | Notes                           |
| -------------- | -------------------------- | -------------------------- | ------------------------------- |
| `X1 (Mpa)`     | Pressure (MPa)             | Yes                        | Will be normalized for modeling |
| `X2(Celcius)`  | Temperature (°C)           | Yes                        |                                 |
| `X3 (seconds)` | Time (seconds)             | Yes                        |                                 |
| `Y1(%)`        | Main quality metric to max | Yes                        | Importance = 5                  |
| `Y2(%)`        | Minimize                   | Yes                        | Importance = 2                  |
| `Y3(-)`        | Minimize                   | Yes                        | Importance = 3                  |
| `Y4(%)`        | Maximize                   | Yes                        | Importance = 4                  |
| `Y5(%)`        | Minimize                   | Yes                        | Importance = 3                  |

---
### Initial Observations:
- X1-X3 values are outside the final optimization ranges. (ex. X1 = 200 MPa, should be 400–600 MPa in final analysis).
---
### Project Goal Recap: 
We are predicting 5 target quality outcomes (Y1–Y5) from 3 input conditions (X1: Pressure, X2: Temperature, X3: Time). The optimization challenge is to find the combination of X1–X3 that:

- **Maximizes** Y1 (most important) and Y4
    
- **Minimizes** Y2, Y3, and Y5
    
- While ensuring **X1–X3 and Y1–Y5 stay within defined bounds**
    

This is a **multi-output regression** + **constrained multi-objective optimization** problem.

---
#### Weeks 1–2: Foundation Building

I started by learning **multi-output regression**, which means instead of training five separate models, I use one framework that can predict all five Y values from the three Xs. This helps keep the model consistent and organized.

To check how well the model is working, I learned to use **three key metrics**:

- **R²** tells me how much of the variation in the data the model can explain.
    
- **RMSE** and **MAE** help me measure prediction errors — RMSE especially penalizes big mistakes.
    

I also explored **preprocessing** the data. For example, I make sure everything is in the right range, there are no missing values, and that I’m respecting any constraints in the problem — like pressure staying between 400 and 600.

Before jumping into modeling, I also did **EDA**, or exploratory data analysis. I made heatmaps and scatterplots to better understand how each variable is related. This helped me spot patterns and understand which inputs matter most for each output.

#### Weeks 3–4: Model Building and Interpretability

Next, I split my dataset into training and test sets so I can evaluate the model fairly. This helps prevent overfitting — where the model memorizes the training data but doesn’t generalize.

I started by using a **Random Forest Regressor** because it’s very good at capturing patterns and it works well out of the box with this kind of structured data.

After that, I moved to **XGBoost**, which is like a smarter version of decision trees. It builds trees one after another and fixes errors as it goes, which makes it more powerful and accurate.

Finally, I’m using **SHAP values** to understand which features influence the model’s predictions the most. This is important because it gives me insight into **why** the model is predicting certain things — not just what it predicts. For example, SHAP can show that temperature has a stronger effect on Y4 than time does, and that helps explain the system better.

#### Final Goal

The end goal is not just to predict quality, but to **find the best pressure, temperature, and time settings** that will:

- Maximize Y1 and Y4 (the most important qualities),
    
- Minimize things like Y2, Y3, and Y5 (which represent defects),
    
- And still stay within all the constraints.

---
- artificial neural network model that combines all factors. 
- connected through hidden layers, weights and biases
- the model plan is a feed forward back propagation ANN.
	- x to y is in the forward direction. 
	- the model will automatically predict Y based on X using the data that is given and the outputs given.
	- The network should be built to where the system automatically assumes what the Y value will be when actual is about 105. The difference is 5. That is the error. The Y is now back propagated of error.
- use AI to find HOW X and Y are related. There are hidden layers, between the path of X and Y, each line between a x and a layer is "weight"
- how weights between each layer between X and Y are measured and correlation, so it can identify this in unknown data
- model is trained on 70-80% data and the unknown 20%  is used for validating the model.
![[Pasted image 20250725192209.png]]- The B values are Bias
there is possibility for multiple Y-values to be an output as well. 
![[Pasted image 20250725192623.png]]
- Like this. Error is minimized on the bias values
- This is called network development. This does not provide optimized condition. We have to learn genetic algorithm. 