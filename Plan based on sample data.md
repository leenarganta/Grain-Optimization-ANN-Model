 [[Sample data provided by Dr. Chakraborty]]
---
---
###  **Data Details:**

- **Inputs (Independent variables)**:
    
    - `X1`: Pressure (MPa)
        
    - `X2`: Temperature (°C)
        
    - `X3`: Time (seconds)
        
- **Outputs (Dependent variables / Targets)**:
    
    - `Y1` to `Y5`: Bread quality or chemical/structural properties (% or normalized)
        

###  **Goals (Multi-Objective):**

- **Maximize**: `Y1` (most important), `Y4`
    
- **Minimize**: `Y2`, `Y3`, `Y5`
    
- With constraints on `X1–X3` and output bounds for `Y1–Y5`
    

> So, this is now **clearly a multi-output regression problem with optimization** layered on top.

---
### Weeks 1 & 2: 
| Skill                                                 | Resource                                                                                                                             |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| Multi-output regression (predicting Y1–Y5 from X1–X3) | [Scikit-learn MultiOutputRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputRegressor.html) |
| Regression evaluation: R², RMSE, MAE per output       | [Model Evaluation Metrics](https://towardsdatascience.com/regression-model-metrics-in-python-part-1-mae-mse-rmse-rmsle-9e4c70e53046) |
| Feature scaling + preprocessing                       | [Scikit-learn Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)                                             |
| EDA: heatmaps, pairplots, distributions               | [EDA Examples in Python](https://www.kaggle.com/code/sudalairajkumar/simple-exploration-notebook-zomato-reviews)                     |

---
### Weeks 3 & 4:
|Skill|Resource|
|---|---|
|Train/test split for regression|[Train Test Split Tutorial](https://realpython.com/train-test-split-python-data/)|
|Random Forest Regressor|[Random Forest Regression](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)|
|XGBoost Regressor|[XGBoost Intro](https://xgboost.readthedocs.io/en/stable/python/python_api.html)|
|SHAP or permutation importance (feature impact)|[SHAP Explainer Guide](https://github.com/slundberg/shap)|

---
### Weeks 5 & 6:
|Skill|Resource|
|---|---|
|Constrained optimization with SciPy|[SciPy Optimize.minimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)|
|Weighted multi-objective scoring function|[Multi-objective optimization example](https://towardsdatascience.com/multi-objective-optimization-in-python-7700f9e8e30e)|
|Encoding "importance" weights into cost function|[Linear combination of objectives](https://www.sciencedirect.com/topics/computer-science/multi-objective-optimization)|

---
### Weeks 7 & 8:
|Task|Resource|
|---|---|
|Packaging model (joblib/pickle)|[Save & Load Model](https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/)|
|Plotting results: Actual vs. Predicted|[Seaborn regplot, matplotlib](https://seaborn.pydata.org/generated/seaborn.regplot.html)|
|Writing research paper/report|[Basic Research Template](https://www.overleaf.com/latex/templates/template-for-research-papers/nxqbdtngmshm)|

---
## Objective: 
To develop a multi-output regression model that predicts five quality outcomes (`Y1–Y5`) based on process inputs (`X1–X3`: Pressure, Temperature, and Time). Using this model, we will identify optimal input values that:

- **Maximize** high-priority outputs (`Y1` and `Y4`)
    
- **Minimize** undesirable outputs (`Y2`, `Y3`, `Y5`)
    
- While staying within the defined constraints of all parameters

### **Dataset Description:**

- **Inputs**:
    
    - `X1`: Pressure (400–600 MPa)
        
    - `X2`: Temperature (50–60 °C, minimized)
        
    - `X3`: Time (10–20 seconds, minimized)
        
- **Outputs**:
    
    - `Y1`: % — Maximize (importance: 5)
        
    - `Y2`: % — Minimize (importance: 2)
        
    - `Y3`: Score — Minimize (importance: 3)
        
    - `Y4`: % — Maximize (importance: 4)
        
    - `Y5`: % — Minimize (importance: 3)
        

### **Planned Methodology:**

1. **Data Cleaning & Preprocessing**
    
    - Normalize X1–X3
        
    - Visual EDA of Y1–Y5
        
2. **Model Development**
    
    - Train a multi-output regression model (Random Forest / XGBoost)
        
    - Evaluate with R², RMSE per output
        
3. **Optimization**
    
    - Build a weighted scoring function from target goals and importance levels
        
    - Apply constrained optimization to find best X1–X3 combo
        
4. **Deliverables**
    
    - A working model + optimization module
        
    - Visualization dashboard or report
        
    - Research paper draft (if allowed)
        

###  **Final Outcome:**

A tool that accepts pressure, temperature, and time as inputs — or searches for optimal values — to recommend the best configuration that meets all the quality objectives. This work may support scalable and intelligent grain processing in food tech.