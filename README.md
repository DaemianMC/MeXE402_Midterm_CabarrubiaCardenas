# MeXE402_Midterm_CabarrubiaCardenas
##### Midterm Project - 8
---
## Linear Regression
### Introduction
Linear regression models the relationship between a dependent variable and one or more independent variables by fitting a line (or hyperplane) that best predicts the dependent variable from the independents.
* ### Key Concepts
1. Dependent Variables (y)
   * This is the variable you want to predict or explain.
2. Independent Variables (X)
   * These are the variables that you believe influence the dependent variable.
3. Equation of the line
   * When there are multiple independent variables, the equation extends to:
     
     **Y = b<sub>0</sub> + b<sub>1</sub> X<sub>1</sub> + b<sub>2</sub> X<sub>2</sub> + ... + b<sub>n</sub> X<sub>n</sub> + ε**
   * Here, each b coefficient represents the effect of each corresponding X variable on Y.
     
---


### Dataset Description
In the dataset [weather.csv](weather.csv), it contains information about weather data, including humidity, wind speed, and precipitation. It can be used to predict temperature patterns using linear regression.
* The dependent variable in this dataset is ***Temperature (C)***.
* The independent variables in this dataset are ***Apparent Temperature (C), Humidity, Wind Speed (km/h), Wind Bearings (degrees), Visibility (km)*** and ***Pressure (millibars)*** 

---


### Project Objectives
In doing this project, these are the things we want to achieve with our analysis:
* To understand how the dataset can be used to predict temperature
* To develop a program that accurately predicts temperature using linear regression
* To evaluate the model using various metrics such as ***R-squared, Adjusted R-squared, Mean Squared Error and Root Mean Squared Error***.
* To better understand how to predict temperature using the dataset, we plan to provide visualizations such as a ***correlation heatmap, pairplot, and scatter plot***.

--- 

### Methodology
For the documentation, here are the step-by-step process we come up to create the program that would be useful in predicting the temperature using linear regression:
1. **Importing the dataset**
   * Reading the given dataset which is in CSV file and loading it into Pandas DataFrame as a variable called 'dataset'
2. **Modifying the dataset**
   * We remove the unnecessary variables that are not needed in predicting the temperature
   * We also placed the dependent variable on the last column so it would be easier for us to get the input and output.
3. **Getting the inputs and output**
   * Selecting the inputs or the independent variable then storing them in 'X' variable
   * Selecting the output or the dependent variable then storing them in 'y' variable
   * After that, the inputs and output are converted into a NumPy array
4. **Creating the Training Set and the Test Set**
   * Splitting our dataset to create a Training Set and Test Set
5. **Building and Training the model**
   * Setting up a linear regression model using scikit-learn to build the model
   * Training the linear regression model using the train set data that we created
6. **Inference**
   * Making predictions based on the test set data *(X_test)*
   * Making the prediction of a single data point using the first row of the dataset
7. **Evaluating the Model**
   * Evaluating the performarce of our regression model using the metric R-squared
   * Determining the Adjusted R-Squared to get a more reliable score especially that our dataset has multiple independent variables or features
   * Calculating the Mean Squared Error (MSE)
   * Calculating the Root Mean Squared Error (RMSE) for us to interpret the error in the same units as the R-squared
8. **Visualization**
   * Correlation of the variables to each other
   * Visualizing the relation of each independent variables to the dependent variable uisng pairplot function
   * Visualizing the relationship between the actual data and the predicted data using scatter plot
   * Visualizing the relationship of each independent variables with the dependent variables using scatter plot













## Logistic Regression
* ### Introduction
   * Logistic regression models the probability of a binary outcome using one or more predictors. Unlike linear regression, it predicts probabilities to classify inputs into two classes, such as 0 or 1 (e.g., "yes" or "no"). 

### Dataset Description
&nbsp; &nbsp; [Insert text here]

### Project Objectives
&nbsp; &nbsp; [Insert text here]
