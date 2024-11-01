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
In the dataset [weather.csv](https://github.com/DaemianMC/MeXE402_Midterm_CabarrubiaCardenas/blob/main/Linear_Regression/weather.csv), it contains information about weather data, including humidity, wind speed, and precipitation. It can be used to predict temperature patterns using linear regression.
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

   <p align="center">
       <img src="https://github.com/DaemianMC/MeXE402_Midterm_CabarrubiaCardenas/blob/main/Linear_Regression/Visualization/Correlation_Heatmap.png" alt="Correlation Heatmap" width="700" />
       <br><i>Figure 1.1 Correlation Heatmap of Dataset Features</i>
   </p>


   * Visualizing the relation of each independent variable to the dependent variable uisng pairplot function
  
     <p align="center">
       <img src="https://github.com/DaemianMC/MeXE402_Midterm_CabarrubiaCardenas/blob/main/Linear_Regression/Visualization/Pairplot.png" alt="Pairplot" width="700" />
       <br><i>Figure 1.2 Pairwise Relationship of Dataset Features with Correlation to the Dependent Variable</i>
     </p>


   * Visualizing the relationship between the actual data and the predicted data using scatter plot
  
     <p align="center">
       <img src="https://github.com/DaemianMC/MeXE402_Midterm_CabarrubiaCardenas/blob/main/Linear_Regression/Visualization/Scatter.png" alt="Scatter" width="700" />
       <br><i>Figure 1.3 Relationship of Actual Data and Predicted Data</i>
     </p>


   * Visualizing the relationship of each independent variable with the dependent variables using scatter plot
  
     <p align="center">
       <img src="https://github.com/DaemianMC/MeXE402_Midterm_CabarrubiaCardenas/blob/main/Linear_Regression/Visualization/Scatter_2.png" alt="Scatter_2" width="700" />
       <br><i>Figure 1.4 Relationship of Each Independent Variable to the Dependent Variable</i>
     </p>


---

### Results
<p>After doing this midterm project, here are some of our findings:</p>

**R-Squared**
  * jhksdfsd







<br>
<br>

## Logistic Regression
### Introduction
Logistic regression is a statistical method that helps us classify data into two categories, like Yes or No, by predicting the probability of an outcome based on one or more factors
* ### Key Concepts
1. Binary Outcome
   * Logistic regression is mainly used for binary classification, where the outcome falls into one of two categories (e.g., success/failure, yes/no).
2. Logistic Function
   * The core of logistic regression is the logistic function (or sigmoid function), which transforms linear combinations of the input variables into probabilities ranging from 0 to 1:

     **P(Y=1|X) = 1 / (1 + e^(- (β₀ + β₁X₁ + β₂X₂ + ... + βₙXₙ)))**

---

    
### Dataset Description
In the dataset [adult.csv](https://github.com/DaemianMC/MeXE402_Midterm_CabarrubiaCardenas/blob/main/Logistic_Regression/adult.csv), it contains demographic information about individuals in the US, including age, work experience, education level, and their income classification (greater than or equal to $50K or less than $50K). It's a classic example for binary classification using logistic regression to their gender.
* The dependent variable in this dataset is ***sex***
* The independent variables in this dataset are ***age, workclass, education.num, marital-status, occupation, relationship, race, capital.gain, capital.loss, hours.per.week, native.country*** and ***income***

---


### Project Objectives
In doing this project, these are the things we want to achieve with our analysis:
* To analyze the impact of independent variables on the likelihood of the outcome.
* To develop a model to estimate the probability of a binary outcome
* To assess model accuracy and effectiveness using metrics such as *accuracy, precision, recall,* and *ROC-AUC*.

---


### Methodology
For the documentation, here are the step-by-step process we come up to create the program that would be useful in categorizing their ***sex*** using logistic regression:
1. **Importing the dataset**
   * Reading the given dataset which is in CSV file and loading it into Pandas DataFrame as a variable called 'dataset'
2. **Modifying the dataset**
   * We remove the unnecessary variables that are not needed in categorizing their sex
   * We also placed the dependent variable on the last column so it would be easier for us to get the input and output.
3. **Getting the inputs and output**
   * Selecting the inputs or the independent variable then storing them in 'X' variable
   * Selecting the output or the dependent variable then storing them in 'y' variable
   * After that, the inputs and output are converted into a NumPy array
4. **Creating the Training Set and the Test Set**
   * Splitting our dataset to create a Training Set and Test Set
5. **Balancing and Bagging & Stacking**
   * We used SMOTE to generate synthetic samples for the minority class in X_train and y_train
   * Creates a pipeline that applies SMOTE for oversampling the minority class and then trains a RandomForestClassifier
6. **Feature Scaling**
   * We normalize the range of independent variables in a dataset
7. **Building and Training the model**
   * We used logistic regression for classification tasks
   * We perform hyperparameter tuning using GridSearchCV to optimize the accuracy using specified parameters and retrieving the best model.
   * Training the logistic regression regression model using the train set data that we created
8. **Inference**
   * Making the prediction of the datapoints in the test set
9. **Evaluating the model**
    * Computes the confusion matrix by comparing the true labels y_test with the predicted labels y_pred
    * Checked the accuracy by inputting the numbers obtained from the confusion matrix into the formula for computing accuracy
9. **Visualization**
    * Visualizes a classification model's performance by showing counts of true positives, true negatives, false positives, and false negatives, facilitating evaluation of accuracy and other metrics.
      
       <p align="center">
       <img src="https://github.com/DaemianMC/MeXE402_Midterm_CabarrubiaCardenas/blob/main/Logistic_Regression/Visualization/Confusion%20Matrix.png" alt="Confusion Matrix" width="700" />
       <br><i>Figure 2.1 Confusion Matrix</i>
     </p>


    * Visualizes the trade-off between true positive rate and false positive rate across various thresholds, helping assess and optimize binary classification performance
      <p align="center">
       <img src="https://github.com/DaemianMC/MeXE402_Midterm_CabarrubiaCardenas/blob/main/Logistic_Regression/Visualization/ROC.png" alt="Receiver Operating Characteristic" width="700" />
       <br><i>Figure 2.2 Receiver Operating Characteristic</i>
     </p>


    * Visualizes how input changes affect predicted probabilities in binary outcomes, illustrating the decision boundary and the model's sensitivity
      <p align="center">
       <img src="https://github.com/DaemianMC/MeXE402_Midterm_CabarrubiaCardenas/blob/main/Logistic_Regression/Visualization/Sigmoid%20Curve.png" alt="Sigmoid Curve" width="700" />
       <br><i>Figure 2.3 Logistic Function (Sigmoid Curve)</i>
     </p>


    * Visualizes the trade-off between precision and recall across thresholds, aiding in the assessment of model performance, especially in imbalanced datasets
      <p align="center">
       <img src="https://github.com/DaemianMC/MeXE402_Midterm_CabarrubiaCardenas/blob/main/Logistic_Regression/Visualization/Precision.png" alt="Precision-Recall Curve" width="700" />
        <br><i>Figure 2.4 Precision-Recall Curve</i>
     </p>

---


### Results
After doing this midterm project, here are some of our findings:
* Due to having a lot of variables in our dataset, we find it hard to increase the accuracy of the logistic regression
* The relationship between the variables affect the result of your output

---


### Discussion
In terms of reflecting on the result, there are some things we noticed when comparing the two regression methods:
1. **Type of dependent variable**
   * **Linear Regression:** Used for predicting continuous outcomes
   * **Logistic Regression:** Used for predicting binary outcomes
2. **Output Interpretation**
   * **Linear Regression:** Produces a continuous output, which can be interpreted directly as the predicted value.
   * **Logistic Regression:** Produces probabilities that can be interpreted as the likelihood of the outcome belonging to a particular class, often transformed using the sigmoid function.
3. **Performance Metrics**
   * **Linear Regression:** Commonly evaluated using metrics like Mean Squared Error (MSE) or R-squared.
   * **Logistic Regression:** Evaluated using metrics like accuracy, precision, recall, F1-score, and AUC-ROC.
4. Interpretability
   * Both models are generally easy to understand, but in logistic regression, the coefficients reflect changes in the log-odds of the outcome, which can be a bit less straightforward compared to the clear, direct interpretations you get from linear regression.

In summary, comparing linear and logistic regression means looking at how each method handles different types of data, what assumptions they make, how they produce outputs, and how we measure their performance. This understanding helps us choose the right approach for the specific problem we’re trying to solve.

---

