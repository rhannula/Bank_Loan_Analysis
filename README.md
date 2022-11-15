# [<--](https://rhannula.github.io/Robert_Portfolio/)

## Overview:

The business request for this data analyst project is about Thera Bank which has a growing customer base. Majority of the customers are liability customers (depositors) with varying sizes of deposits. We want to compare to the borrowers (asset customers) and see if there is growth potential for this base which the bank would be interested in bringing more loan business and in the process, earn more through the interest on loans. The management wants to explore ways of converting its liability customers to personal loan customers (while retaining them as depositors). 
The department wants to build a model that will help them identify the potential customers who have a higher probability of converting. For this we will identify through hypothesis testing. 


The tools used in this particular project is Python and its multiple libraries, such as Pandas, Numpy, Matplotlib, Missingno and Scipy Statsmodel.

•• The implementation of the prediction model (Machine Learning) will be implemented in the future. For now, this project will remain as Exploratory Data Analysis (EDA).  



## Data cleansing and transformation

Checking for any missing values in our DataFrame.

![](https://raw.githubusercontent.com/rhannula/Loan_Analysis/main/Pictures/Screenshot%202022-11-09%20at%2002.14.12.png)
![](https://raw.githubusercontent.com/rhannula/Loan_Analysis/main/Pictures/Screenshot%202022-11-09%20at%2002.14.37.png)

To create the necessary data model for doing analysis, we've created one for exploring discretized continuous variables (continuous variables which have been sorted into some kind of category) and another for exploring continuous variables. We may have a quantitative variable(s) in our data set that we want to discretize it or bin it or categorise it based on the values of the variable such that it is 0 or 1.

The goal for this is to figure out how best to process the data so our machine learning model can learn from it.••


Ideally, all the features will be encoded into a numerical value of some kind.

`df_bin = bankloan.copy() # subframe to host discretized continuous variables (binned)`

`df_con = bankloan.copy() # subframe to host continuous variables as it is and not bin/bucket them`


We notice odd values in the “Experience” column so we eliminate the values as there can not be a negative amount of professional experience. 
We also remove unnecessary columns such as “ID” and “Zip Code.”



## Exploratory Data Analysis

Some relevant questions we want to analyse:

- What is the relationship between Personal Loans and different variables?
- What is the distribution of the client population?
- What types of accounts and security do our clients have?
- What is the most important factor to give a personal loan?



## Visualisation and Outliers

Analysis was done with the target questions in mind. Several conclusions were obtained which will be presented below:

The vast majority of depositors do not have a security nor credit account. We see that there is a huge potential to increase revenue by encouraging the existing customers to subscribe to the assets of the organisation. 

![](https://raw.githubusercontent.com/rhannula/Loan_Analysis/main/Pictures/Screenshot%202022-11-09%20at%2014.39.18.png)


When it comes to distribution in relation to personal loan, income and family size were the most influential attributes

![](https://raw.githubusercontent.com/rhannula/Loan_Analysis/main/Pictures/Screenshot%202022-11-09%20at%2014.43.09.png)
![](https://raw.githubusercontent.com/rhannula/Loan_Analysis/main/Pictures/Screenshot%202022-11-09%20at%2014.51.06.png)

As for hypothesis testing to assert the statistical significance of different elements in relation to the personal loan. For this, we tested the following values: Mortgage, Age, Income, Family, Education, CCAvg and Experience. We found out through conducting a T-test that only Age and Experience elements do not have a significant impact on getting a personal loan but the rest does have enough statistical evidence for availing a personal loan. For this hypothesis test, we have set the significance value to 0.05 to ensure 95% confidence rate. 


![](https://raw.githubusercontent.com/rhannula/Loan_Analysis/main/Pictures/pooled-t-statistics.png)

For this analysis, we use two independent sample t-test formula as shown above

### P-Value:
- **Mortgage**: `5.73034172e-024`
- **Age**: `5.84959264e-001`
- **Income**: `0.00000000e+000`
- **Family**: `1.40990407e-005`
- **Education**: `2.70966319e-022`
- **CCAvg**: `3.81568364e-159`
- **Experience**: `3.20753602e-001`



![](https://raw.githubusercontent.com/rhannula/Loan_Analysis/main/Pictures/Screenshot%202022-11-09%20at%2016.33.38.png)

![](https://raw.githubusercontent.com/rhannula/Loan_Analysis/main/Pictures/Screenshot%202022-11-09%20at%2016.46.00.png)
![](https://raw.githubusercontent.com/rhannula/Loan_Analysis/main/Pictures/Screenshot%202022-11-09%20at%2016.52.17.png)
![](https://raw.githubusercontent.com/rhannula/Loan_Analysis/main/Pictures/Screenshot%202022-11-09%20at%2016.52.32.png)
![](https://raw.githubusercontent.com/rhannula/Loan_Analysis/main/Pictures/Screenshot%202022-11-09%20at%2016.52.47.png)
![](https://raw.githubusercontent.com/rhannula/Loan_Analysis/main/Pictures/Screenshot%202022-11-09%20at%2016.53.01.png)
![](https://raw.githubusercontent.com/rhannula/Loan_Analysis/main/Pictures/Screenshot%202022-11-09%20at%2016.53.13.png)





Click [HERE](https://github.com/rhannula/Bank_Loan_Analysis/blob/main/Code/Bank%20Loan%20Analysis.ipynb) to view the entire script.
