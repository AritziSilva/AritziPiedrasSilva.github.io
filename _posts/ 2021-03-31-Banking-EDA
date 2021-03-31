# Banking EDA 


```python
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import thinkstats2 
import thinkplot
import seaborn as sns 
import statsmodels.formula.api as smf

```


```python
#opens file
df = pd.read_csv('Banking_Kaggle.csv')
```


```python
#rows
print(f'Total Data : {df.shape[0]}')
#columns
print(f'Total Variables : {df.shape[1]}')
#checks missing values 
print(f'Missing Values : {df.isnull().sum().values.sum()}\n')
# df info 
print('Type of Variables in Data:\n')
print(df.info())
```

    Total Data : 10000
    Total Variables : 14
    Missing Values : 0
    
    Type of Variables in Data:
    
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10000 entries, 0 to 9999
    Data columns (total 14 columns):
    RowNumber          10000 non-null int64
    CustomerId         10000 non-null int64
    Surname            10000 non-null object
    CreditScore        10000 non-null int64
    Geography          10000 non-null object
    Gender             10000 non-null object
    Age                10000 non-null int64
    Tenure             10000 non-null int64
    Balance            10000 non-null float64
    NumOfProducts      10000 non-null int64
    HasCrCard          10000 non-null int64
    IsActiveMember     10000 non-null int64
    EstimatedSalary    10000 non-null float64
    Exited             10000 non-null int64
    dtypes: float64(2), int64(9), object(3)
    memory usage: 1.1+ MB
    None
    


```python
# tail of df , 10 lines
df.tail(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RowNumber</th>
      <th>CustomerId</th>
      <th>Surname</th>
      <th>CreditScore</th>
      <th>Geography</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>EstimatedSalary</th>
      <th>Exited</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>9995</td>
      <td>9996</td>
      <td>15606229</td>
      <td>Obijiaku</td>
      <td>771</td>
      <td>France</td>
      <td>Male</td>
      <td>39</td>
      <td>5</td>
      <td>0.00</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>96270.64</td>
      <td>0</td>
    </tr>
    <tr>
      <td>9996</td>
      <td>9997</td>
      <td>15569892</td>
      <td>Johnstone</td>
      <td>516</td>
      <td>France</td>
      <td>Male</td>
      <td>35</td>
      <td>10</td>
      <td>57369.61</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>101699.77</td>
      <td>0</td>
    </tr>
    <tr>
      <td>9997</td>
      <td>9998</td>
      <td>15584532</td>
      <td>Liu</td>
      <td>709</td>
      <td>France</td>
      <td>Female</td>
      <td>36</td>
      <td>7</td>
      <td>0.00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>42085.58</td>
      <td>1</td>
    </tr>
    <tr>
      <td>9998</td>
      <td>9999</td>
      <td>15682355</td>
      <td>Sabbatini</td>
      <td>772</td>
      <td>Germany</td>
      <td>Male</td>
      <td>42</td>
      <td>3</td>
      <td>75075.31</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>92888.52</td>
      <td>1</td>
    </tr>
    <tr>
      <td>9999</td>
      <td>10000</td>
      <td>15628319</td>
      <td>Walker</td>
      <td>792</td>
      <td>France</td>
      <td>Female</td>
      <td>28</td>
      <td>4</td>
      <td>130142.79</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>38190.78</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# prints columns of df
for col in df.columns:
    print(col)
```

    RowNumber
    CustomerId
    Surname
    CreditScore
    Geography
    Gender
    Age
    Tenure
    Balance
    NumOfProducts
    HasCrCard
    IsActiveMember
    EstimatedSalary
    Exited
    


```python
# columns i wont use
df = df.drop(['CustomerId', 'Surname', 'RowNumber'], axis = 1)

```


```python
# statistic of df , distirbution of data & properties
df.describe(include ='all')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CreditScore</th>
      <th>Geography</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>EstimatedSalary</th>
      <th>Exited</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>10000.000000</td>
      <td>10000</td>
      <td>10000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.00000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
    </tr>
    <tr>
      <td>unique</td>
      <td>NaN</td>
      <td>3</td>
      <td>2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>top</td>
      <td>NaN</td>
      <td>France</td>
      <td>Male</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>freq</td>
      <td>NaN</td>
      <td>5014</td>
      <td>5457</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>650.528800</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>38.921800</td>
      <td>5.012800</td>
      <td>76485.889288</td>
      <td>1.530200</td>
      <td>0.70550</td>
      <td>0.515100</td>
      <td>100090.239881</td>
      <td>0.203700</td>
    </tr>
    <tr>
      <td>std</td>
      <td>96.653299</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10.487806</td>
      <td>2.892174</td>
      <td>62397.405202</td>
      <td>0.581654</td>
      <td>0.45584</td>
      <td>0.499797</td>
      <td>57510.492818</td>
      <td>0.402769</td>
    </tr>
    <tr>
      <td>min</td>
      <td>350.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>18.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>11.580000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>584.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>32.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>51002.110000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>652.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>37.000000</td>
      <td>5.000000</td>
      <td>97198.540000</td>
      <td>1.000000</td>
      <td>1.00000</td>
      <td>1.000000</td>
      <td>100193.915000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>718.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>44.000000</td>
      <td>7.000000</td>
      <td>127644.240000</td>
      <td>2.000000</td>
      <td>1.00000</td>
      <td>1.000000</td>
      <td>149388.247500</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <td>max</td>
      <td>850.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>92.000000</td>
      <td>10.000000</td>
      <td>250898.090000</td>
      <td>4.000000</td>
      <td>1.00000</td>
      <td>1.000000</td>
      <td>199992.480000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# missing median 
print(df.median())
```

    CreditScore           652.000
    Age                    37.000
    Tenure                  5.000
    Balance             97198.540
    NumOfProducts           1.000
    HasCrCard               1.000
    IsActiveMember          1.000
    EstimatedSalary    100193.915
    Exited                  0.000
    dtype: float64
    

### histograms 


```python
# Gender histogram

plt.hist(x = df.Gender , bins = 10)
plt.title('Gender')
plt.xlabel('Gender')
plt.ylabel('Frequency')
plt.show()
```


    
![png](output_10_0.png)
    



```python
#age in the dataset
plt.hist(x = df.Age, bins = 15)
plt.title('Histogram of Age')
plt.ylabel('Frequency')
plt.xlabel('Age')
plt.grid()
plt.show()
```


    
![png](output_11_0.png)
    



```python
# credit 
plt.hist(x = df.CreditScore, bins = 15)
plt.title('Credit Score')
plt.xlabel('Credit Score')
plt.ylabel('Frequency')
plt.grid()
plt.show()
```


    
![png](output_12_0.png)
    



```python
# Tenure
plt.hist(x = df.Tenure, bins =20)
plt.title('Tenure')
plt.xlabel('Tenure')
plt.ylabel('Frequency')
plt.show()
```


    
![png](output_13_0.png)
    



```python
# Balance
plt.hist(x = df.Balance, bins =20)
plt.title('Balance')
plt.xlabel('Balance')
plt.ylabel('Frequency')
plt.grid()
plt.show()
```


    
![png](output_14_0.png)
    


### CDF


```python
# create on CDF - going to create a cdf of credit score..
cdf = thinkstats2.Cdf(df.CreditScore, label = 'Credit Score')
thinkplot.Cdf(cdf)
thinkplot.Show(xlabel= 'Credit Score', ylabel ='CDF')
plt.show()
```


    
![png](output_16_0.png)
    



    <Figure size 576x432 with 0 Axes>


### The CDF plot show that the median credit score comes at 650

# pmf


```python
# here im creating a PMF of gender and CreditScore.. i want to find out how CreditScore runs diff depending on gender

male = df[df.Gender == 'Male']
female = df[df.Gender == 'Female']
fig, ax = plt.subplots()
ax.hist([female.CreditScore, male.CreditScore], 10, (350,850),
        histtype='bar', label=('Female','Male'))
ax.set_title('Customer\'s Gender and Credit Score')
ax.legend()
plt.show()
```


    
![png](output_19_0.png)
    


### in this PMF i see that males seems to have a higher score than women. 


```python
# here im creating a PMF of gender and tenure.. i want to find out how tenure runs diff depending on gender

male = df[df.Gender == 'Male']
female = df[df.Gender == 'Female']
fig, ax = plt.subplots()
ax.hist([female.Tenure, male.Tenure], 10, (0,10),
        histtype='bar', label=('Female','Male'))
ax.set_title('Customer\'s Gender and Tenure')
ax.legend()
plt.show()
```


    
![png](output_21_0.png)
    


### Again males have a greater tenure than women.. except at the beggining ofcourse 

# Analytic Distribution


```python
mu, var = thinkstats2.TrimmedMeanVar(df.CreditScore, p=0.002)
print('Mean:{}\n Var: {}'.format(mu, var,sigma))
```

    Mean:650.693172690763
     Var: 9139.008467444071
    


```python
sigma = np.sqrt(var)
print('Sigma:',sigma)
```

    Sigma: 95.59816142292733
    


```python
xs,ps = thinkstats2.RenderNormalCdf(mu,sigma,low =350, high = 850)

thinkplot.Plot(xs,ps, label = 'model', color = 'black')
cdf2 = thinkstats2.Cdf(df.CreditScore,label='data')

thinkplot.PrePlot(1)
thinkplot.Cdf(cdf2)
thinkplot.Config(title='Credit Score',
                xlabel = 'Score',
                ylabel = 'CDF')
```


    
![png](output_26_0.png)
    


#### seems that the credit score is a good fit... too well.... , although the actual data might stretch slightly.. 


```python
# cdf of credit score 
diffs = df.CreditScore.diff()
cdf1 = thinkstats2.Cdf(diffs, label ='actual')
thinkplot.Cdf(cdf1)
thinkplot.Show(xlabel='Credit Score', ylabel ='CDF')
```


    
![png](output_28_0.png)
    



    <Figure size 576x432 with 0 Axes>



```python
# Ccdf on log scale 
thinkplot.Cdf(cdf, complement= True)
thinkplot.Show(xlabel = 'Credit Score', ylabel='CCDF', yscale='log')
```


    
![png](output_29_0.png)
    



    <Figure size 576x432 with 0 Axes>


####  yikes, tried this 2 different ways hard to read this one 

# 2 scatter plots 


```python
# scatterplot of creditscore vs age 
credit = df['CreditScore']
age = df['Age']
thinkplot.figure(figsize=(12,8))
thinkplot.Scatter(credit, age, alpha= 1.0, color='white',s= 20, edgecolors='blue')
thinkplot.Show(xlabel='Credit Score',
                     ylabel='Age',
                     xlim=[400, 850],
                     ylim=[17, 90],
                     legend=False ,
                     title="Scatter plot of Credit Score vs Age")
```


    
![png](output_32_0.png)
    



    <Figure size 576x432 with 0 Axes>



```python
# finding out correlation between credit ang age wit pearson
pearson = thinkstats2.Corr(credit,age)
print('Pearson\'s Correlation:{}'.format(pearson))
```

    Pearson's Correlation:-0.00396490552539007
    


```python
# correlation between credit and age with spearman
spearman = thinkstats2.SpearmanCorr(credit,age)
print('Spearman\'s Correlation:{}'.format(spearman))
```

    Spearman's Correlation:-0.007974044311824222
    


```python
# scatterplot of creditscore vs salary
#credit = df['CreditScore']
salary = df['EstimatedSalary']
thinkplot.figure(figsize=(12,8))
thinkplot.Scatter(credit, salary, alpha= 1.0, color='white',s= 20, edgecolors='blue')
thinkplot.Show(xlabel='Credit Score',
                     ylabel='Salary',
                     xlim=[400, 850],
                     ylim=[10, 200000],
                     legend=False ,
                     title="Scatter plot of Credit Score vs Salary")
```


    
![png](output_35_0.png)
    



    <Figure size 576x432 with 0 Axes>



```python
# finding out correlation between credit ang salary wit pearson
pearson2 = thinkstats2.Corr(credit,salary)
print('Pearson\'s Correlation:{}'.format(pearson2))
```

    Pearson's Correlation:-0.0013842928679845183
    


```python
# correlation between credit and age with spearman
spearman2 = thinkstats2.SpearmanCorr(credit,salary)
print('Spearman\'s Correlation:{}'.format(spearman2))
```

    Spearman's Correlation:0.0012365243785171628
    


```python
# scatterplot of balance vs credit
Balance = df['Balance']
thinkplot.figure(figsize=(20,10))
thinkplot.Scatter(credit, Balance, alpha= 1.0, color='white',s= 20, edgecolors='blue')
thinkplot.Show(xlabel='Credit Score',
                     ylabel='Balance',
                     xlim=[400, 850],
                     ylim=[0, 250898],
                     legend=False ,
                     title="Scatter plot of Credit Score vs Balance")
```


    
![png](output_38_0.png)
    



    <Figure size 576x432 with 0 Axes>



```python
# finding out correlation between credit ang balance with pearson
pearson3 = thinkstats2.Corr(credit,Balance)
print('Pearson\'s Correlation:{}'.format(pearson3))
```

    Pearson's Correlation:0.0062683816160087205
    


```python
# correlation between credit and balance with spearman
spearman3 = thinkstats2.SpearmanCorr(credit,Balance)
print('Spearman\'s Correlation:{}'.format(spearman3))
```

    Spearman's Correlation:0.005686570567648804
    


```python
# covariance 
# function grabbed from textbook
def Cov(xs, ys, meanx=None, meany=None):
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    if meanx is None:
        meanx = np.mean(xs)
    if meany is None:
        meany = np.mean(ys)

    cov = np.dot(xs-meanx, ys-meany) / len(xs)
    return cov
```


```python
# Covariance between creditscore and Customer Exit status
Cov(df.CreditScore, df.Exited)
```




    -1.0546165599999997




```python
# Covariance between creditscore and Customer Exit status
Cov(df.CreditScore, df.Balance)
```




    37800.29659050566



###  now, covariance shoes us if those two variables are positive or negative but it does not show us the strength. 


```python
# pearson 
correlation = df.corr()
#plt.figure(figsize=(10,10))
sns.heatmap(correlation,annot=True)
# fixes the borders found this code on google
b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # Subtract 0.5 from the top
plt.ylim(b, t) # update the ylim(bottom, top) values
plt.show() 

```


    
![png](output_45_0.png)
    



```python
# okay so now this correlation map is going to show me which variables are linear or non linear. 
#seems that credit score and age are sighlty correlated as well as active member.
```


```python
# Spearman
correlation = df.corr('spearman')
#plt.figure(figsize=(10,10))
sns.heatmap(correlation,annot=True)
# fixes the borders found this code on google
b, t = plt.ylim() # discover the values for bottom and top
b += 0.5 # Add 0.5 to the bottom
t -= 0.5 # Subtract 0.5 from the top
plt.ylim(b, t) # update the ylim(bottom, top) values
plt.show() 
```


    
![png](output_47_0.png)
    



```python
# linear Regression 
formula = 'Exited ~ CreditScore'
model = smf.ols(formula, data=df)
results = model.fit()

```


```python
results.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>         <td>Exited</td>      <th>  R-squared:         </th> <td>   0.001</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.001</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   7.345</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Sat, 30 May 2020</td> <th>  Prob (F-statistic):</th>  <td>0.00674</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>12:16:03</td>     <th>  Log-Likelihood:    </th> <td> -5091.3</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td> 10000</td>      <th>  AIC:               </th> <td>1.019e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  9998</td>      <th>  BIC:               </th> <td>1.020e+04</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
       <td></td>          <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>   <td>    0.2771</td> <td>    0.027</td> <td>   10.115</td> <td> 0.000</td> <td>    0.223</td> <td>    0.331</td>
</tr>
<tr>
  <th>CreditScore</th> <td>   -0.0001</td> <td> 4.17e-05</td> <td>   -2.710</td> <td> 0.007</td> <td>   -0.000</td> <td>-3.12e-05</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>2041.209</td> <th>  Durbin-Watson:     </th> <td>   1.993</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>3612.134</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 1.470</td>  <th>  Prob(JB):          </th> <td>    0.00</td>
</tr>
<tr>
  <th>Kurtosis:</th>       <td> 3.165</td>  <th>  Cond. No.          </th> <td>4.48e+03</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 4.48e+03. This might indicate that there are<br/>strong multicollinearity or other numerical problems.




```python
inter = results.params['Intercept']
slope = results.params['CreditScore']
inter,slope
```




    (0.27714651348262426, -0.00011290278536880139)


