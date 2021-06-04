```python
#Importing libraries

import pandas as pd
import numpy as np
```


```python
# Getting data

train= pd.read_csv("train.csv")
test= pd.read_csv("test.csv")
```


```python
## loading Pycaret with data

import pycaret
```


```python
#print first 5 lines of my data

train.head()
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
      <th>Id</th>
      <th>Open Date</th>
      <th>City</th>
      <th>City Group</th>
      <th>Type</th>
      <th>P1</th>
      <th>P2</th>
      <th>P3</th>
      <th>P4</th>
      <th>P5</th>
      <th>...</th>
      <th>P29</th>
      <th>P30</th>
      <th>P31</th>
      <th>P32</th>
      <th>P33</th>
      <th>P34</th>
      <th>P35</th>
      <th>P36</th>
      <th>P37</th>
      <th>revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>07/17/1999</td>
      <td>İstanbul</td>
      <td>Big Cities</td>
      <td>IL</td>
      <td>4</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>2</td>
      <td>...</td>
      <td>3.0</td>
      <td>5</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
      <td>5</td>
      <td>4</td>
      <td>3</td>
      <td>4</td>
      <td>5653753.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>02/14/2008</td>
      <td>Ankara</td>
      <td>Big Cities</td>
      <td>FC</td>
      <td>4</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>1</td>
      <td>...</td>
      <td>3.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6923131.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>03/09/2013</td>
      <td>Diyarbakır</td>
      <td>Other</td>
      <td>IL</td>
      <td>2</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>2</td>
      <td>...</td>
      <td>3.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2055379.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>02/02/2012</td>
      <td>Tokat</td>
      <td>Other</td>
      <td>IL</td>
      <td>6</td>
      <td>4.5</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>4</td>
      <td>...</td>
      <td>7.5</td>
      <td>25</td>
      <td>12</td>
      <td>10</td>
      <td>6</td>
      <td>18</td>
      <td>12</td>
      <td>12</td>
      <td>6</td>
      <td>2675511.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>05/09/2009</td>
      <td>Gaziantep</td>
      <td>Other</td>
      <td>IL</td>
      <td>3</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>2</td>
      <td>...</td>
      <td>3.0</td>
      <td>5</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
      <td>4316715.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 43 columns</p>
</div>




```python
test.head()
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
      <th>Id</th>
      <th>Open Date</th>
      <th>City</th>
      <th>City Group</th>
      <th>Type</th>
      <th>P1</th>
      <th>P2</th>
      <th>P3</th>
      <th>P4</th>
      <th>P5</th>
      <th>...</th>
      <th>P28</th>
      <th>P29</th>
      <th>P30</th>
      <th>P31</th>
      <th>P32</th>
      <th>P33</th>
      <th>P34</th>
      <th>P35</th>
      <th>P36</th>
      <th>P37</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>01/22/2011</td>
      <td>Niğde</td>
      <td>Other</td>
      <td>FC</td>
      <td>1</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>1</td>
      <td>...</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>03/18/2011</td>
      <td>Konya</td>
      <td>Other</td>
      <td>IL</td>
      <td>3</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>2</td>
      <td>...</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>10/30/2013</td>
      <td>Ankara</td>
      <td>Big Cities</td>
      <td>FC</td>
      <td>3</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>2</td>
      <td>...</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>05/06/2013</td>
      <td>Kocaeli</td>
      <td>Other</td>
      <td>IL</td>
      <td>2</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>2</td>
      <td>...</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>07/31/2013</td>
      <td>Afyonkarahisar</td>
      <td>Other</td>
      <td>FC</td>
      <td>2</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>1</td>
      <td>...</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 42 columns</p>
</div>



# Data Dictionary: 

#####  Id : Restaurant id. 

#####  Open Date : opening date for a restaurant
#####  City : City that the restaurant is in. Note that there are unicode in the names. 
#####  City Group: Type of the city. Big cities, or Other. 
#####  Type: Type of the restaurant. FC: Food Court, IL: Inline, DT: Drive Thru, MB: Mobile
#####  P1, P2 - P37: There are three categories of these obfuscated data. Demographic data are gathered from third party providers with GIS systems. These include population in any given area, age and gender distribution, development scales. Real estate data mainly relate to the m2 of the location, front facade of the location, car park availability. Commercial data mainly include the existence of points of interest including schools, banks, other QSR operators.
#####  Revenue: The revenue column indicates a (transformed) revenue of the restaurant in a given year and is the target of predictive analysis. Please note that the values are transformed so they don't mean real dollar values. 


```python
#data shapes 
train.shape
```




    (137, 43)




```python
test.shape
```




    (100000, 42)




```python
train.describe(include='all')
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
      <th>Id</th>
      <th>Open Date</th>
      <th>City</th>
      <th>City Group</th>
      <th>Type</th>
      <th>P1</th>
      <th>P2</th>
      <th>P3</th>
      <th>P4</th>
      <th>P5</th>
      <th>...</th>
      <th>P29</th>
      <th>P30</th>
      <th>P31</th>
      <th>P32</th>
      <th>P33</th>
      <th>P34</th>
      <th>P35</th>
      <th>P36</th>
      <th>P37</th>
      <th>revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>137.000000</td>
      <td>137</td>
      <td>137</td>
      <td>137</td>
      <td>137</td>
      <td>137.000000</td>
      <td>137.000000</td>
      <td>137.000000</td>
      <td>137.000000</td>
      <td>137.000000</td>
      <td>...</td>
      <td>137.000000</td>
      <td>137.000000</td>
      <td>137.000000</td>
      <td>137.000000</td>
      <td>137.000000</td>
      <td>137.000000</td>
      <td>137.000000</td>
      <td>137.000000</td>
      <td>137.000000</td>
      <td>1.370000e+02</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>NaN</td>
      <td>134</td>
      <td>34</td>
      <td>2</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
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
      <th>top</th>
      <td>NaN</td>
      <td>02/23/2010</td>
      <td>İstanbul</td>
      <td>Big Cities</td>
      <td>FC</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
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
      <th>freq</th>
      <td>NaN</td>
      <td>2</td>
      <td>50</td>
      <td>78</td>
      <td>76</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
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
      <th>mean</th>
      <td>68.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.014599</td>
      <td>4.408759</td>
      <td>4.317518</td>
      <td>4.372263</td>
      <td>2.007299</td>
      <td>...</td>
      <td>3.135036</td>
      <td>2.729927</td>
      <td>1.941606</td>
      <td>2.525547</td>
      <td>1.138686</td>
      <td>2.489051</td>
      <td>2.029197</td>
      <td>2.211679</td>
      <td>1.116788</td>
      <td>4.453533e+06</td>
    </tr>
    <tr>
      <th>std</th>
      <td>39.692569</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.910391</td>
      <td>1.514900</td>
      <td>1.032337</td>
      <td>1.016462</td>
      <td>1.209620</td>
      <td>...</td>
      <td>1.680887</td>
      <td>5.536647</td>
      <td>3.512093</td>
      <td>5.230117</td>
      <td>1.698540</td>
      <td>5.165093</td>
      <td>3.436272</td>
      <td>4.168211</td>
      <td>1.790768</td>
      <td>2.576072e+06</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.149870e+06</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>34.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>...</td>
      <td>2.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.999068e+06</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>68.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.000000</td>
      <td>5.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>2.000000</td>
      <td>...</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.939804e+06</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>102.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>2.000000</td>
      <td>...</td>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>5.166635e+06</td>
    </tr>
    <tr>
      <th>max</th>
      <td>136.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>12.000000</td>
      <td>7.500000</td>
      <td>7.500000</td>
      <td>7.500000</td>
      <td>8.000000</td>
      <td>...</td>
      <td>7.500000</td>
      <td>25.000000</td>
      <td>15.000000</td>
      <td>25.000000</td>
      <td>6.000000</td>
      <td>24.000000</td>
      <td>15.000000</td>
      <td>20.000000</td>
      <td>8.000000</td>
      <td>1.969694e+07</td>
    </tr>
  </tbody>
</table>
<p>11 rows × 43 columns</p>
</div>




```python
train.isnull().sum()
```




    Id            0
    Open Date     0
    City          0
    City Group    0
    Type          0
    P1            0
    P2            0
    P3            0
    P4            0
    P5            0
    P6            0
    P7            0
    P8            0
    P9            0
    P10           0
    P11           0
    P12           0
    P13           0
    P14           0
    P15           0
    P16           0
    P17           0
    P18           0
    P19           0
    P20           0
    P21           0
    P22           0
    P23           0
    P24           0
    P25           0
    P26           0
    P27           0
    P28           0
    P29           0
    P30           0
    P31           0
    P32           0
    P33           0
    P34           0
    P35           0
    P36           0
    P37           0
    revenue       0
    dtype: int64




```python
test.isnull().sum()
```




    Id            0
    Open Date     0
    City          0
    City Group    0
    Type          0
    P1            0
    P2            0
    P3            0
    P4            0
    P5            0
    P6            0
    P7            0
    P8            0
    P9            0
    P10           0
    P11           0
    P12           0
    P13           0
    P14           0
    P15           0
    P16           0
    P17           0
    P18           0
    P19           0
    P20           0
    P21           0
    P22           0
    P23           0
    P24           0
    P25           0
    P26           0
    P27           0
    P28           0
    P29           0
    P30           0
    P31           0
    P32           0
    P33           0
    P34           0
    P35           0
    P36           0
    P37           0
    dtype: int64




```python
#data type conversion 
train = train.astype({"City":'category',"City Group":'category',"Type":'category', "Open Date":'datetime64[ns]'})
test = test.astype({"City":'category',"City Group":'category',"Type":'category', "Open Date":'datetime64[ns]'})
```


```python
# 1. Check categories in train and test data
# df_train : City, City group, Type
cities = train.City.unique().tolist()
print(len(cities))
city_group = train['City Group'].unique().tolist()
print(city_group)
type_ = train['Type'].unique().tolist()
print(type_)
```

    34
    ['Big Cities', 'Other']
    ['IL', 'FC', 'DT']
    


```python
# 1. Check categories in train and test data
# test : City, City group, Type
# Note that test does not have city in it. 

city_group = test['City Group'].unique().tolist()
print(city_group)
type_ = test['Type'].unique().tolist()
print(type_)
```

    ['Other', 'Big Cities']
    ['FC', 'IL', 'DT', 'MB']
    


```python
cities = test.City.unique().tolist()
print(len(cities))
city_group = train['City Group'].unique().tolist()
print(city_group)
type_ = train['Type'].unique().tolist()
print(type_)
```

    57
    ['Big Cities', 'Other']
    ['IL', 'FC', 'DT']
    


```python
train.head()
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
      <th>Id</th>
      <th>Open Date</th>
      <th>City</th>
      <th>City Group</th>
      <th>Type</th>
      <th>P1</th>
      <th>P2</th>
      <th>P3</th>
      <th>P4</th>
      <th>P5</th>
      <th>...</th>
      <th>P29</th>
      <th>P30</th>
      <th>P31</th>
      <th>P32</th>
      <th>P33</th>
      <th>P34</th>
      <th>P35</th>
      <th>P36</th>
      <th>P37</th>
      <th>revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1999-07-17</td>
      <td>İstanbul</td>
      <td>Big Cities</td>
      <td>IL</td>
      <td>4</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>2</td>
      <td>...</td>
      <td>3.0</td>
      <td>5</td>
      <td>3</td>
      <td>4</td>
      <td>5</td>
      <td>5</td>
      <td>4</td>
      <td>3</td>
      <td>4</td>
      <td>5653753.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2008-02-14</td>
      <td>Ankara</td>
      <td>Big Cities</td>
      <td>FC</td>
      <td>4</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>1</td>
      <td>...</td>
      <td>3.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>6923131.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2013-03-09</td>
      <td>Diyarbakır</td>
      <td>Other</td>
      <td>IL</td>
      <td>2</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>2</td>
      <td>...</td>
      <td>3.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2055379.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>2012-02-02</td>
      <td>Tokat</td>
      <td>Other</td>
      <td>IL</td>
      <td>6</td>
      <td>4.5</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>4</td>
      <td>...</td>
      <td>7.5</td>
      <td>25</td>
      <td>12</td>
      <td>10</td>
      <td>6</td>
      <td>18</td>
      <td>12</td>
      <td>12</td>
      <td>6</td>
      <td>2675511.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>2009-05-09</td>
      <td>Gaziantep</td>
      <td>Other</td>
      <td>IL</td>
      <td>3</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>2</td>
      <td>...</td>
      <td>3.0</td>
      <td>5</td>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>3</td>
      <td>4316715.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 43 columns</p>
</div>




```python
test.head()
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
      <th>Id</th>
      <th>Open Date</th>
      <th>City</th>
      <th>City Group</th>
      <th>Type</th>
      <th>P1</th>
      <th>P2</th>
      <th>P3</th>
      <th>P4</th>
      <th>P5</th>
      <th>...</th>
      <th>P28</th>
      <th>P29</th>
      <th>P30</th>
      <th>P31</th>
      <th>P32</th>
      <th>P33</th>
      <th>P34</th>
      <th>P35</th>
      <th>P36</th>
      <th>P37</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2011-01-22</td>
      <td>Niğde</td>
      <td>Other</td>
      <td>FC</td>
      <td>1</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>1</td>
      <td>...</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2011-03-18</td>
      <td>Konya</td>
      <td>Other</td>
      <td>IL</td>
      <td>3</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>2</td>
      <td>...</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2013-10-30</td>
      <td>Ankara</td>
      <td>Big Cities</td>
      <td>FC</td>
      <td>3</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>2</td>
      <td>...</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>2013-05-06</td>
      <td>Kocaeli</td>
      <td>Other</td>
      <td>IL</td>
      <td>2</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>2</td>
      <td>...</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>2013-07-31</td>
      <td>Afyonkarahisar</td>
      <td>Other</td>
      <td>FC</td>
      <td>2</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>1</td>
      <td>...</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 42 columns</p>
</div>




```python
train['City Group'] = train[['City Group']].apply(lambda x: x.cat.codes)
train['Type'] = train[['Type']].apply(lambda x: x.cat.codes)
```


```python
## After analysis, i'm dropping ID, Open Date, and City to make both data frames consistent for each. This is will allow for ML processing. 

train = train.drop(columns = ['Id','Open Date','City'],axis=1)
test = test.drop(columns = ['Id','Open Date','City'],axis=1)
```


```python
# pandas profile 
from pandas_profiling import ProfileReport
```


```python
#Generating report 
profile = ProfileReport(train, title="Pandas Profiling Report")
```


```python
#To view the report
profile.to_widgets()
```


    HBox(children=(HTML(value='Summarize dataset'), FloatProgress(value=0.0, max=53.0), HTML(value='')))


    
    


    HBox(children=(HTML(value='Generate report structure'), FloatProgress(value=0.0, max=1.0), HTML(value='')))


    
    


    HBox(children=(HTML(value='Render widgets'), FloatProgress(value=0.0, max=1.0), HTML(value='')))



    VBox(children=(Tab(children=(Tab(children=(GridBox(children=(VBox(children=(GridspecLayout(children=(HTML(valu…



```python
#import regression module 
from pycaret.regression import *
```


```python
#intialize the setup (in Notebook env)
exp_reg = setup(train, target = 'revenue')
```


<style  type="text/css" >
</style><table id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedf" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Description</th>        <th class="col_heading level0 col1" >Value</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedflevel0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow0_col0" class="data row0 col0" >session_id</td>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow0_col1" class="data row0 col1" >233</td>
            </tr>
            <tr>
                        <th id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedflevel0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow1_col0" class="data row1 col0" >Target</td>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow1_col1" class="data row1 col1" >revenue</td>
            </tr>
            <tr>
                        <th id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedflevel0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow2_col0" class="data row2 col0" >Original Data</td>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow2_col1" class="data row2 col1" >(137, 40)</td>
            </tr>
            <tr>
                        <th id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedflevel0_row3" class="row_heading level0 row3" >3</th>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow3_col0" class="data row3 col0" >Missing Values</td>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow3_col1" class="data row3 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedflevel0_row4" class="row_heading level0 row4" >4</th>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow4_col0" class="data row4 col0" >Numeric Features</td>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow4_col1" class="data row4 col1" >10</td>
            </tr>
            <tr>
                        <th id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedflevel0_row5" class="row_heading level0 row5" >5</th>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow5_col0" class="data row5 col0" >Categorical Features</td>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow5_col1" class="data row5 col1" >29</td>
            </tr>
            <tr>
                        <th id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedflevel0_row6" class="row_heading level0 row6" >6</th>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow6_col0" class="data row6 col0" >Ordinal Features</td>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow6_col1" class="data row6 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedflevel0_row7" class="row_heading level0 row7" >7</th>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow7_col0" class="data row7 col0" >High Cardinality Features</td>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow7_col1" class="data row7 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedflevel0_row8" class="row_heading level0 row8" >8</th>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow8_col0" class="data row8 col0" >High Cardinality Method</td>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow8_col1" class="data row8 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedflevel0_row9" class="row_heading level0 row9" >9</th>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow9_col0" class="data row9 col0" >Transformed Train Set</td>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow9_col1" class="data row9 col1" >(95, 184)</td>
            </tr>
            <tr>
                        <th id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedflevel0_row10" class="row_heading level0 row10" >10</th>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow10_col0" class="data row10 col0" >Transformed Test Set</td>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow10_col1" class="data row10 col1" >(42, 184)</td>
            </tr>
            <tr>
                        <th id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedflevel0_row11" class="row_heading level0 row11" >11</th>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow11_col0" class="data row11 col0" >Shuffle Train-Test</td>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow11_col1" class="data row11 col1" >True</td>
            </tr>
            <tr>
                        <th id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedflevel0_row12" class="row_heading level0 row12" >12</th>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow12_col0" class="data row12 col0" >Stratify Train-Test</td>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow12_col1" class="data row12 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedflevel0_row13" class="row_heading level0 row13" >13</th>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow13_col0" class="data row13 col0" >Fold Generator</td>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow13_col1" class="data row13 col1" >KFold</td>
            </tr>
            <tr>
                        <th id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedflevel0_row14" class="row_heading level0 row14" >14</th>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow14_col0" class="data row14 col0" >Fold Number</td>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow14_col1" class="data row14 col1" >10</td>
            </tr>
            <tr>
                        <th id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedflevel0_row15" class="row_heading level0 row15" >15</th>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow15_col0" class="data row15 col0" >CPU Jobs</td>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow15_col1" class="data row15 col1" >-1</td>
            </tr>
            <tr>
                        <th id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedflevel0_row16" class="row_heading level0 row16" >16</th>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow16_col0" class="data row16 col0" >Use GPU</td>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow16_col1" class="data row16 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedflevel0_row17" class="row_heading level0 row17" >17</th>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow17_col0" class="data row17 col0" >Log Experiment</td>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow17_col1" class="data row17 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedflevel0_row18" class="row_heading level0 row18" >18</th>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow18_col0" class="data row18 col0" >Experiment Name</td>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow18_col1" class="data row18 col1" >reg-default-name</td>
            </tr>
            <tr>
                        <th id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedflevel0_row19" class="row_heading level0 row19" >19</th>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow19_col0" class="data row19 col0" >USI</td>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow19_col1" class="data row19 col1" >393c</td>
            </tr>
            <tr>
                        <th id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedflevel0_row20" class="row_heading level0 row20" >20</th>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow20_col0" class="data row20 col0" >Imputation Type</td>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow20_col1" class="data row20 col1" >simple</td>
            </tr>
            <tr>
                        <th id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedflevel0_row21" class="row_heading level0 row21" >21</th>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow21_col0" class="data row21 col0" >Iterative Imputation Iteration</td>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow21_col1" class="data row21 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedflevel0_row22" class="row_heading level0 row22" >22</th>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow22_col0" class="data row22 col0" >Numeric Imputer</td>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow22_col1" class="data row22 col1" >mean</td>
            </tr>
            <tr>
                        <th id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedflevel0_row23" class="row_heading level0 row23" >23</th>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow23_col0" class="data row23 col0" >Iterative Imputation Numeric Model</td>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow23_col1" class="data row23 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedflevel0_row24" class="row_heading level0 row24" >24</th>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow24_col0" class="data row24 col0" >Categorical Imputer</td>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow24_col1" class="data row24 col1" >constant</td>
            </tr>
            <tr>
                        <th id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedflevel0_row25" class="row_heading level0 row25" >25</th>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow25_col0" class="data row25 col0" >Iterative Imputation Categorical Model</td>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow25_col1" class="data row25 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedflevel0_row26" class="row_heading level0 row26" >26</th>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow26_col0" class="data row26 col0" >Unknown Categoricals Handling</td>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow26_col1" class="data row26 col1" >least_frequent</td>
            </tr>
            <tr>
                        <th id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedflevel0_row27" class="row_heading level0 row27" >27</th>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow27_col0" class="data row27 col0" >Normalize</td>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow27_col1" class="data row27 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedflevel0_row28" class="row_heading level0 row28" >28</th>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow28_col0" class="data row28 col0" >Normalize Method</td>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow28_col1" class="data row28 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedflevel0_row29" class="row_heading level0 row29" >29</th>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow29_col0" class="data row29 col0" >Transformation</td>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow29_col1" class="data row29 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedflevel0_row30" class="row_heading level0 row30" >30</th>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow30_col0" class="data row30 col0" >Transformation Method</td>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow30_col1" class="data row30 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedflevel0_row31" class="row_heading level0 row31" >31</th>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow31_col0" class="data row31 col0" >PCA</td>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow31_col1" class="data row31 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedflevel0_row32" class="row_heading level0 row32" >32</th>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow32_col0" class="data row32 col0" >PCA Method</td>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow32_col1" class="data row32 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedflevel0_row33" class="row_heading level0 row33" >33</th>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow33_col0" class="data row33 col0" >PCA Components</td>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow33_col1" class="data row33 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedflevel0_row34" class="row_heading level0 row34" >34</th>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow34_col0" class="data row34 col0" >Ignore Low Variance</td>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow34_col1" class="data row34 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedflevel0_row35" class="row_heading level0 row35" >35</th>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow35_col0" class="data row35 col0" >Combine Rare Levels</td>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow35_col1" class="data row35 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedflevel0_row36" class="row_heading level0 row36" >36</th>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow36_col0" class="data row36 col0" >Rare Level Threshold</td>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow36_col1" class="data row36 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedflevel0_row37" class="row_heading level0 row37" >37</th>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow37_col0" class="data row37 col0" >Numeric Binning</td>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow37_col1" class="data row37 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedflevel0_row38" class="row_heading level0 row38" >38</th>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow38_col0" class="data row38 col0" >Remove Outliers</td>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow38_col1" class="data row38 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedflevel0_row39" class="row_heading level0 row39" >39</th>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow39_col0" class="data row39 col0" >Outliers Threshold</td>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow39_col1" class="data row39 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedflevel0_row40" class="row_heading level0 row40" >40</th>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow40_col0" class="data row40 col0" >Remove Multicollinearity</td>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow40_col1" class="data row40 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedflevel0_row41" class="row_heading level0 row41" >41</th>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow41_col0" class="data row41 col0" >Multicollinearity Threshold</td>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow41_col1" class="data row41 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedflevel0_row42" class="row_heading level0 row42" >42</th>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow42_col0" class="data row42 col0" >Clustering</td>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow42_col1" class="data row42 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedflevel0_row43" class="row_heading level0 row43" >43</th>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow43_col0" class="data row43 col0" >Clustering Iteration</td>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow43_col1" class="data row43 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedflevel0_row44" class="row_heading level0 row44" >44</th>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow44_col0" class="data row44 col0" >Polynomial Features</td>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow44_col1" class="data row44 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedflevel0_row45" class="row_heading level0 row45" >45</th>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow45_col0" class="data row45 col0" >Polynomial Degree</td>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow45_col1" class="data row45 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedflevel0_row46" class="row_heading level0 row46" >46</th>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow46_col0" class="data row46 col0" >Trignometry Features</td>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow46_col1" class="data row46 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedflevel0_row47" class="row_heading level0 row47" >47</th>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow47_col0" class="data row47 col0" >Polynomial Threshold</td>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow47_col1" class="data row47 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedflevel0_row48" class="row_heading level0 row48" >48</th>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow48_col0" class="data row48 col0" >Group Features</td>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow48_col1" class="data row48 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedflevel0_row49" class="row_heading level0 row49" >49</th>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow49_col0" class="data row49 col0" >Feature Selection</td>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow49_col1" class="data row49 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedflevel0_row50" class="row_heading level0 row50" >50</th>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow50_col0" class="data row50 col0" >Feature Selection Method</td>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow50_col1" class="data row50 col1" >classic</td>
            </tr>
            <tr>
                        <th id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedflevel0_row51" class="row_heading level0 row51" >51</th>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow51_col0" class="data row51 col0" >Features Selection Threshold</td>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow51_col1" class="data row51 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedflevel0_row52" class="row_heading level0 row52" >52</th>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow52_col0" class="data row52 col0" >Feature Interaction</td>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow52_col1" class="data row52 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedflevel0_row53" class="row_heading level0 row53" >53</th>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow53_col0" class="data row53 col0" >Feature Ratio</td>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow53_col1" class="data row53 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedflevel0_row54" class="row_heading level0 row54" >54</th>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow54_col0" class="data row54 col0" >Interaction Threshold</td>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow54_col1" class="data row54 col1" >None</td>
            </tr>
            <tr>
                        <th id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedflevel0_row55" class="row_heading level0 row55" >55</th>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow55_col0" class="data row55 col0" >Transform Target</td>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow55_col1" class="data row55 col1" >False</td>
            </tr>
            <tr>
                        <th id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedflevel0_row56" class="row_heading level0 row56" >56</th>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow56_col0" class="data row56 col0" >Transform Target Method</td>
                        <td id="T_ae36e350_c48d_11eb_af52_3c9c0f5cfedfrow56_col1" class="data row56 col1" >box-cox</td>
            </tr>
    </tbody></table>



```python
# return best model
best = compare_models()
```


<style  type="text/css" >
    #T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedf th {
          text-align: left;
    }#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow0_col0,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow0_col2,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow1_col0,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow1_col1,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow1_col3,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow1_col4,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow1_col5,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow1_col6,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow2_col0,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow2_col1,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow2_col2,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow2_col3,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow2_col4,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow2_col5,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow2_col6,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow3_col0,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow3_col1,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow3_col2,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow3_col3,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow3_col4,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow3_col5,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow3_col6,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow4_col0,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow4_col1,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow4_col2,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow4_col3,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow4_col4,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow4_col5,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow4_col6,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow5_col0,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow5_col1,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow5_col2,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow5_col3,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow5_col4,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow5_col5,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow5_col6,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow6_col0,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow6_col1,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow6_col2,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow6_col3,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow6_col4,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow6_col5,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow6_col6,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow7_col0,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow7_col1,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow7_col2,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow7_col3,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow7_col4,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow7_col5,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow7_col6,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow8_col0,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow8_col1,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow8_col2,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow8_col3,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow8_col4,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow8_col5,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow8_col6,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow9_col0,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow9_col1,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow9_col2,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow9_col3,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow9_col4,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow9_col5,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow9_col6,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow10_col0,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow10_col1,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow10_col2,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow10_col3,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow10_col4,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow10_col5,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow10_col6,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow11_col0,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow11_col1,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow11_col2,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow11_col3,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow11_col4,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow11_col5,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow11_col6,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow12_col0,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow12_col1,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow12_col2,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow12_col3,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow12_col4,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow12_col5,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow12_col6,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow13_col0,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow13_col1,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow13_col2,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow13_col3,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow13_col4,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow13_col5,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow13_col6,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow14_col0,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow14_col1,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow14_col2,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow14_col3,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow14_col4,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow14_col5,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow14_col6,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow15_col0,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow15_col1,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow15_col2,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow15_col3,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow15_col4,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow15_col5,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow15_col6{
            text-align:  left;
            text-align:  left;
        }#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow0_col1,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow0_col3,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow0_col4,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow0_col5,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow0_col6,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow1_col2{
            text-align:  left;
            text-align:  left;
            background-color:  yellow;
        }#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow0_col7,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow1_col7,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow2_col7,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow3_col7,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow4_col7,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow5_col7,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow6_col7,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow7_col7,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow8_col7,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow11_col7,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow12_col7,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow13_col7,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow14_col7,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow15_col7{
            text-align:  left;
            text-align:  left;
            background-color:  lightgrey;
        }#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow9_col7,#T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow10_col7{
            text-align:  left;
            text-align:  left;
            background-color:  yellow;
            background-color:  lightgrey;
        }</style><table id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedf" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Model</th>        <th class="col_heading level0 col1" >MAE</th>        <th class="col_heading level0 col2" >MSE</th>        <th class="col_heading level0 col3" >RMSE</th>        <th class="col_heading level0 col4" >R2</th>        <th class="col_heading level0 col5" >RMSLE</th>        <th class="col_heading level0 col6" >MAPE</th>        <th class="col_heading level0 col7" >TT (Sec)</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedflevel0_row0" class="row_heading level0 row0" >huber</th>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow0_col0" class="data row0 col0" >Huber Regressor</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow0_col1" class="data row0 col1" >1514471.0286</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow0_col2" class="data row0 col2" >6907184517554.9922</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow0_col3" class="data row0 col3" >2205471.0161</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow0_col4" class="data row0 col4" >-0.1225</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow0_col5" class="data row0 col5" >0.4347</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow0_col6" class="data row0 col6" >0.3553</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow0_col7" class="data row0 col7" >0.0140</td>
            </tr>
            <tr>
                        <th id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedflevel0_row1" class="row_heading level0 row1" >en</th>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow1_col0" class="data row1 col0" >Elastic Net</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow1_col1" class="data row1 col1" >1638893.9812</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow1_col2" class="data row1 col2" >6881556653670.4004</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow1_col3" class="data row1 col3" >2299780.4250</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow1_col4" class="data row1 col4" >-0.3554</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow1_col5" class="data row1 col5" >0.4648</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow1_col6" class="data row1 col6" >0.4303</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow1_col7" class="data row1 col7" >0.0150</td>
            </tr>
            <tr>
                        <th id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedflevel0_row2" class="row_heading level0 row2" >br</th>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow2_col0" class="data row2 col0" >Bayesian Ridge</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow2_col1" class="data row2 col1" >1681550.5954</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow2_col2" class="data row2 col2" >7237293365214.4990</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow2_col3" class="data row2 col3" >2350934.6729</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow2_col4" class="data row2 col4" >-0.4396</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow2_col5" class="data row2 col5" >0.4771</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow2_col6" class="data row2 col6" >0.4472</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow2_col7" class="data row2 col7" >0.0130</td>
            </tr>
            <tr>
                        <th id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedflevel0_row3" class="row_heading level0 row3" >par</th>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow3_col0" class="data row3 col0" >Passive Aggressive Regressor</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow3_col1" class="data row3 col1" >1671814.6831</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow3_col2" class="data row3 col2" >8084756685445.5811</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow3_col3" class="data row3 col3" >2462869.9719</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow3_col4" class="data row3 col4" >-0.5431</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow3_col5" class="data row3 col5" >0.4766</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow3_col6" class="data row3 col6" >0.3755</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow3_col7" class="data row3 col7" >0.0280</td>
            </tr>
            <tr>
                        <th id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedflevel0_row4" class="row_heading level0 row4" >knn</th>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow4_col0" class="data row4 col0" >K Neighbors Regressor</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow4_col1" class="data row4 col1" >1692318.9125</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow4_col2" class="data row4 col2" >7533950192844.7998</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow4_col3" class="data row4 col3" >2416942.7250</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow4_col4" class="data row4 col4" >-0.5886</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow4_col5" class="data row4 col5" >0.4745</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow4_col6" class="data row4 col6" >0.4341</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow4_col7" class="data row4 col7" >0.0130</td>
            </tr>
            <tr>
                        <th id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedflevel0_row5" class="row_heading level0 row5" >lightgbm</th>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow5_col0" class="data row5 col0" >Light Gradient Boosting Machine</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow5_col1" class="data row5 col1" >1857755.7842</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow5_col2" class="data row5 col2" >7994818965480.7715</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow5_col3" class="data row5 col3" >2540181.6501</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow5_col4" class="data row5 col4" >-0.7679</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow5_col5" class="data row5 col5" >0.5233</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow5_col6" class="data row5 col6" >0.4991</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow5_col7" class="data row5 col7" >0.7830</td>
            </tr>
            <tr>
                        <th id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedflevel0_row6" class="row_heading level0 row6" >rf</th>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow6_col0" class="data row6 col0" >Random Forest Regressor</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow6_col1" class="data row6 col1" >1725662.2307</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow6_col2" class="data row6 col2" >7930431536666.5596</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow6_col3" class="data row6 col3" >2486480.6553</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow6_col4" class="data row6 col4" >-0.9626</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow6_col5" class="data row6 col5" >0.4726</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow6_col6" class="data row6 col6" >0.4556</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow6_col7" class="data row6 col7" >0.1240</td>
            </tr>
            <tr>
                        <th id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedflevel0_row7" class="row_heading level0 row7" >ada</th>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow7_col0" class="data row7 col0" >AdaBoost Regressor</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow7_col1" class="data row7 col1" >1910704.7979</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow7_col2" class="data row7 col2" >9855852410799.5977</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow7_col3" class="data row7 col3" >2743949.9017</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow7_col4" class="data row7 col4" >-1.6465</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow7_col5" class="data row7 col5" >0.4969</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow7_col6" class="data row7 col6" >0.4849</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow7_col7" class="data row7 col7" >0.0210</td>
            </tr>
            <tr>
                        <th id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedflevel0_row8" class="row_heading level0 row8" >gbr</th>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow8_col0" class="data row8 col0" >Gradient Boosting Regressor</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow8_col1" class="data row8 col1" >1935623.4878</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow8_col2" class="data row8 col2" >9856676824226.8066</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow8_col3" class="data row8 col3" >2823393.0897</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow8_col4" class="data row8 col4" >-1.6775</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow8_col5" class="data row8 col5" >0.5349</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow8_col6" class="data row8 col6" >0.5070</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow8_col7" class="data row8 col7" >0.0280</td>
            </tr>
            <tr>
                        <th id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedflevel0_row9" class="row_heading level0 row9" >ridge</th>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow9_col0" class="data row9 col0" >Ridge Regression</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow9_col1" class="data row9 col1" >2281464.2500</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow9_col2" class="data row9 col2" >10979599056896.0000</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow9_col3" class="data row9 col3" >3141712.5750</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow9_col4" class="data row9 col4" >-2.2670</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow9_col5" class="data row9 col5" >0.7655</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow9_col6" class="data row9 col6" >0.5906</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow9_col7" class="data row9 col7" >0.0090</td>
            </tr>
            <tr>
                        <th id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedflevel0_row10" class="row_heading level0 row10" >omp</th>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow10_col0" class="data row10 col0" >Orthogonal Matching Pursuit</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow10_col1" class="data row10 col1" >2396859.4035</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow10_col2" class="data row10 col2" >12207356313660.7891</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow10_col3" class="data row10 col3" >3208785.9189</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow10_col4" class="data row10 col4" >-2.3409</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow10_col5" class="data row10 col5" >0.8032</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow10_col6" class="data row10 col6" >0.6237</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow10_col7" class="data row10 col7" >0.0090</td>
            </tr>
            <tr>
                        <th id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedflevel0_row11" class="row_heading level0 row11" >dt</th>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow11_col0" class="data row11 col0" >Decision Tree Regressor</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow11_col1" class="data row11 col1" >2384171.9056</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow11_col2" class="data row11 col2" >12464759844001.9023</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow11_col3" class="data row11 col3" >3289677.7640</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow11_col4" class="data row11 col4" >-3.1346</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow11_col5" class="data row11 col5" >0.6273</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow11_col6" class="data row11 col6" >0.6190</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow11_col7" class="data row11 col7" >0.0120</td>
            </tr>
            <tr>
                        <th id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedflevel0_row12" class="row_heading level0 row12" >et</th>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow12_col0" class="data row12 col0" >Extra Trees Regressor</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow12_col1" class="data row12 col1" >2277544.0862</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow12_col2" class="data row12 col2" >12626580305082.5977</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow12_col3" class="data row12 col3" >3292123.2448</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow12_col4" class="data row12 col4" >-3.2312</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow12_col5" class="data row12 col5" >0.5886</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow12_col6" class="data row12 col6" >0.5811</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow12_col7" class="data row12 col7" >0.1120</td>
            </tr>
            <tr>
                        <th id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedflevel0_row13" class="row_heading level0 row13" >llar</th>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow13_col0" class="data row13 col0" >Lasso Least Angle Regression</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow13_col1" class="data row13 col1" >3857219.4541</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow13_col2" class="data row13 col2" >46682722186753.5078</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow13_col3" class="data row13 col3" >5265587.2145</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow13_col4" class="data row13 col4" >-16.9142</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow13_col5" class="data row13 col5" >0.8610</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow13_col6" class="data row13 col6" >1.2844</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow13_col7" class="data row13 col7" >1.4440</td>
            </tr>
            <tr>
                        <th id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedflevel0_row14" class="row_heading level0 row14" >lasso</th>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow14_col0" class="data row14 col0" >Lasso Regression</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow14_col1" class="data row14 col1" >11439729.6500</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow14_col2" class="data row14 col2" >237262012520857.5938</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow14_col3" class="data row14 col3" >14304036.2500</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow14_col4" class="data row14 col4" >-109.0969</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow14_col5" class="data row14 col5" >1.4256</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow14_col6" class="data row14 col6" >3.1846</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow14_col7" class="data row14 col7" >0.0150</td>
            </tr>
            <tr>
                        <th id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedflevel0_row15" class="row_heading level0 row15" >lr</th>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow15_col0" class="data row15 col0" >Linear Regression</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow15_col1" class="data row15 col1" >474342958.3500</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow15_col2" class="data row15 col2" >11739908947909908480.0000</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow15_col3" class="data row15 col3" >1115915193.7000</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow15_col4" class="data row15 col4" >-450492.0103</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow15_col5" class="data row15 col5" >2.0239</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow15_col6" class="data row15 col6" >113.0511</td>
                        <td id="T_db69aae9_c48d_11eb_a6d9_3c9c0f5cfedfrow15_col7" class="data row15 col7" >0.9500</td>
            </tr>
    </tbody></table>



```python
# train linear regression model
huber = create_model('huber') #huber is the id of the model
```


<style  type="text/css" >
#T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow10_col0,#T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow10_col1,#T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow10_col2,#T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow10_col3,#T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow10_col4,#T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow10_col5{
            background:  yellow;
        }</style><table id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedf" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >MAE</th>        <th class="col_heading level0 col1" >MSE</th>        <th class="col_heading level0 col2" >RMSE</th>        <th class="col_heading level0 col3" >R2</th>        <th class="col_heading level0 col4" >RMSLE</th>        <th class="col_heading level0 col5" >MAPE</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedflevel0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow0_col0" class="data row0 col0" >961845.3602</td>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow0_col1" class="data row0 col1" >1487696780425.6975</td>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow0_col2" class="data row0 col2" >1219711.7612</td>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow0_col3" class="data row0 col3" >-0.0209</td>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow0_col4" class="data row0 col4" >0.4309</td>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow0_col5" class="data row0 col5" >0.4312</td>
            </tr>
            <tr>
                        <th id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedflevel0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow1_col0" class="data row1 col0" >929693.5132</td>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow1_col1" class="data row1 col1" >1671325484086.2356</td>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow1_col2" class="data row1 col2" >1292797.5418</td>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow1_col3" class="data row1 col3" >0.3338</td>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow1_col4" class="data row1 col4" >0.3211</td>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow1_col5" class="data row1 col5" >0.2699</td>
            </tr>
            <tr>
                        <th id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedflevel0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow2_col0" class="data row2 col0" >1104011.2956</td>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow2_col1" class="data row2 col1" >1961497099196.9431</td>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow2_col2" class="data row2 col2" >1400534.5762</td>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow2_col3" class="data row2 col3" >0.0615</td>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow2_col4" class="data row2 col4" >0.3346</td>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow2_col5" class="data row2 col5" >0.2930</td>
            </tr>
            <tr>
                        <th id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedflevel0_row3" class="row_heading level0 row3" >3</th>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow3_col0" class="data row3 col0" >1614610.4190</td>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow3_col1" class="data row3 col1" >4345951575348.9058</td>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow3_col2" class="data row3 col2" >2084694.6000</td>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow3_col3" class="data row3 col3" >-1.0927</td>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow3_col4" class="data row3 col4" >0.4125</td>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow3_col5" class="data row3 col5" >0.2693</td>
            </tr>
            <tr>
                        <th id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedflevel0_row4" class="row_heading level0 row4" >4</th>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow4_col0" class="data row4 col0" >1518981.7647</td>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow4_col1" class="data row4 col1" >3622916112112.9058</td>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow4_col2" class="data row4 col2" >1903395.9420</td>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow4_col3" class="data row4 col3" >-0.3658</td>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow4_col4" class="data row4 col4" >0.3714</td>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow4_col5" class="data row4 col5" >0.2814</td>
            </tr>
            <tr>
                        <th id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedflevel0_row5" class="row_heading level0 row5" >5</th>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow5_col0" class="data row5 col0" >1197463.1364</td>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow5_col1" class="data row5 col1" >2368848939314.6255</td>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow5_col2" class="data row5 col2" >1539106.5393</td>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow5_col3" class="data row5 col3" >-0.0315</td>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow5_col4" class="data row5 col4" >0.5390</td>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow5_col5" class="data row5 col5" >0.5655</td>
            </tr>
            <tr>
                        <th id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedflevel0_row6" class="row_heading level0 row6" >6</th>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow6_col0" class="data row6 col0" >2891914.8312</td>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow6_col1" class="data row6 col1" >20956733994889.3711</td>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow6_col2" class="data row6 col2" >4577852.5528</td>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow6_col3" class="data row6 col3" >-0.1428</td>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow6_col4" class="data row6 col4" >0.6344</td>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow6_col5" class="data row6 col5" >0.4357</td>
            </tr>
            <tr>
                        <th id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedflevel0_row7" class="row_heading level0 row7" >7</th>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow7_col0" class="data row7 col0" >650627.2049</td>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow7_col1" class="data row7 col1" >641819290500.7419</td>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow7_col2" class="data row7 col2" >801136.2496</td>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow7_col3" class="data row7 col3" >0.1298</td>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow7_col4" class="data row7 col4" >0.2623</td>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow7_col5" class="data row7 col5" >0.2392</td>
            </tr>
            <tr>
                        <th id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedflevel0_row8" class="row_heading level0 row8" >8</th>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow8_col0" class="data row8 col0" >2655096.4085</td>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow8_col1" class="data row8 col1" >28370429374755.0273</td>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow8_col2" class="data row8 col2" >5326389.9007</td>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow8_col3" class="data row8 col3" >-0.0869</td>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow8_col4" class="data row8 col4" >0.6209</td>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow8_col5" class="data row8 col5" >0.3863</td>
            </tr>
            <tr>
                        <th id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedflevel0_row9" class="row_heading level0 row9" >9</th>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow9_col0" class="data row9 col0" >1620466.3524</td>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow9_col1" class="data row9 col1" >3644626524919.4702</td>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow9_col2" class="data row9 col2" >1909090.4968</td>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow9_col3" class="data row9 col3" >-0.0093</td>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow9_col4" class="data row9 col4" >0.4200</td>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow9_col5" class="data row9 col5" >0.3811</td>
            </tr>
            <tr>
                        <th id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedflevel0_row10" class="row_heading level0 row10" >Mean</th>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow10_col0" class="data row10 col0" >1514471.0286</td>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow10_col1" class="data row10 col1" >6907184517554.9922</td>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow10_col2" class="data row10 col2" >2205471.0161</td>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow10_col3" class="data row10 col3" >-0.1225</td>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow10_col4" class="data row10 col4" >0.4347</td>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow10_col5" class="data row10 col5" >0.3553</td>
            </tr>
            <tr>
                        <th id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedflevel0_row11" class="row_heading level0 row11" >SD</th>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow11_col0" class="data row11 col0" >698728.7141</td>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow11_col1" class="data row11 col1" >9094765090727.6680</td>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow11_col2" class="data row11 col2" >1429364.2345</td>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow11_col3" class="data row11 col3" >0.3660</td>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow11_col4" class="data row11 col4" >0.1195</td>
                        <td id="T_128b467d_c48e_11eb_971c_3c9c0f5cfedfrow11_col5" class="data row11 col5" >0.0977</td>
            </tr>
    </tbody></table>



```python
# tune hyperparameters to optimize MAE
tuned_huber = tune_model(huber, optimize = 'MAE') #default is 'R2'
```


<style  type="text/css" >
#T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow10_col0,#T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow10_col1,#T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow10_col2,#T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow10_col3,#T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow10_col4,#T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow10_col5{
            background:  yellow;
        }</style><table id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedf" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >MAE</th>        <th class="col_heading level0 col1" >MSE</th>        <th class="col_heading level0 col2" >RMSE</th>        <th class="col_heading level0 col3" >R2</th>        <th class="col_heading level0 col4" >RMSLE</th>        <th class="col_heading level0 col5" >MAPE</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedflevel0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow0_col0" class="data row0 col0" >1280364.9568</td>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow0_col1" class="data row0 col1" >2430975157264.6777</td>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow0_col2" class="data row0 col2" >1559158.4773</td>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow0_col3" class="data row0 col3" >-0.6682</td>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow0_col4" class="data row0 col4" >0.5189</td>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow0_col5" class="data row0 col5" >0.5712</td>
            </tr>
            <tr>
                        <th id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedflevel0_row1" class="row_heading level0 row1" >1</th>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow1_col0" class="data row1 col0" >1174372.8762</td>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow1_col1" class="data row1 col1" >2510357986495.4492</td>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow1_col2" class="data row1 col2" >1584410.9273</td>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow1_col3" class="data row1 col3" >-0.0007</td>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow1_col4" class="data row1 col4" >0.3991</td>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow1_col5" class="data row1 col5" >0.3574</td>
            </tr>
            <tr>
                        <th id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedflevel0_row2" class="row_heading level0 row2" >2</th>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow2_col0" class="data row2 col0" >1236727.3732</td>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow2_col1" class="data row2 col1" >2139489270080.0415</td>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow2_col2" class="data row2 col2" >1462699.3095</td>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow2_col3" class="data row2 col3" >-0.0237</td>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow2_col4" class="data row2 col4" >0.3598</td>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow2_col5" class="data row2 col5" >0.3448</td>
            </tr>
            <tr>
                        <th id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedflevel0_row3" class="row_heading level0 row3" >3</th>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow3_col0" class="data row3 col0" >1489272.5436</td>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow3_col1" class="data row3 col1" >3718351580039.5698</td>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow3_col2" class="data row3 col2" >1928302.7719</td>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow3_col3" class="data row3 col3" >-0.7905</td>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow3_col4" class="data row3 col4" >0.3601</td>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow3_col5" class="data row3 col5" >0.2481</td>
            </tr>
            <tr>
                        <th id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedflevel0_row4" class="row_heading level0 row4" >4</th>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow4_col0" class="data row4 col0" >1452198.7090</td>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow4_col1" class="data row4 col1" >3846956142324.9072</td>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow4_col2" class="data row4 col2" >1961365.8869</td>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow4_col3" class="data row4 col3" >-0.4502</td>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow4_col4" class="data row4 col4" >0.3709</td>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow4_col5" class="data row4 col5" >0.2643</td>
            </tr>
            <tr>
                        <th id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedflevel0_row5" class="row_heading level0 row5" >5</th>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow5_col0" class="data row5 col0" >1379393.0361</td>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow5_col1" class="data row5 col1" >2621424276512.3242</td>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow5_col2" class="data row5 col2" >1619081.3063</td>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow5_col3" class="data row5 col3" >-0.1415</td>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow5_col4" class="data row5 col4" >0.5575</td>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow5_col5" class="data row5 col5" >0.6250</td>
            </tr>
            <tr>
                        <th id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedflevel0_row6" class="row_heading level0 row6" >6</th>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow6_col0" class="data row6 col0" >2939539.0391</td>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow6_col1" class="data row6 col1" >21839976697272.5781</td>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow6_col2" class="data row6 col2" >4673326.0851</td>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow6_col3" class="data row6 col3" >-0.1910</td>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow6_col4" class="data row6 col4" >0.6518</td>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow6_col5" class="data row6 col5" >0.4570</td>
            </tr>
            <tr>
                        <th id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedflevel0_row7" class="row_heading level0 row7" >7</th>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow7_col0" class="data row7 col0" >902760.8868</td>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow7_col1" class="data row7 col1" >1299533553777.5125</td>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow7_col2" class="data row7 col2" >1139970.8565</td>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow7_col3" class="data row7 col3" >-0.7619</td>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow7_col4" class="data row7 col4" >0.3481</td>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow7_col5" class="data row7 col5" >0.3374</td>
            </tr>
            <tr>
                        <th id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedflevel0_row8" class="row_heading level0 row8" >8</th>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow8_col0" class="data row8 col0" >2647644.4690</td>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow8_col1" class="data row8 col1" >28420597864879.3281</td>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow8_col2" class="data row8 col2" >5331097.2477</td>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow8_col3" class="data row8 col3" >-0.0888</td>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow8_col4" class="data row8 col4" >0.6315</td>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow8_col5" class="data row8 col5" >0.4024</td>
            </tr>
            <tr>
                        <th id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedflevel0_row9" class="row_heading level0 row9" >9</th>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow9_col0" class="data row9 col0" >1595107.6072</td>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow9_col1" class="data row9 col1" >3709630392105.6753</td>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow9_col2" class="data row9 col2" >1926040.0806</td>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow9_col3" class="data row9 col3" >-0.0273</td>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow9_col4" class="data row9 col4" >0.4340</td>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow9_col5" class="data row9 col5" >0.3902</td>
            </tr>
            <tr>
                        <th id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedflevel0_row10" class="row_heading level0 row10" >Mean</th>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow10_col0" class="data row10 col0" >1609738.1497</td>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow10_col1" class="data row10 col1" >7253729292075.2061</td>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow10_col2" class="data row10 col2" >2318545.2949</td>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow10_col3" class="data row10 col3" >-0.3144</td>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow10_col4" class="data row10 col4" >0.4632</td>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow10_col5" class="data row10 col5" >0.3998</td>
            </tr>
            <tr>
                        <th id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedflevel0_row11" class="row_heading level0 row11" >SD</th>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow11_col0" class="data row11 col0" >622511.6068</td>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow11_col1" class="data row11 col1" >9090023382008.1621</td>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow11_col2" class="data row11 col2" >1370429.4975</td>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow11_col3" class="data row11 col3" >0.3053</td>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow11_col4" class="data row11 col4" >0.1113</td>
                        <td id="T_170a95d0_c48e_11eb_8aef_3c9c0f5cfedfrow11_col5" class="data row11 col5" >0.1156</td>
            </tr>
    </tbody></table>



```python
# plot a model 
plot_model(tuned_huber)
```


    
![png](output_27_0.png)
    



```python
# evaluate a model 
evaluate_model(tuned_huber)
```


    interactive(children=(ToggleButtons(description='Plot Type:', icons=('',), options=(('Hyperparameters', 'param…



```python
# finalize model
huber_final = finalize_model(tuned_huber)
```


```python
pred_holdout = predict_model(huber_final)
```


<style  type="text/css" >
</style><table id="T_29183dbf_c48e_11eb_894a_3c9c0f5cfedf" ><thead>    <tr>        <th class="blank level0" ></th>        <th class="col_heading level0 col0" >Model</th>        <th class="col_heading level0 col1" >MAE</th>        <th class="col_heading level0 col2" >MSE</th>        <th class="col_heading level0 col3" >RMSE</th>        <th class="col_heading level0 col4" >R2</th>        <th class="col_heading level0 col5" >RMSLE</th>        <th class="col_heading level0 col6" >MAPE</th>    </tr></thead><tbody>
                <tr>
                        <th id="T_29183dbf_c48e_11eb_894a_3c9c0f5cfedflevel0_row0" class="row_heading level0 row0" >0</th>
                        <td id="T_29183dbf_c48e_11eb_894a_3c9c0f5cfedfrow0_col0" class="data row0 col0" >Huber Regressor</td>
                        <td id="T_29183dbf_c48e_11eb_894a_3c9c0f5cfedfrow0_col1" class="data row0 col1" >1712036.2452</td>
                        <td id="T_29183dbf_c48e_11eb_894a_3c9c0f5cfedfrow0_col2" class="data row0 col2" >6125950157434.3662</td>
                        <td id="T_29183dbf_c48e_11eb_894a_3c9c0f5cfedfrow0_col3" class="data row0 col3" >2475065.6875</td>
                        <td id="T_29183dbf_c48e_11eb_894a_3c9c0f5cfedfrow0_col4" class="data row0 col4" >-0.0095</td>
                        <td id="T_29183dbf_c48e_11eb_894a_3c9c0f5cfedfrow0_col5" class="data row0 col5" >0.5101</td>
                        <td id="T_29183dbf_c48e_11eb_894a_3c9c0f5cfedfrow0_col6" class="data row0 col6" >0.4503</td>
            </tr>
    </tbody></table>



```python
pd.options.display.float_format = '{:,.2f}'.format
```


```python
pred_holdout.rename(columns={"Label":"Revenue Prediction"})
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
      <th>City Group</th>
      <th>Type</th>
      <th>P2</th>
      <th>P3</th>
      <th>P4</th>
      <th>P13</th>
      <th>P26</th>
      <th>P27</th>
      <th>P28</th>
      <th>P29</th>
      <th>...</th>
      <th>P36_5</th>
      <th>P37_1</th>
      <th>P37_2</th>
      <th>P37_3</th>
      <th>P37_4</th>
      <th>P37_5</th>
      <th>P37_6</th>
      <th>P37_8</th>
      <th>revenue</th>
      <th>Revenue Prediction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2</td>
      <td>5.00</td>
      <td>4.00</td>
      <td>4.00</td>
      <td>4.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>3.00</td>
      <td>3.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>2,390,534.00</td>
      <td>4,151,066.47</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2</td>
      <td>3.00</td>
      <td>4.00</td>
      <td>4.00</td>
      <td>5.00</td>
      <td>3.00</td>
      <td>2.00</td>
      <td>2.00</td>
      <td>3.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4,015,749.00</td>
      <td>4,132,673.90</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>2</td>
      <td>5.00</td>
      <td>4.00</td>
      <td>4.00</td>
      <td>4.00</td>
      <td>5.00</td>
      <td>1.00</td>
      <td>5.00</td>
      <td>1.00</td>
      <td>...</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>13,575,224.00</td>
      <td>4,200,567.88</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>7.50</td>
      <td>7.50</td>
      <td>6.00</td>
      <td>7.50</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>10.00</td>
      <td>7.50</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>6,782,425.00</td>
      <td>4,242,965.19</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>2</td>
      <td>4.00</td>
      <td>5.00</td>
      <td>4.00</td>
      <td>3.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>2.00</td>
      <td>3.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4,651,866.00</td>
      <td>4,124,117.21</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>1</td>
      <td>3.00</td>
      <td>4.00</td>
      <td>4.00</td>
      <td>5.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>2.00</td>
      <td>3.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>3,351,383.00</td>
      <td>4,111,789.17</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>1</td>
      <td>3.00</td>
      <td>4.00</td>
      <td>3.00</td>
      <td>5.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>3.00</td>
      <td>2.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>3,258,837.00</td>
      <td>4,119,328.58</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>1</td>
      <td>4.00</td>
      <td>4.00</td>
      <td>4.00</td>
      <td>4.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>2.00</td>
      <td>3.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>2,999,068.00</td>
      <td>4,121,900.98</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>2</td>
      <td>7.50</td>
      <td>6.00</td>
      <td>6.00</td>
      <td>7.50</td>
      <td>10.00</td>
      <td>10.00</td>
      <td>10.00</td>
      <td>7.50</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>4,136,425.00</td>
      <td>4,265,040.08</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>1</td>
      <td>5.00</td>
      <td>5.00</td>
      <td>4.00</td>
      <td>5.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4.00</td>
      <td>1.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>3,939,804.00</td>
      <td>4,166,425.93</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0</td>
      <td>1</td>
      <td>5.00</td>
      <td>4.00</td>
      <td>5.00</td>
      <td>5.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>3.00</td>
      <td>3.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>7,217,634.00</td>
      <td>4,141,867.96</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>2</td>
      <td>5.00</td>
      <td>4.00</td>
      <td>4.00</td>
      <td>5.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>3.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>2,083,447.00</td>
      <td>4,109,867.49</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0</td>
      <td>2</td>
      <td>7.50</td>
      <td>6.00</td>
      <td>7.50</td>
      <td>6.00</td>
      <td>5.00</td>
      <td>2.50</td>
      <td>7.50</td>
      <td>2.50</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>3,871,344.00</td>
      <td>4,227,520.67</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0</td>
      <td>2</td>
      <td>5.00</td>
      <td>4.00</td>
      <td>5.00</td>
      <td>4.00</td>
      <td>2.00</td>
      <td>1.00</td>
      <td>3.00</td>
      <td>3.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>2,156,098.00</td>
      <td>4,145,788.86</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1</td>
      <td>1</td>
      <td>6.00</td>
      <td>6.00</td>
      <td>6.00</td>
      <td>7.50</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>10.00</td>
      <td>2.50</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>3,745,135.00</td>
      <td>4,233,960.29</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0</td>
      <td>1</td>
      <td>5.00</td>
      <td>4.00</td>
      <td>4.00</td>
      <td>4.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>5.00</td>
      <td>1.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>6,694,797.00</td>
      <td>4,180,713.90</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1</td>
      <td>1</td>
      <td>4.00</td>
      <td>5.00</td>
      <td>5.00</td>
      <td>5.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>2.00</td>
      <td>3.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>3,778,621.00</td>
      <td>4,123,015.14</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0</td>
      <td>1</td>
      <td>5.00</td>
      <td>4.00</td>
      <td>4.00</td>
      <td>5.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>2.00</td>
      <td>3.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>3,600,467.00</td>
      <td>4,129,883.83</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0</td>
      <td>1</td>
      <td>7.50</td>
      <td>6.00</td>
      <td>6.00</td>
      <td>6.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>7.50</td>
      <td>5.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4,350,573.00</td>
      <td>4,219,515.31</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0</td>
      <td>1</td>
      <td>5.00</td>
      <td>4.00</td>
      <td>4.00</td>
      <td>5.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4.00</td>
      <td>3.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4,100,886.00</td>
      <td>4,157,994.95</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0</td>
      <td>2</td>
      <td>3.00</td>
      <td>4.00</td>
      <td>4.00</td>
      <td>5.00</td>
      <td>4.00</td>
      <td>5.00</td>
      <td>1.00</td>
      <td>3.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>5,166,635.00</td>
      <td>4,118,714.83</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1</td>
      <td>2</td>
      <td>6.00</td>
      <td>6.00</td>
      <td>6.00</td>
      <td>7.50</td>
      <td>7.50</td>
      <td>2.50</td>
      <td>5.00</td>
      <td>7.50</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4,780,607.00</td>
      <td>4,175,353.21</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1</td>
      <td>2</td>
      <td>4.50</td>
      <td>6.00</td>
      <td>7.50</td>
      <td>7.50</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>5.00</td>
      <td>2.50</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>5,444,227.00</td>
      <td>4,162,690.65</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0</td>
      <td>1</td>
      <td>5.00</td>
      <td>4.00</td>
      <td>5.00</td>
      <td>5.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>5.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>3,347,767.00</td>
      <td>4,182,354.97</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1</td>
      <td>2</td>
      <td>4.00</td>
      <td>4.00</td>
      <td>4.00</td>
      <td>5.00</td>
      <td>1.00</td>
      <td>5.00</td>
      <td>1.00</td>
      <td>3.00</td>
      <td>...</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>5,525,735.00</td>
      <td>4,130,038.53</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1</td>
      <td>1</td>
      <td>6.00</td>
      <td>6.00</td>
      <td>4.50</td>
      <td>6.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>5.00</td>
      <td>7.50</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>3,752,885.00</td>
      <td>4,169,473.22</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1</td>
      <td>1</td>
      <td>3.00</td>
      <td>4.00</td>
      <td>4.00</td>
      <td>5.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>3.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1,763,231.00</td>
      <td>4,094,673.14</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1</td>
      <td>1</td>
      <td>4.00</td>
      <td>4.00</td>
      <td>4.00</td>
      <td>4.00</td>
      <td>3.00</td>
      <td>1.00</td>
      <td>2.00</td>
      <td>3.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>5,286,212.00</td>
      <td>4,126,821.21</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1</td>
      <td>1</td>
      <td>4.00</td>
      <td>4.00</td>
      <td>4.00</td>
      <td>5.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>3.00</td>
      <td>3.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>7,592,272.00</td>
      <td>4,128,686.30</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1</td>
      <td>2</td>
      <td>4.00</td>
      <td>4.00</td>
      <td>4.00</td>
      <td>4.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>2.00</td>
      <td>3.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>2,021,934.00</td>
      <td>4,116,180.56</td>
    </tr>
    <tr>
      <th>30</th>
      <td>1</td>
      <td>1</td>
      <td>2.00</td>
      <td>4.00</td>
      <td>3.00</td>
      <td>5.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>2.00</td>
      <td>3.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1,882,131.00</td>
      <td>4,095,705.18</td>
    </tr>
    <tr>
      <th>31</th>
      <td>0</td>
      <td>2</td>
      <td>7.50</td>
      <td>7.50</td>
      <td>4.50</td>
      <td>7.50</td>
      <td>7.50</td>
      <td>5.00</td>
      <td>10.00</td>
      <td>5.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1,847,826.00</td>
      <td>4,259,017.11</td>
    </tr>
    <tr>
      <th>32</th>
      <td>1</td>
      <td>1</td>
      <td>1.00</td>
      <td>4.00</td>
      <td>3.00</td>
      <td>5.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>2.00</td>
      <td>3.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4,263,629.00</td>
      <td>4,087,405.90</td>
    </tr>
    <tr>
      <th>33</th>
      <td>0</td>
      <td>1</td>
      <td>5.00</td>
      <td>4.00</td>
      <td>4.00</td>
      <td>5.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4.00</td>
      <td>3.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>9,652,350.00</td>
      <td>4,157,138.63</td>
    </tr>
    <tr>
      <th>34</th>
      <td>0</td>
      <td>2</td>
      <td>5.00</td>
      <td>4.00</td>
      <td>5.00</td>
      <td>5.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>3.00</td>
      <td>1.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>2,018,785.00</td>
      <td>4,146,369.75</td>
    </tr>
    <tr>
      <th>35</th>
      <td>0</td>
      <td>1</td>
      <td>7.50</td>
      <td>6.00</td>
      <td>6.00</td>
      <td>7.50</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>7.50</td>
      <td>7.50</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>8,630,682.00</td>
      <td>4,213,680.60</td>
    </tr>
    <tr>
      <th>36</th>
      <td>0</td>
      <td>2</td>
      <td>5.00</td>
      <td>4.00</td>
      <td>4.00</td>
      <td>5.00</td>
      <td>5.00</td>
      <td>3.00</td>
      <td>5.00</td>
      <td>1.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>2,525,375.00</td>
      <td>4,179,875.85</td>
    </tr>
    <tr>
      <th>37</th>
      <td>1</td>
      <td>2</td>
      <td>4.00</td>
      <td>5.00</td>
      <td>4.00</td>
      <td>5.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>2.00</td>
      <td>3.00</td>
      <td>...</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4,758,476.00</td>
      <td>4,135,763.41</td>
    </tr>
    <tr>
      <th>38</th>
      <td>0</td>
      <td>1</td>
      <td>5.00</td>
      <td>4.00</td>
      <td>4.00</td>
      <td>5.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>3.00</td>
      <td>3.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>9,262,754.00</td>
      <td>4,152,100.83</td>
    </tr>
    <tr>
      <th>39</th>
      <td>1</td>
      <td>2</td>
      <td>4.00</td>
      <td>4.00</td>
      <td>4.00</td>
      <td>5.00</td>
      <td>2.00</td>
      <td>3.00</td>
      <td>3.00</td>
      <td>1.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>3,248,660.00</td>
      <td>4,155,477.40</td>
    </tr>
    <tr>
      <th>40</th>
      <td>1</td>
      <td>1</td>
      <td>4.00</td>
      <td>4.00</td>
      <td>4.00</td>
      <td>5.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>3.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1,270,499.00</td>
      <td>4,101,001.16</td>
    </tr>
    <tr>
      <th>41</th>
      <td>0</td>
      <td>2</td>
      <td>5.00</td>
      <td>5.00</td>
      <td>4.00</td>
      <td>4.00</td>
      <td>4.00</td>
      <td>5.00</td>
      <td>3.00</td>
      <td>3.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>3,784,230.00</td>
      <td>4,162,302.14</td>
    </tr>
  </tbody>
</table>
<p>42 rows × 186 columns</p>
</div>




```python

```
