```python
#Do wines with higher alcoholic content receive better ratings?
```


```python
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas_profiling as pp
from matplotlib import __version__ as mpv
```


```python
# this function will allow me to set up my data in a prettier way 
def pretty(data):
    return pd.read_csv(data, delimiter = ';', encoding='utf-8')
#loads dataframe into variable
df = pretty('winequality-red.csv')
```


```python
# number of rows and columns
df.shape
```




    (1599, 12)




```python
#Basic stats of the current dataset
df.describe()
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
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
      <td>1599.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>8.319637</td>
      <td>0.527821</td>
      <td>0.270976</td>
      <td>2.538806</td>
      <td>0.087467</td>
      <td>15.874922</td>
      <td>46.467792</td>
      <td>0.996747</td>
      <td>3.311113</td>
      <td>0.658149</td>
      <td>10.422983</td>
      <td>5.636023</td>
    </tr>
    <tr>
      <td>std</td>
      <td>1.741096</td>
      <td>0.179060</td>
      <td>0.194801</td>
      <td>1.409928</td>
      <td>0.047065</td>
      <td>10.460157</td>
      <td>32.895324</td>
      <td>0.001887</td>
      <td>0.154386</td>
      <td>0.169507</td>
      <td>1.065668</td>
      <td>0.807569</td>
    </tr>
    <tr>
      <td>min</td>
      <td>4.600000</td>
      <td>0.120000</td>
      <td>0.000000</td>
      <td>0.900000</td>
      <td>0.012000</td>
      <td>1.000000</td>
      <td>6.000000</td>
      <td>0.990070</td>
      <td>2.740000</td>
      <td>0.330000</td>
      <td>8.400000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>7.100000</td>
      <td>0.390000</td>
      <td>0.090000</td>
      <td>1.900000</td>
      <td>0.070000</td>
      <td>7.000000</td>
      <td>22.000000</td>
      <td>0.995600</td>
      <td>3.210000</td>
      <td>0.550000</td>
      <td>9.500000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>7.900000</td>
      <td>0.520000</td>
      <td>0.260000</td>
      <td>2.200000</td>
      <td>0.079000</td>
      <td>14.000000</td>
      <td>38.000000</td>
      <td>0.996750</td>
      <td>3.310000</td>
      <td>0.620000</td>
      <td>10.200000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>9.200000</td>
      <td>0.640000</td>
      <td>0.420000</td>
      <td>2.600000</td>
      <td>0.090000</td>
      <td>21.000000</td>
      <td>62.000000</td>
      <td>0.997835</td>
      <td>3.400000</td>
      <td>0.730000</td>
      <td>11.100000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <td>max</td>
      <td>15.900000</td>
      <td>1.580000</td>
      <td>1.000000</td>
      <td>15.500000</td>
      <td>0.611000</td>
      <td>72.000000</td>
      <td>289.000000</td>
      <td>1.003690</td>
      <td>4.010000</td>
      <td>2.000000</td>
      <td>14.900000</td>
      <td>8.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#the type of data i have
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1599 entries, 0 to 1598
    Data columns (total 12 columns):
    fixed acidity           1599 non-null float64
    volatile acidity        1599 non-null float64
    citric acid             1599 non-null float64
    residual sugar          1599 non-null float64
    chlorides               1599 non-null float64
    free sulfur dioxide     1599 non-null float64
    total sulfur dioxide    1599 non-null float64
    density                 1599 non-null float64
    pH                      1599 non-null float64
    sulphates               1599 non-null float64
    alcohol                 1599 non-null float64
    quality                 1599 non-null int64
    dtypes: float64(11), int64(1)
    memory usage: 150.0 KB
    


```python
#checks for missing values 
df.isnull().sum()
```




    fixed acidity           0
    volatile acidity        0
    citric acid             0
    residual sugar          0
    chlorides               0
    free sulfur dioxide     0
    total sulfur dioxide    0
    density                 0
    pH                      0
    sulphates               0
    alcohol                 0
    quality                 0
    dtype: int64




```python
from sklearn.model_selection import train_test_split
```


```python
# Splitting the data so i can cross validaty models according to their effectiveness
seed = 22
x_train, x_test, y_train, y_test = train_test_split(df[df.columns[:-1]],
                                                    df['quality'],
                                                    train_size=0.75,
                                                    random_state=seed,
                                                    stratify=df['quality'])
```


```python
# lookin at my split data
x_train.head()
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
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>367</td>
      <td>10.4</td>
      <td>0.575</td>
      <td>0.61</td>
      <td>2.6</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>24.0</td>
      <td>1.00000</td>
      <td>3.16</td>
      <td>0.69</td>
      <td>9.0</td>
    </tr>
    <tr>
      <td>383</td>
      <td>8.3</td>
      <td>0.260</td>
      <td>0.42</td>
      <td>2.0</td>
      <td>0.080</td>
      <td>11.0</td>
      <td>27.0</td>
      <td>0.99740</td>
      <td>3.21</td>
      <td>0.80</td>
      <td>9.4</td>
    </tr>
    <tr>
      <td>1150</td>
      <td>8.2</td>
      <td>0.330</td>
      <td>0.32</td>
      <td>2.8</td>
      <td>0.067</td>
      <td>4.0</td>
      <td>12.0</td>
      <td>0.99473</td>
      <td>3.30</td>
      <td>0.76</td>
      <td>12.8</td>
    </tr>
    <tr>
      <td>725</td>
      <td>9.0</td>
      <td>0.660</td>
      <td>0.17</td>
      <td>3.0</td>
      <td>0.077</td>
      <td>5.0</td>
      <td>13.0</td>
      <td>0.99760</td>
      <td>3.29</td>
      <td>0.55</td>
      <td>10.4</td>
    </tr>
    <tr>
      <td>479</td>
      <td>9.4</td>
      <td>0.685</td>
      <td>0.11</td>
      <td>2.7</td>
      <td>0.077</td>
      <td>6.0</td>
      <td>31.0</td>
      <td>0.99840</td>
      <td>3.19</td>
      <td>0.70</td>
      <td>10.1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# y train set 
y_train.head()
```




    367     5
    383     6
    1150    7
    725     5
    479     6
    Name: quality, dtype: int64




```python
#lookin at the value counts
y_train.value_counts()
```




    5    511
    6    478
    7    149
    4     40
    8     13
    3      8
    Name: quality, dtype: int64




```python
# turning my quality variables into binary 
def NewQuality(quality):
    if quality > 5:
        return 1
    else:
        return 0
```


```python
#applies function
y_binary_train = y_train.map(NewQuality)
```


```python
#confirming that my binary proccessed correctly
y_binary_train.value_counts()
```




    1    640
    0    559
    Name: quality, dtype: int64




```python
y = y_binary_train.value_counts()
sns.barplot(['Bad','Good'],y.values)
plt.show()
```


    
![png](output_15_0.png)
    



```python
#matrix of relationships between each variable for an instant examination.. helped as jumping point to think ahead 
#as far as regression 
sns.pairplot(x_train)
sns.set()
```


    
![png](output_16_0.png)
    



```python
# i want to look at plots that caught my attention 
plt.title('fixed acidity')
num_bins= 5
plt.hist(x_train['fixed acidity'], num_bins, alpha = 0.5)
plt.show()
```


    
![png](output_17_0.png)
    



```python
# fixed acidity has outliers and seems to peek at 8ish. 
```


```python
plt.title('fixed acidity')
plt.hist(x_train['volatile acidity'], alpha = 0.5)
plt.show()
```


    
![png](output_19_0.png)
    



```python
# has a few outliers with "higer" values
```


```python
plt.title('Citric Acid')
plt.hist(x_train['citric acid'], alpha = 0.5)
plt.show()
```


    
![png](output_21_0.png)
    



```python
# Most wines have 0g of citric acid! [this works for my heart burn] but spikes at 0.2 to 0.5
```


```python
plt.title('Density')
plt.hist(x_train['density'], alpha = 0.5)
plt.show()
```


    
![png](output_23_0.png)
    



```python
# FINALLY A NORMAL DISTRIBUTION!!! a few outliers 
```


```python
plt.title('Residual Sugar')
plt.hist(x_train['residual sugar'], alpha = 0.5)
plt.show()
```


    
![png](output_25_0.png)
    



```python
# distribution skewed to the right. Shows alot of outliers 
```


```python
plt.title('Alcohol')
plt.hist(x_train['alcohol'], alpha = 0.5)
plt.show()
```


    
![png](output_27_0.png)
    



```python
# this one i was interested in seeing, seems to be skewed - i want to see how quality changes according to alc content 
```


```python
# Compute the correlation matrix
corr = x_train.corr()
corr.head()
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
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>fixed acidity</td>
      <td>1.000000</td>
      <td>-0.249397</td>
      <td>0.680888</td>
      <td>0.137357</td>
      <td>0.099481</td>
      <td>-0.149934</td>
      <td>-0.106284</td>
      <td>0.657164</td>
      <td>-0.681114</td>
      <td>0.181369</td>
      <td>-0.063422</td>
    </tr>
    <tr>
      <td>volatile acidity</td>
      <td>-0.249397</td>
      <td>1.000000</td>
      <td>-0.553954</td>
      <td>0.021662</td>
      <td>0.048290</td>
      <td>0.003683</td>
      <td>0.094074</td>
      <td>0.047382</td>
      <td>0.220695</td>
      <td>-0.260307</td>
      <td>-0.211753</td>
    </tr>
    <tr>
      <td>citric acid</td>
      <td>0.680888</td>
      <td>-0.553954</td>
      <td>1.000000</td>
      <td>0.117189</td>
      <td>0.221447</td>
      <td>-0.067096</td>
      <td>0.024763</td>
      <td>0.363158</td>
      <td>-0.544870</td>
      <td>0.318137</td>
      <td>0.093832</td>
    </tr>
    <tr>
      <td>residual sugar</td>
      <td>0.137357</td>
      <td>0.021662</td>
      <td>0.117189</td>
      <td>1.000000</td>
      <td>0.077776</td>
      <td>0.220518</td>
      <td>0.209586</td>
      <td>0.385265</td>
      <td>-0.095888</td>
      <td>0.023899</td>
      <td>0.021885</td>
    </tr>
    <tr>
      <td>chlorides</td>
      <td>0.099481</td>
      <td>0.048290</td>
      <td>0.221447</td>
      <td>0.077776</td>
      <td>1.000000</td>
      <td>0.001587</td>
      <td>0.050776</td>
      <td>0.206731</td>
      <td>-0.268776</td>
      <td>0.374495</td>
      <td>-0.213196</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(18, 15))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap
sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
```




    <matplotlib.axes._subplots.AxesSubplot at 0x12ea8f667c8>




    
![png](output_30_1.png)
    


 ## reduction method


```python
# Feature Scaling.. i used standardization because i want the scaled features 
# to not be perfectly centered at zero wit unit variance. i think this will help when
#i run it on the test set 
#additionally when i ran head() on the set i noticed that all of my features were numeric,
#some have different scales 
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(x_train)
#dont apply results to test set
#X_test = sc.transform(x_test)
```


```python
# using PCA here to check the covariance between the features and eliminate the ones with shows less covariance towards dependent variable¶
#from sklearn.decomposition import PCA
#pca = PCA(n_components=2)
#X_train = pca.fit(x_train).transform(x_train)
#dont apply results to test set
#X_test= pca.fit(x_test).transform(x_test)
#ex = pca.explained_variance_ratio_
```


```python
#pca.n_components_
```


```python
#print(ex)
```


```python
## Using selectktest for feature selection
#from sklearn.feature_selection import SelectKBest, f_classif, chi2
#feature_selector = SelectKBest(f_classif, k='2')
#X_scaled = feature_selector.fit_transform(x_train,y_binary_train.values.flatten())
#best_features = feature_selector.get_support()
#print(best_features)
```


```python
# yikes this doesnt look correct at all.. im going to do more research on this
```

## Model Selection 


```python
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor

#random forest regressor 
pipeline = make_pipeline(preprocessing.StandardScaler(), 
                         RandomForestRegressor(n_estimators=100))
print(pipeline.get_params())
```

    {'memory': None, 'steps': [('standardscaler', StandardScaler(copy=True, with_mean=True, with_std=True)), ('randomforestregressor', RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
                          max_features='auto', max_leaf_nodes=None,
                          min_impurity_decrease=0.0, min_impurity_split=None,
                          min_samples_leaf=1, min_samples_split=2,
                          min_weight_fraction_leaf=0.0, n_estimators=100,
                          n_jobs=None, oob_score=False, random_state=None,
                          verbose=0, warm_start=False))], 'verbose': False, 'standardscaler': StandardScaler(copy=True, with_mean=True, with_std=True), 'randomforestregressor': RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
                          max_features='auto', max_leaf_nodes=None,
                          min_impurity_decrease=0.0, min_impurity_split=None,
                          min_samples_leaf=1, min_samples_split=2,
                          min_weight_fraction_leaf=0.0, n_estimators=100,
                          n_jobs=None, oob_score=False, random_state=None,
                          verbose=0, warm_start=False), 'standardscaler__copy': True, 'standardscaler__with_mean': True, 'standardscaler__with_std': True, 'randomforestregressor__bootstrap': True, 'randomforestregressor__criterion': 'mse', 'randomforestregressor__max_depth': None, 'randomforestregressor__max_features': 'auto', 'randomforestregressor__max_leaf_nodes': None, 'randomforestregressor__min_impurity_decrease': 0.0, 'randomforestregressor__min_impurity_split': None, 'randomforestregressor__min_samples_leaf': 1, 'randomforestregressor__min_samples_split': 2, 'randomforestregressor__min_weight_fraction_leaf': 0.0, 'randomforestregressor__n_estimators': 100, 'randomforestregressor__n_jobs': None, 'randomforestregressor__oob_score': False, 'randomforestregressor__random_state': None, 'randomforestregressor__verbose': 0, 'randomforestregressor__warm_start': False}
    


```python
#going to use this to declare my hyperparameters
hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                  'randomforestregressor__max_depth': [None, 5, 3, 1]}
 
```


```python
# Tune model using cv pipeline
clf = GridSearchCV(pipeline, hyperparameters, cv=10)
clf.fit(X_train, y_train)
```




    GridSearchCV(cv=10, error_score='raise-deprecating',
                 estimator=Pipeline(memory=None,
                                    steps=[('standardscaler',
                                            StandardScaler(copy=True,
                                                           with_mean=True,
                                                           with_std=True)),
                                           ('randomforestregressor',
                                            RandomForestRegressor(bootstrap=True,
                                                                  criterion='mse',
                                                                  max_depth=None,
                                                                  max_features='auto',
                                                                  max_leaf_nodes=None,
                                                                  min_impurity_decrease=0.0,
                                                                  min_impurity_split=None,
                                                                  min_...
                                                                  min_weight_fraction_leaf=0.0,
                                                                  n_estimators=100,
                                                                  n_jobs=None,
                                                                  oob_score=False,
                                                                  random_state=None,
                                                                  verbose=0,
                                                                  warm_start=False))],
                                    verbose=False),
                 iid='warn', n_jobs=None,
                 param_grid={'randomforestregressor__max_depth': [None, 5, 3, 1],
                             'randomforestregressor__max_features': ['auto', 'sqrt',
                                                                     'log2']},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring=None, verbose=0)




```python
print(clf.best_params_)
```

    {'randomforestregressor__max_depth': None, 'randomforestregressor__max_features': 'sqrt'}
    


```python
print(clf.refit)
```

    True
    


```python
from sklearn.linear_model import LogisticRegression
```


```python
# not really sure if this one works correctly.. i think i should be using multinomial 
# but i also think that because my Y is binary.. it should be fine.. 
#logistic regression model 
# Train and fit model
logreg = LogisticRegression(class_weight='balanced', random_state=seed, solver='liblinear')
logreg.fit(X_train, y_train)
```

    C:\Users\Aritzi Silva\Anaconda3\lib\site-packages\sklearn\linear_model\logistic.py:469: FutureWarning: Default multi_class will be changed to 'auto' in 0.22. Specify the multi_class option to silence this warning.
      "this warning.", FutureWarning)
    




    LogisticRegression(C=1.0, class_weight='balanced', dual=False,
                       fit_intercept=True, intercept_scaling=1, l1_ratio=None,
                       max_iter=100, multi_class='warn', n_jobs=None, penalty='l2',
                       random_state=22, solver='liblinear', tol=0.0001, verbose=0,
                       warm_start=False)




```python
# train the confidence of knn
#KNN model selection
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(X_train, y_train)
```




    KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                         metric_params=None, n_jobs=None, n_neighbors=20, p=2,
                         weights='uniform')




```python

```
