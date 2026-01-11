## Feature Engineering
- A feature also called a dimension is an input variable used to generate model predictions.
- Feature engineering is the process of transforming raw data into relevant information.

<hr>

## Theoretical Foundations

**Understanding Features**
- Features are the input variables used by machine learning models to make predictions.
- Each feature represents a specific aspect of the data.

| Type of Feature | Detail |
|:---|:---|
| Numerical | Continuous values (e.g. height, weight) or Discrete values (e.g. counts) |
| Categorical | Non-numerical values that represent categories (e.g. color, brand).  |
| Ordinal | Categorical variables with a clear ordering (e.g. high school < bachelor < master). |
| Binary | Variables that can take on one of two possible values (e.g. yes/no, true/false). |

**Feature Representation**
- The way features are represented can greatly affect a model’s ability to learn.
- Different algorithms require different types of feature representations.

| Models | Detail |
|:---|:---|
| Linear | Perform well with linearly separable data. |
| Tree | Naturally handle linear and non-linear relationships. |
      
**Curse of Dimensionality**
- As the number of features increases, the volume of the space increases leading to sparsity.
- In high-dimensional space, data points become less similar making it difficult for algorithms to generalize well.
- Effective feature engineering can mitigate this by reducing dimensionality.
 
**Feature Importance**
- Understanding which features contribute most to the model’s predictions can guide feature selection.
- Techniques like feature importance scores from tree-based models or recursive feature elimination can help.

<hr>

## Concepts Covered

### Data Standardization
- Data standardization involves scaling your data to have a mean of zero and a standard deviation of one.
- This process is particularly useful when features have different units and scales.

### Example
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)
```

---

### Data Normalization
- Normalization scales the features to a range between 0 and 1.
- This technique is beneficial for algorithms that rely on distance measurements, like KNN.

### Example
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)
```

---

### Encoding Categorical Data
- Categorical data needs to be converted into numerical format for most machine learning algorithms.
- Common techniques include one-hot encoding and label encoding.

### Example
```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X_categorical).toarray()
```

---

### Sklearn ColumnTransformer
- This allows you to apply different preprocessing steps to different columns of your dataset in a concise way.

### Example
```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ]
)

X_transformed = preprocessor.fit_transform(X)
```

---

### Sklearn Pipeline
- This enables you to streamline the preprocessing and modeling steps into a single pipeline.

### Example
```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', SomeEstimator())
])

pipeline.fit(X_train, y_train)
```

---

### Handling Mixed Variables
- When your dataset contains both numerical and categorical variables,
- It's important to apply appropriate preprocessing to each type.
- Use `ColumnTransformer` as mentioned above for effective handling.

---

### Missing Categorical Data
- Most common way to handle missing data in categorical variable is to replace them with most frequent category.

### Example
```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='most_frequent')
X_imputed = imputer.fit_transform(X_categorical)
```

---

### KNNImputer
- It uses KNN algorithm to impute missing values, considering the similarity between data points.

### Example
```python
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X)
```

---

### SimpleImputer
- It is a simple way to handle missing values using different strategies (mean, median, most frequent, constant).

### Example
```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X_numeric)
```

---

### Outlier Detection
- Outliers can significantly impact the performance of a machine learning model (but not always).
- Several techniques can be employed for outlier detection, like :

### Using IQR
- Interquartile Range detects outliers by calculating the range between the first (Q1) and third quartiles (Q3).

### Example
```python
Q1 = np.percentile(X, 25)
Q3 = np.percentile(X, 75)
IQR = Q3 - Q1
outliers = (X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))
```

### Using Z-Score
- Z-score measures how many standard deviations away an element is from the mean.
- A common threshold is 3 to -3.

### Example
```python
X_mean = X.mean()
X_std = X.std()

X_Zscore = (X - X.mean())/X.std()

outliers = (X_Zscore > 3) | (X_Zscore < -3)
```

### Using Winsorization
- Winsorization involves capping extreme values to reduce the impact of outliers.

### Example
```python
upper_limit = np.percentile(X, 95)
lower_limit = np.percentile(X, 5)

outliers = (X < lower_limit) | (X > upper_limit)
```

---

### Function Transformer
- The `FunctionTransformer` allows you to apply any custom function to your data as part of a pipeline.

### Example
```python
from sklearn.preprocessing import FunctionTransformer

def custom_function(X):
    return X ** 2

transformer = FunctionTransformer(func=custom_function)
X_transformed = transformer.fit_transform(X)
```

---

### Power Transformer
- The `PowerTransformer` can help stabilize variance and make the data more gaussian-like.
- This is useful for improving the performance of models that assume normally distributed data.

### Example
```python
from sklearn.preprocessing import PowerTransformer

transformer = PowerTransformer()
X_transformed = transformer.fit_transform(X)
```

---

### Imbalance Data
- Imbalanced data refers to a situation where the distribution of classes within a dataset is not uniform.
- This is particularly common in classification problems where one class significantly outnumbers the other class.

### Example
```python
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler()
X_rus, y_rus = rus.fit_resample(X_train, y_train)

from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler()
X_ros, y_ros = ros.fit_resample(X_train, y_train)

from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_smote, y_smote = smote.fit_resample(X_train, y_train)
```

---

### Principal Component Analysis
- It reduces the number of dimensions in a large dataset by retaining most of the original information.
- It does so by transforming correlated variables into a smaller set of variables called principal components.

### Example
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

X_train_pca = pca.fit_transform(X_train_scaled)

X_test_pca = pca.transform(X_test_scaled)
```

<hr>

## Getting Started
- Clone this repository to your local machine by using the following command :
 ```bash
 git clone https://github.com/themrityunjaypathak/Feature-Engineering.git
 ```
- Install the Jupyter Notebook :
 ```bash
 pip install notebook
 ```
- Launch the Jupyter Notebook :
 ```bash
 jupyter notebook
 ```
- Open the desired notebook from the repository in your Jupyter environment and start coding.

<hr>

## Contributing
- Contributions are Welcome! 
- If you'd like to contribute to this repository, feel free to submit a pull request.

<hr>

## License
- This repository is licensed under the [MIT License](LICENSE). 
- You are free to use, modify and distribute the code in this repository.

<div align='left'>
  
**[`^        Scroll to Top       ^`](#feature-engineering)**

</div>
