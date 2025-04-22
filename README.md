## EXNO-3-DS
## NAME: DIVYA SRI V
## REGISTER NO: 212224230070
# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
* Log Transformation
* Reciprocal Transformation
* Square Root Transformation
* Square Transformation
  # 2. POWER TRANSFORMATION
* Boxcox method
* Yeojohnson method

# CODING AND OUTPUT:
```
df=pd.read_csv("/content/Encoding Data.csv")
df
```
![image](https://github.com/user-attachments/assets/a0642646-d602-4480-9352-de6d983ac7ac)
```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/user-attachments/assets/58300092-af1e-4204-9187-eb2c18ad17ca)
```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/user-attachments/assets/5770df14-b514-443d-8e07-cd6557079264)
```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/b177e0b2-0556-483a-be09-adbb6c4575c6)
```
# one hot encoding
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/user-attachments/assets/d6f303ef-ca3d-405e-8c16-fe31686dad54)
```
pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/user-attachments/assets/7a6c82df-6970-4779-909a-9fac8d991a14)
```
pip install --upgrade category_encoders
```
```
from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df
```
![image](https://github.com/user-attachments/assets/3e1e331a-a03a-4679-9b6b-f3c66a3d74b9)
```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
```
![image](https://github.com/user-attachments/assets/f9fee27b-fbb4-44a6-a3dc-c3e83fe1db9c)
```
dfb=pd.concat([df,nd],axis=1)
dfb
```
![image](https://github.com/user-attachments/assets/b762ee43-6ade-49e7-8d43-650a16bf2b3f)
```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
![image](https://github.com/user-attachments/assets/d4a9a6a4-6aeb-4262-a5b0-4e769f98a7e0)
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
```
![image](https://github.com/user-attachments/assets/cc69c1aa-1db2-4f0f-87f8-c123301f3c22)
```
df.skew()
```
![image](https://github.com/user-attachments/assets/14fb8731-3446-439f-999e-9872bf202064)
```
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/46716a16-2221-4e8d-92b8-17de8f4386d3)
```
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/user-attachments/assets/7f6d4b41-5203-48de-8066-deb6e961acc4)
```
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/9518b9c3-a009-481e-a47a-dcf270693c9a)
```
np.square(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/7494fc01-bc34-4796-8b6a-a0cf96085d65)
```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/d365e86a-30b2-48fe-a769-344ba24da50f)
```
df.skew()
```
![image](https://github.com/user-attachments/assets/c50c37ea-4ed3-4e8c-b3e4-77c4a1029e99)
```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
![image](https://github.com/user-attachments/assets/9eb3a6c6-dc01-4c23-b0bd-65b6dea743ee)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
![image](https://github.com/user-attachments/assets/dd117971-de43-4e69-87e7-3db147d871e6)
```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/c2383128-5e11-4278-a563-94df084a38c5)
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/c88c7218-2e31-4b9f-a6f0-3d9787f1057b)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/424392e7-8879-49df-9682-148071167fd7)
```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/f1eb4453-6308-40ab-b095-09587a8ef5f5)
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/1a7da441-bb3f-4f8f-89a1-328f76a23eb1)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/cfd31748-7a4b-403c-8084-9e6a273ed266)
```
dt=pd.read_csv("/content/titanic_dataset (1).csv")
dt
```
![image](https://github.com/user-attachments/assets/984aaf6f-0861-4ac4-83ed-4ac4d5c97e52)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/ce31f9f6-b601-4fae-88af-84851b4bde8f)
```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/06ce7848-d9c1-4576-a953-1cc8e41ae305)
## RESULT:
    
   Thus the given data, Feature Encoding, Transformation process and save the data to a file
was performed successfully.

       
