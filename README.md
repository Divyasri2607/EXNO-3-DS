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

import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df

![image](https://github.com/user-attachments/assets/f30b002e-fbb4-4a37-953e-e22a113344cc)


from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])

![image](https://github.com/user-attachments/assets/b37d6158-a39c-4214-bd3f-52f8f812eefb)


df['bo2']=e1.fit_transform(df[["ord_2"]])
df

![image](https://github.com/user-attachments/assets/171ceed3-26e6-4d5b-8f7b-f9796e20e3a7)


le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc

![image](https://github.com/user-attachments/assets/594ee959-f482-4c99-bdc5-7616d0453d64)


from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df2

![image](https://github.com/user-attachments/assets/80d4529f-d383-46d1-8a3a-d3bb18e814ba)


pd.get_dummies(df2,columns=["nom_0"])

![image](https://github.com/user-attachments/assets/57c359ae-81bd-4d8f-9426-5475b86e7827)


pip install --upgrade category_encoders

![image](https://github.com/user-attachments/assets/0acdad49-bf87-4668-9724-0b08704502d2)


import pandas as pd
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df

![image](https://github.com/user-attachments/assets/3dedba49-b330-40c7-a91a-fbc80aff8bab)


from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC

![image](https://github.com/user-attachments/assets/370002b0-2e1f-4de4-871d-6549c73af164)


import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df

![image](https://github.com/user-attachments/assets/73010ba5-85f7-429a-9dfd-63d830b7546a)


df.skew()

![image](https://github.com/user-attachments/assets/a7ddba59-21ad-4dcc-964a-7a134125c4af)


np.log(df["Highly Positive Skew"])

![image](https://github.com/user-attachments/assets/2767afeb-aa21-4cc7-88b7-f5bd73b611bd)


np.reciprocal(df["Moderate Positive Skew"])

![image](https://github.com/user-attachments/assets/de610002-d6a2-4337-b573-bab15d2bd54e)


np.sqrt(df["Highly Positive Skew"])

![image](https://github.com/user-attachments/assets/a5fc7519-f2f0-4ebb-874e-dd2fb5dacb09)


np.square(df["Highly Positive Skew"])

![image](https://github.com/user-attachments/assets/a8a9730b-e262-42d8-8f38-f14ca42140da)


df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df

![image](https://github.com/user-attachments/assets/bcae818c-932c-4f35-b703-f2eef31bfde3)


df.skew()

![image](https://github.com/user-attachments/assets/4b167f59-171e-4e4b-b933-45456fc841a1)


df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()

![image](https://github.com/user-attachments/assets/dc8e9e80-a780-4d33-a425-44b7930cbe45)


from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df

![image](https://github.com/user-attachments/assets/583fcacd-c9c7-4989-a2c6-53988ade13e1)


import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()

![image](https://github.com/user-attachments/assets/244a8e3c-8939-4d84-a3e1-11ee2f1cd977)


sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()

![image](https://github.com/user-attachments/assets/e7d65e85-43b5-4d65-a0d4-84457997f941)


from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()

![image](https://github.com/user-attachments/assets/07d5af41-e59b-4953-b640-8e971a3054e5)


df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()

![image](https://github.com/user-attachments/assets/2da646c8-4c6b-44ec-b51c-8b3ef31738ac)


dt=pd.read_csv("/content/titanic_dataset (1).csv")
dt

![image](https://github.com/user-attachments/assets/2484364f-8491-4ede-86fb-cca6aa156524)


from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45')
plt.show()

![image](https://github.com/user-attachments/assets/b84f37dd-1e37-432c-b3d1-29667d566906)


sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()

![image](https://github.com/user-attachments/assets/522173b8-201a-4960-8560-be83902cd015)


# RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file
was performed successfully

       
