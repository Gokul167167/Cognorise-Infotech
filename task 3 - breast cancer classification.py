# importing dependencies
import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score

warnings.filterwarnings("ignore",category=UserWarning)

# reading the dataset into a dataframe using pandas
data=pd.read_csv('data.csv')

# analysing through the dataset to preprocess
print(data.head())
print(data.shape)
print(data.describe)
print(data.columns)
print(data.isnull().sum())
print(data.dtypes)

# converting object type into numerical value using label encoder
le=LabelEncoder()
data['diagnosis']=le.fit_transform(data['diagnosis'])
data['compactness_mean']=le.fit_transform(data['compactness_mean'])
print(data.dtypes)

# declaring dependent and independent variables from the dataset as x and y
x=data[['radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']]
y=data['diagnosis']

# splitting the dataset into 4 arrays as training and testing data from x and y
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)

# training the model using random forest classifier
mod=RandomForestClassifier(n_estimators=10,random_state=42)
mod.fit(x_train,y_train)

# making predictions on training data and checking its accuracy and classification report
y_pred1=mod.predict(x_train)
acc=accuracy_score(y_train,y_pred1)
cls=classification_report(y_train,y_pred1)
print("The accuracy on training data:",acc)
print("The classification report on training data:\n",cls)

# making predictions on testing data and checking its accuracy and classifiction report
y_pred2=mod.predict(x_test)
acc2=accuracy_score(y_test,y_pred2)
cls2=classification_report(y_test,y_pred2)
print("The accuracy on testing data:",acc2)
print("The classifiction report on testing data:\n",cls2)

# predicting whether a person has breast cancer by giving an input
input_data=[['0.3001','0.1471','0.2419','0.07871','1.095','0.915','10.38','122.8','1001','0.1184','0.2776','0.53','8.589','153.4','0.006399','0.04904','0.05373','0.01587','0.03003','0.006193','25.38','17.33','184.6','2019','0.1622','0.6656','0.7119','0.2654','0.4601','0.1189']]
prediction=mod.predict(input_data)
if prediction[0]==0:
    print('''The person doesn't have breast cancer''')
else:
    print("The person has breast cancer")