# importing necessary packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# loading the data set from a file into a data framework using pandas
raw_data =pd.read_csv('spam.csv')

# removing null values in the data set
pre_data = raw_data.where((pd.notnull(raw_data)),'')

# labelling spam as 0 and ham as 1 in the Category column of the dataset
pre_data.loc[pre_data['Category'] == 'spam', 'Category',] = 0
pre_data.loc[pre_data['Category'] == 'ham', 'Category',] = 1

# declaring the Message column as x and Category column as y in the dataset
x = pre_data['Message']
y = pre_data['Category']

# splitting the dataset into four arrays namely from x and y as training data and testing data
x_train,x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)

# transform the text data to vectors that can be used as input to the Logistic regression
feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase=True)
x_train_features = feature_extraction.fit_transform(x_train)
x_test_features = feature_extraction.transform(x_test)

# converting y_train and y_test into int datatype
y_train = y_train.astype('int')
y_test = y_test.astype('int')

# training the model using logistic regression
model = LogisticRegression()
model.fit(x_train_features, y_train)

# prediction and accuracy test of training data
prediction_on_training = model.predict(x_train_features)
accuracy_on_training = accuracy_score(y_train, prediction_on_training)
print('Accuracy on training data : ', accuracy_on_training)

# prediction and accuracy test of testing data
prediction_on_test_data = model.predict(x_test_features)
accuracy_on_test_data = accuracy_score(y_test, prediction_on_test_data)
print('Accuracy on test data : ', accuracy_on_test_data)

# passing a mail to check whether the mail is spam or ham
input_mail=["Please call our customer service representative on 0800 169 6031 between 10am-9pm as you have WON a guaranteed £1000 cash or £5000 prize!"]

# transforming the mail to feature vectors
input_data_features = feature_extraction.transform(input_mail)

# prediction of input mail
prediction = model.predict(input_data_features)
if (prediction[0]==0):
  print('Spam mail')
else:
  print('Ham mail')