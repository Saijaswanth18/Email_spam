import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

df=pd.read_csv('mail_data.csv')
df

data=df.where((pd.notnull(df)),'')
data

data.info()

data.shape

from pandas.core.indexes import category
data.loc[data['Category']=='spam','Category']=0
data.loc[data['Category']=='ham','Category']=1

X=data['Message']
Y=data['Category']

X

Y

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=3)

print(X.shape)
print(X_train.shape)
print(X_test.shape)
print(Y.shape)
print(Y_train.shape)
print(Y_test.shape)

feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

print(X_train)

print(X_train_features)

model=LogisticRegression()
model.fit(X_train_features,Y_train)

prediction_train=model.predict(X_train_features)
accuracy_train=accuracy_score(Y_train,prediction_train)
print("ACC on training data:",accuracy_train)

prediction_test=model.predict(X_test_features)
accuracy_test=accuracy_score(Y_test,prediction_test)
print("ACC on testing data:",accuracy_test)

input_mail=["""Subject: CONGRATULATIONS!!! You’ve WON $1,000,000 – Claim NOW!!!

Dear Lucky Winner,

We are thrilled to inform you that your email address has been randomly selected as the GRAND PRIZE WINNER of ONE MILLION DOLLARS!!! 💰💰💰

To claim your prize:
- Click this exclusive link: http://fake-prize-claim-now.com
- Enter your personal details (name, address, bank account, password)
- Act FAST – this offer expires in 24 HOURS!!!

Failure to respond immediately will result in forfeiting your winnings. Don’t miss this once-in-a-lifetime opportunity!!!

Sincerely,
International Lottery Promotions Team  """]

input_data_feature=feature_extraction.transform(input_mail)
prediction=model.predict(input_data_feature)

print(prediction)
if(prediction[0]==1):
    print("HAM MAIL")
else:
    print("SPAM MAIL")
