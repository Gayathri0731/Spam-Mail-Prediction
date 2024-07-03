import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

raw_data = pd.read_csv('mail_data.csv')

mail_data = raw_data.where((pd.notnull(raw_data)),'')

### Spam = 0
### Ham  = 1

mail_data.loc[mail_data['Category']=='spam','Category'] = 0
mail_data.loc[mail_data['Category']=='ham','Category'] = 1

x = mail_data['Message']
y = mail_data['Category']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)

feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english',lowercase=True)

x_train_features = feature_extraction.fit_transform(x_train)
x_test_features = feature_extraction.transform(x_test)

y_train = y_train.astype('int')
y_test = y_test.astype('int')

model=LogisticRegression()
model.fit(x_train_features,y_train)

prediction_on_training_data = model.predict(x_train_features)
accuracy_on_training_data = accuracy_score(y_train, prediction_on_training_data)

input_mail = ["SIX chances to win CASH! From 100 to 20,000 pounds txt> CSH11 and send to 87575. Cost 150p/day, 6days, 16+ TsandCs apply Reply HL 4 info"]
input_data_features = feature_extraction.transform(input_mail)

prediction = model.predict(input_data_features)
if (prediction[0] == 1):
  print('GOOD MAIL')
else:
  print('SPAM MAIL')
