import numpy as np
import pandas as pd
import pickle
#import matplotlib.pyplot as plt 
dataset = pd.read_csv("C:\\Users\\Donatus\\Documents\\WalmartDeploy\\WalmartSales.csv")
dataset.isnull().sum()

#dataset['rate'].fillna(0, inplace=True)

#dataset['sales_in_first_month'].fillna(dataset['sales_in_first_month'].mean(), inplace=True)

#X = dataset.iloc[:, :3]

#date time
df=dataset.copy()
pd.to_datetime(df['Date'])

df = df.drop('Date', 1)
#convert to numericals
df['IsHoliday']=df['IsHoliday'].apply(lambda x: 0 if x== 'FALSE' else 0)
convert=False

X = df[['Store', 'Dept', 'IsHoliday']] 

y = df['Weekly_Sales']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=.25, random_state=42)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
#Saving the model as pickle file
pickle.dump(regressor, open('savmodel.pkl','wb'))
model = pickle.load(open('savmodel.pkl','rb'))
#print(model.predict([[4, 300, 500]]))

preds = model.predict(x_test)

from sklearn.metrics import r2_score

print(f'Your model scores {r2_score(preds, y_test)} on test set')

# Will improve my model score with the below tutorial
#https://github.com/RAHULKASHYAP02/WALMART-STORE-SALES-PREDICTION-FORECASTING

