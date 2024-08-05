# Description: Bitcoin Prediction Model using Machine Learning in Python
This repository contains historical bitcoin data, the code used to produce the prediction model and report in python, a report of the prediction model and the images used in the report.

Here is a description of what I did to successfully complete this project:<br/>
I downloaded historical data of bitcoin trading from coincodex.com

I followed by importing the following packages to assist me:<br/>
import numpy as np <br/>
import pandas as pd <br/>
import matplotlib.pyplot as plt <br/>
import seaborn as sns <br/>
from sklearn.ensemble import RandomForestRegressor<br/>
from sklearn.linear_model import LinearRegression<br/>
from sklearn.model_selection import train_test_split<br/>
from datetime import datetime as dt<br/>
!pip install python-docx<br/>
from docx import Document<br/>
!pip install dataframe_image<br/>
import dataframe_image as dfi <br/>
!pip install fpdf2<br/>
from fpdf import FPDF

I imported the histrorical data using pandas and viewd the first 5 values:<br/>
df = pd.read_csv('bitcoin.csv')<br/>
df.head()

I wanted to see the growth from start to finish so I had to invert the data to begin at the starting point:<br/>
df = df.sort_values(by=['Start'])<br/>
df.head()

I checked if there was any missing data: <br/>
df.isnull().sum()

There was no missing data so I could proceed. I wanted to create a table that shows the growth of bitcoin which contains the first 5 trading days and the last 5 trading days. I would save this table as an imagine to assist with visualising the growth. However, I would not include the Market Cap values as they were not included in my machine learning algorithm:<br/>
first_rows = df.head(5)<br/>
last_rows = df.tail(5)<br/>
empty_rows = 1<br/>
empty_data = {col: ['...' for _ in range(empty_rows)] for col in df.columns}<br/>
empty_df = pd.DataFrame(empty_data)<br/>
combined_df = pd.concat([first_rows, empty_df, last_rows])<br/>
combined_df = combined_df.reset_index()<br/>
combined_df = combined_df.drop(["index","Market Cap"], axis='columns')<br/>
dfi.export(combined_df, 'dataframe.png')

To better understand the data I obtained a description of it:<br/>
df.describe()

I created a graph to visually illustraate the growth of bitcoin over time and I saved the image to use it in my report:<br/>
df.plot(x='Start',y='Close', figsize=(25,10))<br/>
plt.savefig('close.png')

Next was the machine learning algorithm model that I created using the Random Forest Regressor. My decision trees where the variables, Open, High, Low and Volume which are represented by x and the model we want to predict is Close which is represented by y. For both x and y, I ommited the last row of data in order to predict the final day of trading it and compare it against the actual value of the final day of trading:<br/>
model = RandomForestRegressor()<br/>
x = df[['Open','High','Low','Volume']]<br/>
x = x[:int(len(df)-1)]<br/>
y = df[['Close']]<br/>
y = y[:int(len(df)-1)]<br/>
model.fit(x,y)

Once the model was fit, I chekced the model score:<br/>
predictions = model.predict(x)<br/>
print("The model score is:", round(model.score(x,y),5))

I then predicted the closing value of the final day of trading and put it against the actual value of the final day of trading:<br/>
new_data = df[['Open','High','Low','Volume']].tail(1)<br/>
prediction = model.predict(new_data)<br/>
print('The model predicts the value for the last day is:', prediction)<br/>
print('The actual value for the last day is', df[['Close']].tail(1).values[0][0])

Went further and predicted the closing value for all the trading days:<br/>
X = df[['Open','High','Low','Volume']]<br/>
Y = df[['Close']]<br/>
close_predictions = model.predict(X)<br/>
close_predictions

I created and added a column with the predicted close values to the original dataframe:<br/>
df.insert(6, "Predicted Close", close_predictions, True)<br/>
df.head()

I wanted to create a table that shows the growth of bitcoin including the predicted values which contains the first 5 trading days and the last 5 trading days. I would save this table as an imagine to assist with visualising the growth. However, I would not include the Market Cap values as they were not included in my machine learning algorithm:<br/>
first_rows_1 = df.head(5)<br/>
last_rows_1 = df.tail(5)<br/>
empty_rows = 1<br/>
empty_data = {col: ['...' for _ in range(empty_rows)] for col in df.columns}<br/>
empty_df = pd.DataFrame(empty_data)<br/>
combined_df_1 = pd.concat([first_rows_1, empty_df, last_rows_1])<br/>
combined_df_1 = combined_df_1.reset_index()<br/>
combined_df_1 = combined_df_1.drop(["index","Market Cap"], axis='columns')<br/>
dfi.export(combined_df_1, 'dataframe1.png')

I created a graph to visually illustraate the growth of bitcoin over time including the predicted values and I saved the image to use it in my report:<br/>
df.plot(x='Start',y=['Close','Predicted Close'], figsize=(25,10))<br/>
plt.savefig('predicted close.png')

Lastly I created my report using python:<br/>
pdf.write(8,'Bitcoin Trading Data')<br/>
pdf.image('dataframe.png', x=10,y=45,w=180,h=80)<br/>
pdf.ln(100)

pdf.write(8,'Bitcoin Growth Over Time')
pdf.image('close.png', x=1,y=140,w=200,h=120)<br/>
pdf.ln(140)

pdf.write(8,'After using the random forest regressor to combine multiple decision trees to create one model, the prediction score for this model is 0.99994.' + '\n' + 'When a model has a prediction score of 1, it is considered perfect.' )<br/>
pdf.ln(20)

pdf.write(8,'Bitcoin Trading Data including Predictions')<br/>
pdf.image('dataframe1.png', x=10,y=70,w=180,h=80)<br/>
pdf.ln(100)

pdf.write(8,'Bitcoin Growth Over Time with Predictions')<br/>
pdf.image('predicted close.png', x=1,y=160,w=200,h=120)<br/>
pdf.ln(120)

pdf.write(8,'Our predictive model for Bitcoin has surpassed expectations, achieving an extraordinary accuracy score of 0.99994. This near-perfect alignment with observed data underscores its reliability and robustness.')<br/>
pdf.ln(15)

pdf.write(8, "While celebrating this remarkable accuracy, it is essential to remain vigilant. Continuous monitoring and validation are crucial to ensure consistent performance across varying market conditions. Additionally, consider stress-testing the model to assess its resilience during extreme events.")<br/>
pdf.ln(15)

pdf.write(8, 'In summary, our model provides invaluable insights for cryptocurrency traders, investors, and decision-makers. Its precision opens doors to informed strategies and risk management. As we refine and enhance its capabilities, we anticipate even greater value in navigating the dynamic world of Bitcoin.')

pdf.output('Bitcoin Prediction Model using Machine Learning in Python.pdf','F')
