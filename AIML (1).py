# IMPORT LIBRARIES
import pandas as pd
from sklearn.linear_model import LinearRegression

# ENTERING DATA
details = {
    'space':[100, 160, 200, 250, 350, 800, 1000],
    'bedrooms':[1, 2, 2, 3, 4, 4, 5],
    'timespan':[10, 8, 7, 6, 5, 4, 2],
    'rate':[7, 12, 16, 22, 30, 45, 60]
}

# DATA IN TABULAR FORM
df = pd.DataFrame(details)

# INPUT (X) AND OUTPUT (Y)
X = df[['space','bedrooms','timespan']]
Y = df['rate']

# MODEL CREATION
# FORMULA USED : rate = w1.space + w2.bedrooms + w3.timespan + b
model = LinearRegression()

# TRAINING FUNCTION
model.fit(X,Y)

# LEARNED VALUES

# VALUE OF w1, w2, w3
print(model.coef_) 

# VALUE OF b
print(model.intercept_)

# INPUTS FROM USERS
space = int(input("Space of the house:"))
bedrooms = int(input("Number of bedrooms needed:"))
timespan = int(input("Enter the time span of house:"))

# RATE PREDICTION
a = model.predict([[space,bedrooms,timespan]])

# FINAL RATE
print("Price of the house:",round(a[0],4),"lakhs")