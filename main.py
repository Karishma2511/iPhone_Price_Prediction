import pandas
import matplotlib.pyplot as mat
from sklearn.linear_model import LinearRegression
data = pandas.read_csv('IphonePrice.csv')
mat.scatter(data['version'],data['price'])
mat.show()
model = LinearRegression()
model.fit(data[['version']],data[['price']])
print("VERSION 13: ",model.predict([[13]]))
print("VERSION 14: ",model.predict([[14]]))
print("VERSION 15: ",model.predict([[15]]))
print("VERSION 16: ",model.predict([[16]]))
print("VERSION 20: ",model.predict([[20]]))


