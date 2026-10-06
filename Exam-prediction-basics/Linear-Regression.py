from sklearn.linear_model import LinearRegression

X = [[1], [2], [3], [4], [5], [6]]
y = [35, 43, 52, 61, 72, 81]

model = LinearRegression()

model.fit(X, y)

prediction = model.predict([[7]])

print(prediction)
