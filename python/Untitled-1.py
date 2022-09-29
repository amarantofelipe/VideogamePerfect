from asyncore import read
from pandas import read_csv
from sklearn.tree import DecisionTreeClassifier

giocatori = read_csv ('giocatori.csv')
x = giocatori.drop(columns= ["videogame"])
y = giocatori.drop(columns= ["genere", "anni"])

decisione = DecisionTreeClassifier()
decisione.fit(x.values, y.values)
previsione = decisione.predict([[1, 20]])
print(previsione)