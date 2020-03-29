import pandas as pd
import random

data = pd.read_csv("COVID19.csv",usecols=["age","death","gender"])
print(len(data))

print(data) #checando dataframe

print(data.death.unique()) #checando por valores ruins
print(data.age.unique())
print(data.gender.unique())

data = data.dropna(subset=['gender']) #tirando linhas que não tenham informação sobre o gênero ..

def gender(x): # função para convertendo as variável categórica para valor numérico
    if x == "male":
        return 0
    else:
        return 1

data['gender'] = data['gender'].apply(lambda x: gender(x)) #aplica a função gender no dataframe data coluna gender
print(data.gender.unique())

data = data[(data["death"] == "1" ) | (data["death"] == "0" )] # deixamos apenas os resultados para 1 e 0.

print(data.death.unique())


print(data.mean())
data = data.fillna(data.mean()) #completamos valores não disponíveis pela média de cada coluna.

y = data["death"] 
X = data.drop("death",axis=1)


from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)
# print(X_test)

from sklearn.preprocessing import StandardScaler 
sc_X = StandardScaler() 
X_train = sc_X.fit_transform(X_train) 
X_test = sc_X.transform(X_test)
# print(X_test)

from sklearn.linear_model import LogisticRegression 
classifier = LogisticRegression(random_state=0,class_weight = "balanced") 
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# print(y_pred)

from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_test, y_pred)
print(cm)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap 
X_set, y_set = X_train, y_train 
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01), np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01)) 
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('green', 'red')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max()) 
for i, j in enumerate(np.unique(y_set)): 
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('green', 'red'))(i), label = j) 
plt.title('(COVID) Logistic Regression (Training set)') 
plt.xlabel('Sexo') 
plt.ylabel('Idade') 
plt.legend() 
plt.show()
