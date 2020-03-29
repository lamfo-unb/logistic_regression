# Oficina Logistic Regression

Objetivo e escopo: Apresentar a teoria de modelo de regressão linear, e apllicar a um exemplo em python sobre classificação do coronavírus (Covid-19). Oficina apresentada em reunião LAMFO em 28 de março de 2020.

Apresentadores:
- Lucas Moreira Gomes
- Alixandro Werneck
- Alícia Macedo

## Teoria - Slides e explicação
Os slides utilizados na apresentação estão disponíveis em:

- https://pt.overleaf.com/project/5e7df851dc340200018380cf

e o post blog está disponível na pasta BLOG deste projeto.

## Códigos

O código utilizado para apresentação da oficina aplicou o processo de aprendizado em um data-set de casos de coronavírus.

### Importando dados

Os dados utilizados para os casos de cornavírus são disponibilizados no Kaggle.

- https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-dataset/version/47

Primeiramente importamos os dados e procuramos por valores ruins ... 
```python
import pandas as pd
import random

data = pd.read_csv("COVID19.csv",usecols=["age","death","gender"])
print(len(data))

print(data) #checando dataframe

print(data.death.unique()) #checando por valores ruins
print(data.age.unique())
print(data.gender.unique())
```
Agora corrigimos os valores onde precisamos

```python
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

```

Em seguida preparamos os dados para serem processados. X são nossas observações sobre o output (variáveis) e Y nosso output. Nesse caso, Y é a informação sobre a morte (1) ou não morte (0) de um paciente. 

```python
y = data["death"] 
X = data.drop("death",axis=1)
```

Separamos os dados em dois: Treino e teste. Aqui, usamos 10% para teste, e 90% para treinamento. 

```python
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)
# print(X_test)
```
Antes de converter nossos dados para um vetor que possa ser analisado pelo modelo, precisamos fazer com que todos os valores de X estejam na mesmo proporção. Para isso, fazemos com que eles fiquem entre -1 e 1, para qualquer variável em X usando a função StandardScaler(). 


```python
from sklearn.preprocessing import StandardScaler 
sc_X = StandardScaler() 
X_train = sc_X.fit_transform(X_train) 
X_test = sc_X.transform(X_test)
```

Agora treinamos o modelo, e predizemos os resultados para os valores de treinamento

```python
from sklearn.linear_model import LogisticRegression 
classifier = LogisticRegression(random_state=0,class_weight = "balanced") 
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
```

Para avaliar a precisão do modelo, utilizamos uma matriz de confusão.

```python
from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_test, y_pred)
print(cm)
```

Por último, plotamos os resulados para conferir visualmente nosso classificador (verde é saudável e vermelhor doente).

```python
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
```

![gráfico][COVID.png]

## Referências

- https://www.marktechpost.com/2019/06/12/logistic-regression-with-a-real-world-example-in-python/
- https://towardsdatascience.com/introduction-to-logistic-regression-66248243c148
- https://dataaspirant.com/2017/03/02/how-logistic-regression-model-works/

