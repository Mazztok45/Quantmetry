# coding: utf-8

# Execution en ligne de commande
import sys
import os
sys.path.append(sys.argv[3])
# Import des librairies numpy, pandas, matplotlib, seaborn, scipy, statsmodels et scikit-learn
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as ss
import statsmodels.api as sm

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Lecture des données avec pandas
data = pd.read_csv(sys.argv[1]+'/'+sys.argv[2], usecols=range(2,13))


# # Partie 1
# ## Exercice 1

# Décompte des valeurs manquantes pour chaque variable
data.isnull().sum()


# On constate que les données ont été récupéréés sur 5 années durant
print(pd.to_datetime(data['date'], infer_datetime_format=True).min(),
pd.to_datetime(data['date'], infer_datetime_format=True).max())


# On décompose la colonne 'date' en trois variables 'year', 'month', 'day'
data['year'] = pd.to_datetime(data['date'], infer_datetime_format=True).dt.year
data['month'] = pd.to_datetime(data['date'], infer_datetime_format=True).dt.month
data['day'] = pd.to_datetime(data['date'], infer_datetime_format=True).dt.day


# On observe que les moyennes et médianes de ces variables sont sensiblement proches
print(data['year'].mean(), data['year'].median())
print(data['month'].mean(), data['month'].median())
print(data['day'].mean(), data['day'].median())


# Graphes d'histogramme des variables 'year', 'month' et 'day' sans leurs valeurs manquantes
plt.figure(figsize=(15,8))
plt.subplot(311)
sns.distplot(data['year'].dropna(), bins =30)
plt.subplot(312)
sns.distplot(data['month'].dropna(), bins =30)
plt.subplot(313)
sns.distplot(data['day'].dropna(), bins =30)
plt.show()


# Remplacement des valeurs manquantes par le mode de chaque série
data['year'] = data['year'].fillna(data['year'].mode()[0])
data['month'] = data['month'].fillna(data['month'].mode()[0])
data['day'] = data['day'].fillna(data['day'].mode()[0])


# Graphes d'histogramme de la variable 'age'
plt.figure(figsize=(15,8))
sns.distplot(data.age.dropna(), bins =30)
plt.show()


# On observe que les moyenne, médiane et mode de la série sont sensiblement proches
print(data.age.dropna().mean(), data.age.dropna().median(), data.age.dropna().mode())


# Remplacement des valeurs manquantes par le mode de la série age
data['age'] = data['age'].fillna(35)


# Graphes d'histogramme des variables 'exp', 'salaire' et 'note' sans leurs valeurs manquantes
plt.figure(figsize=(15,8))
plt.subplot(311)
sns.distplot(data.exp.dropna(), bins =30)
plt.subplot(312)
sns.distplot(data.salaire.dropna(), bins =30)
plt.subplot(313)
sns.distplot(data.note.dropna(), bins =30)
plt.show()


# Remplacement des valeurs manquantes par la moyenne de chaque série
data['exp'] = data['exp'].fillna(data['exp'].mean())
data['salaire'] = data['salaire'].fillna(data['salaire'].mean())
data['note'] = data['note'].fillna(data['note'].mean())


# Décompte des valeurs des variables catégorielles
#print(data.cheveux.value_counts(),data.sexe.value_counts(),data.diplome.value_counts(),data.specialite.value_counts(),data.dispo.value_counts(), sep ="\n")


# ## Exercice 2
# ### Question a

#Création d'un  tableau croisé
crosstab = pd.crosstab(data.specialite, data.sexe, rownames=['specialite'], colnames=['sexe'])
  
#Test de cramer
def cramers(crosstab):
    chi2 = ss.chi2_contingency(crosstab)[0]
    n = crosstab.sum().sum()
    return np.sqrt(chi2 / (n*(min(crosstab.shape)-1)))
  
result = cramers(crosstab)
print(result)


# ### Question b

# On prépare les données en excluant les valeurs manquantes de la variable 'cheveux'
data2 = data[data.index.isin(data['cheveux'].dropna().index)][['cheveux', 'salaire']]
data3 = pd.concat([pd.get_dummies(data2.cheveux), data2.salaire], axis = 1)
data3 = data3.dropna()


# Régression
model = sm.OLS(endog=data3.salaire, exog=data3[['blond','brun','chatain', 'roux']]).fit()
model.summary()


# ### Question c

# Préparation des données
data4 = data[['exp', 'note']].dropna()


# Régression
model = sm.OLS(endog=data4.note, exog=data4.exp).fit()
model.summary()


# # Partie 2
# ## Question 1


# On élimine les colonnes 'date' et 'cheveux'
data = data.drop(['date', 'cheveux'], axis=1)
data.head()

# Préparation des features avec dummification des variables catégorielles
X = pd.concat([data[['age', 'exp', 'salaire','note', 'year', 'month', 'day']],
          pd.get_dummies(data['sexe']),
           pd.get_dummies(data['diplome']),
           pd.get_dummies(data['specialite']),
           pd.get_dummies(data['dispo'])
          ], axis=1)

# Target
y = data['embauche']

# Splittage du set de données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

# Création d'une grille des paramètres
param_grid = {
    'max_features': [5, 10, 15],
    'n_estimators': [50, 100, 150, 200]
}

# Instanciation du modèle de forêts aléatoires utilisant la recherche sur grille de paramètres
rf = RandomForestClassifier()
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3)


# Entraînement du modèle et affichage des meilleurs paramètres
grid_search.fit(X_train, y_train)
grid_search.best_params_


# ## Question 2


# On utilise les meilleurs paramètres du modèles entraîné avec grid search pour paramétrer de nouveau l'algorithme de forêts aléatoire.
# On procède ainsi afin d'utiliser la méthode 'feature_importances'.
forest = RandomForestClassifier(**grid_search.best_params_)
forest.fit(X_train, y_train)


# Affichage des variables par ordre d'importance dans le modèle
dic = {}
for i in range(len(X_train.columns)):
    dic[X_train.columns[i]] = forest.feature_importances_[i]
    
sorted(dic.items(),  key=lambda x: x[1], reverse=True)


# ## Question 3

# Matrice de confusion et calcul des metrics accuracy, precision.
cm = confusion_matrix(y_test, forest.predict(X_test))
tn = cm[0][0]
fp = cm[0][1]
fn = cm[1][0]
tp = cm[1][1]
print('acccuracy: {:.4f} %'.format(forest.score(X_test, y_test)*100))
print('precision: {:.4f} %'.format(tp/(tp+fp)*100))


# ## Question 4


# La metric du recall nous indique que les embauchés sont bien prédis à 33 % ce qui est faible. 
# Cela est du à un déséquilibre de proportion entre les embauchés et non embauchés
print('recall: {:.4f} %'.format(tp/(tp+fn)*100))

