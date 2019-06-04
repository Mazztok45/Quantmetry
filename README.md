# Quantmetry - Test technique

## Prérequis
La version de Python utilisée est la 3.6.3. 
On utilise donc les librairies numpy, pandas, matplotlib, seaborn, scipy, statsmodels et scikit-learn compatibles avec cette version.

## Execution
L'utilisateur a la possibilité d'éxecuter le fichier IPythonNotebook (.ipynb) avec Jupyter Notebook, que l'on retrouve notamment sur la distribution Anaconda.
Il a aussi la possibilité d'éxécuter le fichier Python (.py) en ligne de commande moyennant le renseignement de certains paramètres:

_arg0: Le nom du script
_arg1: Le chemin pour accéder au répertoire contenant le fichier de données
_arg2: le nom du fichier de données
_arg3: le chemin d'accès pour l'import des librairies (on part du principe que les librairies se situent dans un même répertoire)

```
python arg0 arg1 arg2 arg3
```

Exemple:

```
python C:/Users/maxen/Desktop/Quantmetry/Quantmetry.py C:/Users/maxen/Desktop/Quantmetry data_v1.0.csv C:/Users/maxen/Anaconda3/lib/site-packages/
```

## Auteur
Maxence AZZOUZ-THUDEROZ