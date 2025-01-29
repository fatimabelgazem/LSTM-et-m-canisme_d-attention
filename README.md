# Projet de Prévision de l'Énergie Photovoltaïque avec LSTM et mécanisme d'attention
---

Ce projet vise à prédire la production d'énergie photovoltaïque en utilisant des modèles LSTM (Long Short-Term Memory) et des mécanismes d'attention. Les données utilisées proviennent de fichiers CSV contenant des informations sur la production d'énergie de 2010 à 2017.

---

## Table des matières
1. [Description](#description)
2. [Prérequis](#prérequis)
3. [Fonctionnement](#Fonctionnement)
4. [Utilisation](#utilisation)
5. [Visualisation](#Visualisation)
6. [Résultats](#Résultats)
7. [Auteurs](#Auteurs)

---

## Description

Le projet utilise des données sur la production d'énergie photovoltaïque pour entraîner un modèle de prédiction basé sur un réseau de neurones LSTM. Ensuite, un mécanisme d'attention est ajouté pour améliorer la précision de la prédiction.

---

## Prérequis

Pour exécuter ce projet, vous devez avoir les bibliothèques suivantes installées :

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `tensorflow` et `keras`

Vous pouvez installer ces bibliothèques avec pip :

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras
```
---

## Fonctionnement
  - Préparation des données :

Les données sont extraites depuis des fichiers CSV et prétraitées (suppression des lignes inutiles, correction des valeurs, extraction des informations sur la date, etc.).
Les données sont ensuite normalisées à l'aide de MinMaxScaler pour les rendre compatibles avec le modèle.

  - Séparation des données :

Les données sont divisées en ensembles d'entraînement et de test (80% pour l'entraînement et 20% pour le test).
Les ensembles d'entraînement et de test sont ensuite transformés en séries temporelles de 7 jours pour être compatibles avec le modèle LSTM.

  - Création du modèle :

Un modèle LSTM est créé avec deux couches LSTM et des couches de Dropout pour éviter le surapprentissage.
Ensuite, un mécanisme d'attention est ajouté au modèle pour améliorer les prédictions en pondérant les informations importantes.
Entraînement et évaluation :

Le modèle est entraîné pendant 50 à 130 époques, avec un taux d'apprentissage adapté, et évalué à l'aide de la perte MSE (Mean Squared Error) et du score R2.

---

## Utilisation

**1. Charger les données**

Les données sont automatiquement téléchargées et prétraitées depuis les fichiers CSV de 2010 à 2017.

```bash
Data_10 = pd.read_csv('https://ws.50hertz.com/web02/api/PhotovoltaicActual/DownloadFile?fileName=2010.csv', sep=';')
Data_10 = reg_data(Data_10)
```

**2. Entraîner le modèle LSTM**

Le modèle LSTM est entraîné avec les données d'entraînement transformées.

```bash
model_train = model.fit(Xtrain, Ytrain, validation_split=0.2, epochs=50, batch_size=8, verbose=1)
```

**3. Ajouter le mécanisme d'attention**

Une couche d'attention est ajoutée pour pondérer les informations importantes au cours de l'entraînement.

```bash
model2 = Model(inputs=x, outputs=outputs)
model2.compile(loss='mse', optimizer=Adam(learning_rate=1e-3))
model_train2 = model2.fit(Xtrain, Ytrain, validation_split=0.2, epochs=130, batch_size=256, verbose=1)

```

**4. Prédire et évaluer**

Après l'entraînement, vous pouvez faire des prédictions et évaluer les performances du modèle.

```bash
testPredict3 = model2.predict(Xtest)
mean_squared_error(Ytest, testPredict3)
r2 = r2_score(Ytest, testPredict3)
```

---

## Visualisation 

Les graphiques de la perte du modèle pendant l'entraînement peuvent être générés pour visualiser la performance du modèle.

```bash
def ploter_Erreur(hist):
    f, axarr = plt.subplots(1, 1, figsize=(20, 5))
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    axarr.set_title('Model loss')
    axarr.set_ylabel('Loss')
    axarr.set_xlabel('Epoch')
    axarr.legend(['Entrainement', 'validation'], loc='upper left')
    plt.show()
```
---

## Résultats

Les performances du modèle sont mesurées par la MSE (Mean Squared Error) et le coefficient de détermination R2.

```bash
Train set MSE = X.XX
Test set MSE = X.XX
R2 Score = X.XX
```
---

## Conclusion

- Ce modèle montre comment utiliser les réseaux LSTM et le mécanisme d'attention pour prédire la production d'énergie photovoltaïque à partir de données historiques. Il peut être amélioré avec des ajustements d'hyperparamètres et une meilleure sélection des features.
--- 

## Auteurs

 - Fatima Belgazem : fatimabelgazem@gmail.com
