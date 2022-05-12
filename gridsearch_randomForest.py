import pandas as pd

from sklearn import ensemble
from sklearn import model_selection

#Dataframe erstellen, Dummies erstellen, Standartisieren
data_path = "../uf_lab.csv"
df = pd.read_csv(data_path)
df = pd.get_dummies(df, columns=['naturraum', 'geology', 'landuse'])
df.drop(columns=['PBEZ.x', 'T1_Gruppe'], inplace=True)
df = (df - df.mean()) / df.std()
df = df.sample(frac=1.0)

#Auswahl Target und Features
targets = ['sand', 'clay', 'silt']
y = df[targets]
X = df.drop(targets, axis = 1)

classifier = ensemble.RandomForestRegressor(random_state=1, n_jobs=-1)
param_grid = {
       "n_estimators": [80, 100],
       "max_depth": [8, 12, 16],
       #"min_samples_split": [3, 4],
       #"max_features": [10, 14, 20]
}

model = model_selection.GridSearchCV(
       estimator=classifier,
       param_grid=param_grid,
       scoring="explained_variance",
       verbose = 10,
       n_jobs=1,
       cv=5,
       error_score='raise'
)

model.fit(X,y)
print(model.best_score_)
print(model.best_estimator_.get_params())