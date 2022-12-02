import pandas as pd
dados = pd.read_csv('/content/drive/MyDrive/IA COVID/heart.csv')
dados.head()
nomes_colunas = dados.columns.to_list()
nomes_colunas = nomes_colunas[:len(nomes_colunas)-1]

features = dados[nomes_colunas]
classes = dados['HeartDisease']

from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder

features['Sex'] = encoder.fit_transform(pd.DataFrame(features['Sex']))
features['ChestPainType'] = encoder.fit_transform(pd.DataFrame(features['ChestPainType']))
features['RestingECG'] = encoder.fit_transform(pd.DataFrame(features['RestingECG']))
features['ExerciseAngina'] = encoder.fit_transform(pd.DataFrame(features['ExerciseAngina']))
features['St_Slope'] = encoder.fit_transform(pd.DataFrame(features['St_Slope']))

features = features.fillna(0)

from sklearn.model_selection import train_test_split

features_treino,features_teste,classes_treino,classes_teste = train_test_split(features,
                                                                               classes,
                                                                               test_size=0.4,
                                                                               random_state=2)
from sklearn.ensemble import RandomForestClassifier 
import pandas as pd 

floresta = RandomForestClassifier(n_estimators=1000) #constroi a floresta
#treinar a floresta
floresta.fit(features_treino,classes_treino)

#testar quanto a floresta acerta
predicoes = floresta.predict(features_teste)








