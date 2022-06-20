import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv('https://raw.githubusercontent.com/MurwanjaniTejoRiyono/colab/main/DATASET/Crop_recommendation.csv')
df = df.drop('rainfall', 1)
df.head(5)

df['label'] = df['label'].replace(['rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas',
                                   'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate',
                                   'banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 'apple',
                                   'orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee'],
                                  ['Padi', 'Jagung', 'Kacang Arab', 'Kacang Merah', 'Kacang Gude',
                                   'Kacang Ngengat', 'Kacang Hijau', 'Lentil Hitam', 'Lentil', 'Delima',
                                   'Pisang', 'Mangga', 'Anggur', 'Semangka', 'Melon', 'Apel',
                                   'Jeruk', 'Pepaya', 'Kelapa', 'Tanaman Kapas', 'Rami', 'Kopi'])

features = df[['N', 'P','K','temperature', 'humidity', 'ph']]
target = df['label']

x_train, x_test, y_train, y_test = train_test_split(features,target,test_size = 0.2,random_state =2)

RF = RandomForestClassifier()
RF.fit(x_train,y_train)

pickle.dump(RF, open("RF.pkl", "wb"))