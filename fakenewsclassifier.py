import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import tensorflow
from tensorflow import keras
from keras.layers import Embedding,Dense,LSTM,Dropout
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot
from keras.models import Sequential
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

data=pd.read_csv("train.csv")
data=data.dropna()
data.head()

x=data.drop('label',axis=1)
y=data['label']

message=x.copy()
message.reset_index(inplace=True)
nltk.download('stopwords')

ps=PorterStemmer()
corpus=[]
for i in range(len(message)):
  review=re.sub('[^a-zA-Z]',' ',message['title'][i])
  review=review.lower()
  review=review.split()

  review=[ps.stem(word) for word in review if not word in stopwords.words('english')]
  review=' '.join(review)
  corpus.append(review)

voc_size=50000
one_hotr=[one_hot(words,voc_size) for words in corpus]

sent_length=20
embedded_docs=pad_sequences(one_hotr,sent_length,padding='pre')

dimenson=40
model=Sequential()
model.add(Embedding(voc_size,dimenson,input_length=sent_length))
model.add(Dropout(0.3))
model.add(LSTM(150))
model.add(Dropout(0.3))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

final_x=np.array(embedded_docs)
final_y=np.array(y)

x_train, x_test, y_train, y_test = train_test_split(final_x,final_y, test_size=0.33, random_state=42)

model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=30,batch_size=65)

y_pred=model.predict_classes(x_test)

val=metrics.accuracy_score(y_test,y_pred)
print("accuracy score=",str(val*100)+" %")










