from sklearn.preprocessing import LabelBinarizer

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout

import pandas as pd

# load dataset
# egitim_seti = train_set

# test seti = test_set
# Veri setlerini yukleyelim.
egitim_seti = pd.read_csv('../input/train.csv', index_col=False)
test_seti = pd.read_csv('../input/test.csv', index_col=False)

# Explore Data

# Veriyi arastiralim.
# print egitim_seti.head()

# Concatenate all test.txt and train.txt
# butun_cumleler = all_texts

# Test ve Egitim Setindeki cumleleri birlestirelim.
butun_cumleler = pd.concat([egitim_seti['text'], test_seti['text']])

# size of train_set

# egitim setinin buyuklugu
m = len(egitim_seti)  # 19579

# Set the target and predictor variables for Model Training.
# Modelin Egitimi icin gerekli hedef ve ongorucu degiskenleri ayarlayalim.

# Target Variable = Authors
# Hedef Degiskenimiz = Yazarlar

# Encode authors as binary.
# We encode as binary 'cause we must submit a csv file with the id,
# and a probability for each of the three classes.
# yazarlari binary olarak kodlayalim.
# bunun amaci, submission dosyasi olustururken,
# tahminlerimizi her bir yazara gore olasilik dagilimi istenmesi.
# EAP = 1 0 0
# HPL = 0 1 0
# MWS = 0 0 1
labelbinarizer = LabelBinarizer()
labelbinarizer.fit(egitim_seti['author'])
y = labelbinarizer.fit_transform(egitim_seti['author'])

# Predictor Variable: Sentences
# These are text, we can not use directly.
# Let's extract some features for machine could understand.
# Transforms each text in texts in a sequence of integers.

# Ongorucu Degiskenimiz: Yazarlarin Kurdugu Cumleler
# Bu cumleler, text oldugu icin direkt kullanamayiz.
# Cumlelerden makinenin anlayabilecegi ozellikler cikartalim.
# Yazarlarin kurdugu cumleleri bir dizi tamsayiya donusturelim.
# texts to sequences islemi dogal olarak her bir cumle icin farkli uzunlukta
# bir tam sayi dizisi donecegi icin,
# pad sequences ile her diziyi en uzun tam sayi dizisinin uzunlugunda
# saklamamizi saglar. Boylece her cumle icin cikardigimiz
# ozellik dizisi ayni uzunluktadir.

tokenizer = Tokenizer()
tokenizer.fit_on_texts(butun_cumleler)
X = tokenizer.texts_to_sequences(egitim_seti['text'])
X = pad_sequences(X)

# X_egitim = X_train
# y_egitim = y_train
X_egitim = X
y_egitim = y

# sozluk_boyutu = size of dictionary

# hangi kelimeden kac tane gectigini hesapladigimizda toplam map'in boyutu
# modelimizi olustururken kullanacagiz.
sozluk_boyutu = len(tokenizer.word_index)  # 29451

# X_test

# submission dosyasini olusturmak icin kullanacagimiz test seti
# ayni sekilde test setindeki cumleleri kullanarak her biri icin
# ozellik dizilerini olusturalim.
X_test = tokenizer.texts_to_sequences(test_seti['text'])
X_test = pad_sequences(X_test)

# Create our model
# our model has four layers

# modelimizi olusturalim
# modelimiz dort katmandan olusuyor

model = Sequential()
model.add(Embedding(input_dim=sozluk_boyutu + 1, output_dim=30))
model.add(Dropout(0.5))
model.add(LSTM(30))
model.add(Dense(15, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit(X_egitim, y_egitim, batch_size=32, epochs=5)

model.summary()

# tahminler = predictions

tahminler = model.predict(X_test, batch_size=16)
test_seti['EAP'] = tahminler[:, 0]
test_seti['HPL'] = tahminler[:, 1]
test_seti['MWS'] = tahminler[:, 2]

test_seti[['id', 'EAP', 'HPL', 'MWS']].to_csv('submission.csv', index=False)
