from sklearn.preprocessing import LabelBinarizer

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout

import pandas as pd

# Veri setlerini yukleyelim.
egitim_seti = pd.read_csv('spooky_train.csv', index_col=False)
test_seti = pd.read_csv('spooky_test.csv', index_col=False)

# Veriyi arastiralim.
# print egitim_seti.head()

# Test ve Egitim Setindeki cumleleri birlestirelim.
butun_cumleler = pd.concat([egitim_seti['text'], test_seti['text']])

# egitim setinin buyuklugu
m = len(egitim_seti)  # 19579

# Modelin Egitimi icin gerekli hedef ve ongorucu degiskenleri ayarlayalim.

# Hedef Degiskenimiz = Yazarlar
# yazarlari binary olarak kodlayalim.
# bunun amaci, submission dosyasi olustururken,
# tahminlerimizi her bir yazara gore olasilik dagilimi istenmesi.
# EAP = 1 0 0
# HPL = 0 1 0
# MWS = 0 0 1
labelbinarizer = LabelBinarizer()
labelbinarizer.fit(egitim_seti['author'])
y = labelbinarizer.fit_transform(egitim_seti['author'])

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

X_egitim = X
y_egitim = y

# hangi kelimeden kac tane gectigini hesapladigimizda toplam map'in boyutu
# modelimizi olustururken kullanacagiz.
sozluk_boyutu = len(tokenizer.word_index)  # 29451

# submission dosyasini olusturmak icin kullanacagimiz test seti
# ayni sekilde test setindeki cumleleri kullanarak her biri icin
# ozellik dizilerini olusturalim.
X_test = tokenizer.texts_to_sequences(test_seti['text'])
X_test = pad_sequences(X_test)

# modelimizi olusturalim
# modelimiz dort katmandan olusuyor
# Sequence classification with LSTM:

model = Sequential()
model.add(Embedding(input_dim=sozluk_boyutu + 1, output_dim=30))
model.add(Dropout(0.5))
model.add(LSTM(30))
model.add(Dense(15, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit(X_egitim, y_egitim, batch_size=32, epochs=5)

model.summary()

tahminler = model.predict(X_test, batch_size=16)
test_seti['EAP'] = tahminler[:, 0]
test_seti['HPL'] = tahminler[:, 1]
test_seti['MWS'] = tahminler[:, 2]

test_seti[['id', 'EAP', 'HPL', 'MWS']].to_csv('submission.csv', index=False)
