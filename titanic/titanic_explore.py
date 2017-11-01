# Seray BESER
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

warnings.filterwarnings('ignore')

# veri setini yukleyelim.
titanic_train = pd.read_csv('titanic_train.csv', delimiter=',')
titanic_test = pd.read_csv('titanic_test.csv', delimiter=',')

# veri setinin orjinalini korumak icin kopyalari uzerinde calisalim.
train = titanic_train.copy()
test = titanic_test.copy()

# egitim setindeki passengerID'ye ihtiyacimiz yok, silebiliriz.
del train['PassengerId']

# veri setini arastiralim.
print (train.head())
print (train.info())
print (train.describe())
print (train.corr())

################################################################################
# 1. YONTEM : Sadece Sayisal Veriler ve Eksik Degerleri Manipule Ederek.

# verileri makine ogrenmesi algoritmasina vermek icin ilk olarak
# sadece sayisal verileri kullanarak ve eksik verileri mean ile manipule
# ederek hazirlayalim.
################################################################################

# object tipli ozellikleri silelim.
del train['Name']
del train['Sex']
del train['Ticket']
del train['Cabin']
del train['Embarked']

# eksik yaslari ortalama yas degerleri ile degistirelim.
train['Age'].fillna(train['Age'].mean(), inplace=True)

# yapilan islemleri test seti icin uygulayalim.

# object tipli ozellikleri silelim.
del test['Name']
del test['Sex']
del test['Ticket']
del test['Cabin']
del test['Embarked']

# eksik yaslari ortalama yas degerleri ile degistirelim.
test['Age'].fillna(test['Age'].mean(), inplace=True)
test['Fare'].fillna(test['Fare'].mean(), inplace=True)

# modele verebilmek icin verilerimizi ayarlayalim.

# egitim verileri
y_train = np.reshape(np.asmatrix(train['Survived']), (len(train), -1))
del train['Survived']
X_train = np.reshape(np.asmatrix(train), (len(train), -1))

# test verileri
# submission dosyasi icin ilk passengerId gerekli.
passenger_id = test['PassengerId'][0]
del test['PassengerId']
X_test = np.reshape(np.asmatrix(test), (len(test), -1))

################################################################################
# kendimiz modellerin dogrulugunu olcebilmek icin
# etiketleri belli olan egitim setini egitim-test diye ikiye ayiralim.
################################################################################

trainX, testX, trainY, testY = train_test_split(X_train, y_train, test_size=0.2)

print ('Sadece Sayisal Degerler ve Eksik Deger Manipulasyonu ile\n\t\tModellerin Accuracy Orani')

model = LogisticRegression()
model.fit(X_train, y_train)
X_pred_logistic = model.predict(X_test)
model.fit(trainX, trainY)
pred = model.predict(testX)
print ("LogisticRegression :" + str(round(accuracy_score(testY, pred) * 100, ndigits=3)))
model = Perceptron()
model.fit(X_train, y_train)
X_pred_perceptron = model.predict(X_test)
model.fit(trainX, trainY)
pred = model.predict(testX)
print  ("Perceptron :" + str(round(accuracy_score(testY, pred) * 100, ndigits=3)))

model = SGDClassifier()
model.fit(X_train, y_train)
X_pred_sgdclassifier = model.predict(X_test)
model.fit(trainX, trainY)
pred = model.predict(testX)
print  ("SGDClassifier :" + str(round(accuracy_score(testY, pred) * 100, ndigits=3)))

model = GaussianNB()
model.fit(X_train, y_train)
X_pred_gaussian = model.predict(X_test)
model.fit(trainX, trainY)
pred = model.predict(testX)
print  ("GaussianNB :" + str(round(accuracy_score(testY, pred) * 100, ndigits=3)))

model = KNeighborsClassifier()
model.fit(X_train, y_train)
X_pred_kneighbors = model.predict(X_test)
model.fit(trainX, trainY)
pred = model.predict(testX)
print  ("KNeighborsClassifier :" + str(round(accuracy_score(testY, pred) * 100, ndigits=3)))

model = SVC()
model.fit(X_train, y_train)
X_pred_svc = model.predict(X_test)
model.fit(trainX, trainY)
pred = model.predict(testX)
print  ("SVC :" + str(round(accuracy_score(testY, pred) * 100, ndigits=3)))

model = LinearSVC()
model.fit(X_train, y_train)
X_pred_linearsvc = model.predict(X_test)
model.fit(trainX, trainY)
pred = model.predict(testX)
print  ("LinearSVC :" + str(round(accuracy_score(testY, pred) * 100, ndigits=3)))

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
X_pred_decisiontree = model.predict(X_test)
model.fit(trainX, trainY)
pred = model.predict(testX)
print  ("DecisionTreeClassifier :" + str(round(accuracy_score(testY, pred) * 100, ndigits=3)))

model = GradientBoostingClassifier()
model.fit(X_train, y_train)
X_pred_gradientboosting = model.predict(X_test)
model.fit(trainX, trainY)
pred = model.predict(testX)
print ("GradientBoostingClassifier :" + str(round(accuracy_score(testY, pred) * 100, ndigits=3)))

model = RandomForestClassifier()
model.fit(X_train, y_train)
X_pred_randomforest = model.predict(X_test)
model.fit(trainX, trainY)
pred = model.predict(testX)
print  ("RandomForestClassifier :" + str(round(accuracy_score(testY, pred) * 100, ndigits=3)))

################################################################################
# 2. YONTEM : Sayisal Veriler, Sex, Embarked ve Eksik Degerleri Manipule Ederek

# verileri makine ogrenmesi algoritmasina vermek icin
# sayisal verileri kullanarak, sex ve embarked kategorik degiskenlerini etiketleyerek
# ve eksik verileri mean ile manipule ederek hazirlayalim.
################################################################################

# veri setinin kopyasini olusturalim.
train = titanic_train.copy()
test = titanic_test.copy()

# egitim setindeki passengerID'ye ihtiyacimiz yok
del train['PassengerId']

# kategorik degiskenleri sayisallastiralim.(etiketleme)
label_encoder = LabelEncoder()
train['Sex'] = label_encoder.fit_transform(train['Sex'])
train['Embarked'] = label_encoder.fit_transform(train['Embarked'].fillna('0'))

# geri kalan object tipli ozellikleri silelim.
del train['Name']
del train['Ticket']
del train['Cabin']

# eksik degerleri ortalamalari ile degistirelim.
train['Age'].fillna(train['Age'].mean(), inplace=True)
train['Embarked'].fillna(train['Embarked'].mean(), inplace=True)

# yapilan islemleri test seti icin uygulayalim.

# kategorik degiskenleri sayisallastiralim.(etiketleme)
test['Sex'] = label_encoder.fit_transform(test['Sex'])
test['Embarked'] = label_encoder.fit_transform(test['Embarked'])

# geri kalan object tipli ozellikleri silelim.
del test['Name']
del test['Ticket']
del test['Cabin']

# eksik degerleri ortalamalari ile degistirelim.
test['Age'].fillna(test['Age'].mean(), inplace=True)
test['Fare'].fillna(test['Fare'].mean(), inplace=True)

# modele verebilmek icin verilerimizi ayarlayalim.

# egitim verileri
y_train = np.reshape(np.asmatrix(train['Survived']), (len(train), -1))
del train['Survived']
X_train = np.reshape(np.asmatrix(train), (len(train), -1))

# test verileri
del test['PassengerId']
X_test = np.reshape(np.asmatrix(test), (len(test), -1))

################################################################################
# kendimiz modellerin dogrulugunu olcebilmek icin
# etiketleri belli olan egitim setini egitim-test diye ikiye ayiralim.
################################################################################

trainX, testX, trainY, testY = train_test_split(X_train, y_train, test_size=0.2)
print ()
print ()
print(
    'Sayisal Degerler, Kategorik Ozellikler(sex ve embarked) ve Eksik Deger Manipulasyonu ile Modellerin Accuracy Orani')

model = LogisticRegression()
model.fit(X_train, y_train)
X_pred_logistic_2 = model.predict(X_test)
model.fit(trainX, trainY)
pred = model.predict(testX)
print ("LogisticRegression :" + str(round(accuracy_score(testY, pred) * 100, ndigits=3)))

model = Perceptron()
model.fit(X_train, y_train)
X_pred_perceptron_2 = model.predict(X_test)
model.fit(trainX, trainY)
pred = model.predict(testX)
print  ("Perceptron :" + str(round(accuracy_score(testY, pred) * 100, ndigits=3)))

model = SGDClassifier()
model.fit(X_train, y_train)
X_pred_sgdclassifier_2 = model.predict(X_test)
model.fit(trainX, trainY)
pred = model.predict(testX)
print  ("SGDClassifier :" + str(round(accuracy_score(testY, pred) * 100, ndigits=3)))

model = GaussianNB()
model.fit(X_train, y_train)
X_pred_gaussian_2 = model.predict(X_test)
model.fit(trainX, trainY)
pred = model.predict(testX)
print("GaussianNB :" + str(round(accuracy_score(testY, pred) * 100, ndigits=3)))

model = KNeighborsClassifier()
model.fit(X_train, y_train)
X_pred_kneighbors_2 = model.predict(X_test)
model.fit(trainX, trainY)
pred = model.predict(testX)
print ("KNeighborsClassifier :" + str(round(accuracy_score(testY, pred) * 100, ndigits=3)))

model = SVC()
model.fit(X_train, y_train)
X_pred_svc_2 = model.predict(X_test)
model.fit(trainX, trainY)
pred = model.predict(testX)
print  ("SVC :" + str(round(accuracy_score(testY, pred) * 100, ndigits=3)))

model = LinearSVC()
model.fit(X_train, y_train)
X_pred_linearsvc_2 = model.predict(X_test)
model.fit(trainX, trainY)
pred = model.predict(testX)
print ("LinearSVC :" + str(round(accuracy_score(testY, pred) * 100, ndigits=3)))

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
X_pred_decisiontree_2 = model.predict(X_test)
model.fit(trainX, trainY)
pred = model.predict(testX)
print ("DecisionTreeClassifier :" + str(round(accuracy_score(testY, pred) * 100, ndigits=3)))

model = GradientBoostingClassifier()
model.fit(X_train, y_train)
X_pred_gradientboosting_2 = model.predict(X_test)
model.fit(trainX, trainY)
pred = model.predict(testX)
print ("GradientBoostingClassifier :" + str(round(accuracy_score(testY, pred) * 100, ndigits=3)))

model = RandomForestClassifier()
model.fit(X_train, y_train)
X_pred_randomforest_2 = model.predict(X_test)
model.fit(trainX, trainY)
pred = model.predict(testX)
print ("RandomForestClassifier :" + str(round(accuracy_score(testY, pred) * 100, ndigits=3)))

################################################################################
# 3. YONTEM : Sayisal Veriler, Sex, Embarked, Ticket ve Eksik Degerleri Manipule Ederek

# verileri makine ogrenmesi algoritmasina vermek icin
# sayisal verileri kullanarak, sex ve embarked kategorik degiskenlerini etiketleyerek,
# ticket degerlerini duzenleyerek
# ve eksik verileri mean ile manipule ederek hazirlayalim.
################################################################################

# veri setinin kopyasini olusturalim.
train = titanic_train.copy()
test = titanic_test.copy()

# egitim setindeki passengerID'yi gerek yok.
del train['PassengerId']

# kategorik degiskenleri etiketleyelim.
label_encoder = LabelEncoder()
train['Sex'] = label_encoder.fit_transform(train['Sex'])
train['Embarked'] = label_encoder.fit_transform(train['Embarked'].fillna('0'))

# Ticket degerlerini duzenleyelim.
# Ticket degerinin ilk karakterini alip etiketliyoruz.
train['Ticket'].fillna('N')
train['Ticket'] = train['Ticket'].apply(lambda x: x[0:1] if isinstance(x, str) else '$')
label_encoder = LabelEncoder()
train['Ticket'] = label_encoder.fit_transform(train['Ticket'])

# kullanmayacagimiz verileri silelim.
del train['Name']
del train['Cabin']

# eksik deger duzenlemesi
train['Age'].fillna(train['Age'].mean(), inplace=True)
train['Embarked'].fillna(train['Embarked'].mean(), inplace=True)

# kategorik degiskenleri etiketleyelim.
test['Sex'] = label_encoder.fit_transform(test['Sex'])
test['Embarked'] = label_encoder.fit_transform(test['Embarked'])

# ayni islemleri test veri seti icin uygulayalim.
test['Ticket'].fillna('N')
test['Ticket'] = test['Ticket'].apply(lambda x: x[0:1] if isinstance(x, str) else '$')
le = LabelEncoder()
test['Ticket'] = le.fit_transform(test['Ticket'])

del test['Name']
del test['Cabin']

test['Age'].fillna(test['Age'].mean(), inplace=True)
test['Fare'].fillna(test['Fare'].mean(), inplace=True)

y_train = np.reshape(np.asmatrix(train['Survived']), (len(train), -1))
del train['Survived']
X_train = np.reshape(np.asmatrix(train), (len(train), -1))

del test['PassengerId']
X_test = np.reshape(np.asmatrix(test), (len(test), -1))

################################################################################
# kendimiz modellerin dogrulugunu olcebilmek icin
# etiketleri belli olan egitim setini egitim-test diye ikiye ayiralim.
################################################################################


trainX, testX, trainY, testY = train_test_split(X_train, y_train, test_size=0.2)
print()
print ()
print (
    'Sayisal Degerler, Kategorik Ozellikler(sex ve embarked), Ticket ve Eksik Deger Manipulasyonu ile Modellerin Accuracy Orani')
model = LogisticRegression()
model.fit(X_train, y_train)
X_pred_logistic_3 = model.predict(X_test)
model.fit(trainX, trainY)
pred = model.predict(testX)
print ("LogisticRegression :" + str(round(accuracy_score(testY, pred) * 100, ndigits=3)))

model = Perceptron()
model.fit(X_train, y_train)
X_pred_perceptron_3 = model.predict(X_test)
model.fit(trainX, trainY)
pred = model.predict(testX)
print ("Perceptron :" + str(round(accuracy_score(testY, pred) * 100, ndigits=3)))

model = SGDClassifier()
model.fit(X_train, y_train)
X_pred_sgdclassifier_3 = model.predict(X_test)
model.fit(trainX, trainY)
pred = model.predict(testX)
print ("SGDClassifier :" + str(round(accuracy_score(testY, pred) * 100, ndigits=3)))

model = GaussianNB()
model.fit(X_train, y_train)
X_pred_gaussian_3 = model.predict(X_test)
model.fit(trainX, trainY)
pred = model.predict(testX)
print ("GaussianNB :" + str(round(accuracy_score(testY, pred) * 100, ndigits=3)))

model = KNeighborsClassifier()
model.fit(X_train, y_train)
X_pred_kneighbors_3 = model.predict(X_test)
model.fit(trainX, trainY)
pred = model.predict(testX)
print ("KNeighborsClassifier :" + str(round(accuracy_score(testY, pred) * 100, ndigits=3)))

model = SVC()
model.fit(X_train, y_train)
X_pred_svc_3 = model.predict(X_test)
model.fit(trainX, trainY)
pred = model.predict(testX)
print("SVC :" + str(round(accuracy_score(testY, pred) * 100, ndigits=3)))

model = LinearSVC()
model.fit(X_train, y_train)
X_pred_linearsvc_3 = model.predict(X_test)
model.fit(trainX, trainY)
pred = model.predict(testX)
print ("LinearSVC :" + str(round(accuracy_score(testY, pred) * 100, ndigits=3)))
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
X_pred_decisiontree_3 = model.predict(X_test)
model.fit(trainX, trainY)
pred = model.predict(testX)
print ("DecisionTreeClassifier :" + str(round(accuracy_score(testY, pred) * 100, ndigits=3)))

model = GradientBoostingClassifier()
model.fit(X_train, y_train)
X_pred_gradientboosting_3 = model.predict(X_test)
model.fit(trainX, trainY)
pred = model.predict(testX)
print ("GradientBoostingClassifier :" + str(round(accuracy_score(testY, pred) * 100, ndigits=3)))

model = RandomForestClassifier()
model.fit(X_train, y_train)
X_pred_randomforest_3 = model.predict(X_test)
model.fit(trainX, trainY)
pred = model.predict(testX)
print  ("RandomForestClassifier :" + str(round(accuracy_score(testY, pred) * 100, ndigits=3)))
####################################################################
# sonuclari dosyaya yazma, submission icin
####################################################################
with open('submission.csv', 'w') as f_sonuc:
    f_sonuc.write('PassengerId,Survived\n')
    # hangi algoritmanin tahminlerini yazacaksaniz
    # buraya (X_pred_...) gelmeli.
    for i in X_pred_gradientboosting_3:
        f_sonuc.write((str(passenger_id) + ',' + str(i) + '\n'))
        passenger_id += 1
