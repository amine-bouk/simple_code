import csv 
iris_data=[]
iris_classe=[]

#convertir la csv data
with open ("iris.csv") as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=",")
	for row in csv_reader:
		iris_data.append([float(row[0]),float(row[1]),float(row[2]),float(row[3])])
		iris_classe.append(row[4])

from sklearn.neural_network import MLPClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing 

#définir l'architecture du réseau
clf=MLPClassifier(solver="lbfgs",alpha=1e-5,hidden_layer_sizes=(6,2),random_state=1)

X_train, X_test, y_train, y_test= train_test_split(iris_data,iris_classe,test_size=0.33 ,random_state=42)

scaler= preprocessing.StandardScaler().fit(X_train)

X_train=scaler.transform(X_train)
X_test = scaler.transform(X_test)
clf.fit(X_train,y_train)

y_pred= clf.predict(X_test)
print("le score est de :"+str(accuracy_score(y_test,y_pred))+"%")
