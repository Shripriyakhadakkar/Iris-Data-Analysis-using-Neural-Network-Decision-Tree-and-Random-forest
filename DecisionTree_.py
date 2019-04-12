import csv
from sklearn import tree
import numpy as np

Data1 = []
Classes = []

with open("IrisTrainData.csv") as csvFile:
	reader = csv.reader(csvFile)
	for row in reader:
		Data1.append(list(row))

k = 5 # for k cross validation

successCount = 0

for i in range(1,k+1):
	DataTrain = []
	DataTest = []
	DataTrainRes = []
	DataTestRes = []
	
	for j in range(1,len(Data1)+1):
		if j%5-i%5 == 0:
			DataTest.append(map(float,Data1[j-1][:-1]))
			DataTestRes.append(int(Data1[j-1][-1]))
		else:
			DataTrain.append(map(float,Data1[j-1][:-1]))
			DataTrainRes.append(int(Data1[j-1][-1]))

	DTclf = tree.DecisionTreeClassifier()
	DTclf.fit(DataTrain,DataTrainRes)

	for i in range(len(DataTest)):
		DTres = DTclf.predict(DataTest[i])
		if DTres[0] == DataTestRes[i]:
			successCount += 1


fold_cross_validation_accuracy = float(successCount)/len(Data1)



Data2 = []

with open("IrisTestData.csv") as csvFile:
	reader = csv.reader(csvFile)
	for row in reader:
		Data2.append(list(row))

confusionMatrix = [["-","Iris-setosa","Iris-versicolor","Iris-virginica","Total"],["Iris-setosa    ",0,0,0,0],["Iris-versicolor",0,0,0,0],["Iris-virginica ",0,0,0,0],["Total          ",0,0,0,0]]

successCount = 0

DataTrain = []
DataTest = []
DataTrainRes = []
DataTestRes = []

for j in range(1,len(Data1)+1):
	DataTrain.append(map(float,Data1[j-1][:-1]))
	DataTrainRes.append(int(Data1[j-1][-1]))

for j in range(1,len(Data2)+1):
	DataTest.append(map(float,Data2[j-1][:-1]))
	DataTestRes.append(int(Data2[j-1][-1]))

DTclf = tree.DecisionTreeClassifier()
DTclf.fit(DataTrain,DataTrainRes)

for i in range(len(DataTest)):
	DTres = DTclf.predict(DataTest[i])
	if DTres[0] == DataTestRes[i]:
		successCount += 1
	confusionMatrix[DataTestRes[i]+1][DTres[0]+1] += 1
	confusionMatrix[-1][DTres[0]+1] += 1
	confusionMatrix[DataTestRes[i]+1][-1] += 1		
	confusionMatrix[-1][-1] += 1

Result = np.matrix(confusionMatrix)

print "\n\n5-fold cross validation accuracy:",fold_cross_validation_accuracy
print "\n\n"
print Result

print "\n\nAccuracy:",float(successCount)/len(Data2)
