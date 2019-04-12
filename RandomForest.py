import csv
from sklearn.ensemble import RandomForestClassifier
import numpy as np

Data1 = []
Classes = []

with open("IrisTrainData.csv") as csvFile:
	reader = csv.reader(csvFile)
	for row in reader:
		Data1.append(list(row))
with open("IrisTestData.csv") as csvFile:
	reader = csv.reader(csvFile)
	for row in reader:
		Data1.append(list(row))

k = 5 # for k cross validation

confusionMatrix = [["-","Iris-setosa","Iris-versicolor","Iris-virginica","Total"],["Iris-setosa    ",0,0,0,0],["Iris-versicolor",0,0,0,0],["Iris-virginica ",0,0,0,0],["Total          ",0,0,0,0]]

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

	RFclf = RandomForestClassifier(max_depth=4, random_state=0, n_estimators=100)
	RFclf.fit(DataTrain,DataTrainRes)

	for i in range(len(DataTest)):
		RFres = RFclf.predict(DataTest[i])
		if RFres[0] == DataTestRes[i]:
			successCount += 1
		confusionMatrix[DataTestRes[i]+1][RFres[0]+1] += 1
		confusionMatrix[-1][RFres[0]+1] += 1
		confusionMatrix[DataTestRes[i]+1][-1] += 1		
		confusionMatrix[-1][-1] += 1

Result = np.matrix(confusionMatrix)
print "\n\n"
print Result

print "\n\nPercent result:",float(successCount)/151
