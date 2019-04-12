
import matplotlib.pyplot as plt
import csv
import numpy as np

def equivalentNumber(name):
	if name=='Iris-setosa':
		return 0;
	if name=='Iris-versicolor':
		return 1;
	if name=='Iris-virginica':
		return 2;

Data1 = []
Head = []

with open("Iris.csv") as csvFile:
	reader = csv.reader(csvFile)
	count = 0
	for row in reader:
		if count==0:
			Head.append(row)
			count = 1
		else:
			Data1.append(list(row))

Data = np.matrix(Data1)


print "There are four features in Dataset of numeric type",Head

SepalLength = [float(x[0]) for x in Data[:,1].tolist()]
SepalWidth = [float(x[0]) for x in Data[:,2].tolist()]
PetalLength = [float(x[0]) for x in Data[:,3].tolist()]
PetalWidth = [float(x[0]) for x in Data[:,4].tolist()]

Classes = [equivalentNumber(x[0]) for x in Data[:,5].tolist()]


#*********************************Logic to handle null values*************
for i in range(len(Classes)):
	if SepalLength[i]==0:
  		SepalLength[i] = round(np.mean(SepalLength),1)
	if SepalWidth[i]==0:
  		SepalWidth[i] = round(np.mean(SepalWidth),1)
	if PetalLength[i]==0:
  		PetalLength[i] = round(np.mean(PetalLength),1)
	if PetalWidth[i]==0:
  		PetalWidth[i] = round(np.mean(PetalWidth),1)

#*********************************Logic to observe outliers***************
plt.boxplot([SepalLength,SepalWidth,PetalLength,PetalWidth])
plt.show()


#*********************************Reduction of Dataset********************
File1 = open("IrisTrainData.csv", 'w')
File2 = open("IrisTestData.csv", 'w')
writer = csv.writer(File1)
writer2 = csv.writer(File2)
for i in range(len(Classes)):
	if(i%5==0):
		writer2.writerow(map(str,[SepalLength[i],SepalWidth[i],PetalLength[i],PetalWidth[i],Classes[i]]))
	else:
		writer.writerow(map(str,[SepalLength[i],SepalWidth[i],PetalLength[i],PetalWidth[i],Classes[i]]))
File1.close()
File2.close()
print "Dataset preprossesing is completed- \n\t New data is written in \"IrisTrainData.csv\" and \"IrisTestData.csv\""




 
