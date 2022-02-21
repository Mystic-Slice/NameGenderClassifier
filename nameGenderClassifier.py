import os
import re
from time import perf_counter
import numpy as np
from sklearn.metrics import accuracy_score
from torch import nn, from_numpy, tensor, load, save
from torch.optim import SGD
from torch.utils.data import random_split

INPUT_SIZE = 26*5
HIDDEN_SIZE = 1024
OUTPUT_SIZE = 1

EPOCHS = 10
MODEL_PATH = os.path.join(os.getcwd(), "classifier_neural_network.model")

# 3-layer neural network
classifierNetwork = nn.Sequential(
    nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
    nn.ReLU(),
    nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE),
    nn.Sigmoid()
)   

def getData():
    with open('names.csv', 'r') as f:
        data = [line.strip().split(',') for line in f]

    names = [entry[0] for entry in data]
    gender = [[1] if entry[1] == "boy" else [0] for entry in data]

    data = [[names[i], gender[i]] for i in range(len(names))]
    return data[1:]

# convert the name into a vector that contains occurrence-indices of each character
def vectorize(name):
    name = name.lower()
    name = re.sub("[^a-z]", "z", name)
    nameVector = np.zeros((26, 5))

    for index, letter in enumerate(name):
        code = ord(letter) - ord('a')
        for i in range(10):
            if nameVector[code][i] == 0:
                nameVector[code][i] = index+1
                break

    return nameVector.flatten()

def vectorizeNames(data):
    return [vectorize(entry[0]) for entry in data]

def splitData(data):
    trainSize = int(0.8 * len(data))
    testSize = len(data) - trainSize
    return random_split(data, [trainSize, testSize])

def trainClassifier(data):
    dataTrain, dataTest = splitData(data)
    dataTrain = [dataTrain[i] for i in range(len(dataTrain)) if i%2 == 0]
    trainSize = len(dataTrain)

    nameTrain = [entry[0] for entry in dataTrain]
    nameVectorTrain = vectorizeNames(nameTrain)
    genderTrain = [entry[1] for entry in dataTrain]

    nameVectorTrain = [from_numpy(nameVector) for nameVector in nameVectorTrain]
    genderTrain = [tensor(genderVector) for genderVector in genderTrain]

    criterion = nn.BCELoss()
    optimizer = SGD(classifierNetwork.parameters(), lr=0.01, momentum=0.9)
    for epoch in range(EPOCHS):
        print(f"Epoch: {epoch+1}/{EPOCHS}")
        startTime = perf_counter()
        for i in range(trainSize):
            print(i,"\r", end='')
            optimizer.zero_grad()
            output = classifierNetwork(nameVectorTrain[i].float())
            loss = criterion(output, genderTrain[i].float())
            loss.backward()
            optimizer.step()
        endTime = perf_counter()
        print(f"Epoch completed. Time taken: {endTime-startTime:0.4f} seconds")
        evaluateClassifier(dataTest)    
        save(classifierNetwork.state_dict(), MODEL_PATH)

def evaluateClassifier(dataTest):
    testSize = len(dataTest)
    predictions = []

    nameTest = [entry[0] for entry in dataTest]
    nameVectorTest = vectorizeNames(nameTest)
    genderTest = [entry[1] for entry in dataTest]

    nameVectorTest = [from_numpy(nameVector) for nameVector in nameVectorTest]
    genderTest = [tensor(genderVector) for genderVector in genderTest]

    for i in range(testSize):
        output = classifierNetwork(nameVectorTest[i].float())
        output = output.detach().numpy()
        output = output.round()

        predictions.append(output)

    predictions, genderTest = np.vstack(predictions), np.vstack(genderTest)

    acc = accuracy_score(genderTest, predictions)
    print("Accuracy: %.3f" % acc)

def predict(name):
    name = tensor(vectorize(name)).float()

    prediction = classifierNetwork(name)
    prediction = prediction.detach().numpy()

    if prediction[0] > 0.5:
        print("Boy")
    else:
        print("Girl")

if os.path.exists(MODEL_PATH):
    print("Pre-existing model loaded")
    classifierNetwork.load_state_dict(load(MODEL_PATH))
else:
    print("Training new model")
    data = getData()
    trainClassifier(data)

while True:
    name = input("Enter name: ")
    if name == "stop":
        break
    predict(name)

save(classifierNetwork.state_dict(), MODEL_PATH)