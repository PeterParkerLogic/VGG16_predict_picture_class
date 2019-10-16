import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.utils.data as Data
import matplotlib.pyplot as plt
import torch.utils.data as data
import torchvision
from torchvision import datasets,transforms, models
import numpy as np        
import csv
import ast
import os
from torch.autograd import Variable 
import time


BATCH_SIZE_train = 4
BATCH_SIZE_test = 1
TRAIN_DATA_PATH = "/home/peterpaker/basic_hw1_restart/cs-ioc5008-hw1 (2)/dataset/dataset/train"
TEST_DATA_PATH = "/home/peterpaker/basic_hw1_restart/cs-ioc5008-hw1 (2)/dataset/dataset/test1"
TRANSFORM_IMG = transforms.Compose([
    transforms.RandomResizedCrop(size = 256, scale = (0.8, 1.0)),
    transforms.RandomRotation(degrees = 15),
    transforms.ColorJitter(),
    transforms.RandomHorizontalFlip(),
    transforms.CenterCrop(size = 224),  # Image net standards
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
])
train_data = torchvision.datasets.ImageFolder(root = TRAIN_DATA_PATH, transform  = TRANSFORM_IMG)
train_data_loader = data.DataLoader(train_data, batch_size = BATCH_SIZE_train, shuffle = True,  num_workers = 4)
test_data = torchvision.datasets.ImageFolder(root = TEST_DATA_PATH, transform = TRANSFORM_IMG)
test_data_loader  = data.DataLoader(test_data, batch_size = BATCH_SIZE_test, shuffle = False, num_workers = 4) 
print("Detected Classes are: ", train_data.class_to_idx)

#VGG definition
from torchvision import models
model = models.vgg16(pretrained = True)
print(model)

for parma in model.parameters():
    parma.requires_grad = False

# Newly created modules have require_grad=True by default
num_features = model.classifier[6].in_features
features = list(model.classifier.children())[:-1] # Remove last layer
features.extend([nn.Linear(num_features, 13)]) # Add our layer with 4 outputs
model.classifier = nn.Sequential(*features) # Replace the model classifier

use_gpu = torch.cuda.is_available()
print(use_gpu)   
if use_gpu:
    model = model.cuda()
print(model)

cost = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters())


def network(activation_function_string, epoch_setting = 150, learning_rate = 0.05):
    print("in network epcoh is "+str(epoch_setting))
    if activation_function_string == 'ELU':
        activation_function = nn.ELU()
    elif activation_function_string == 'ReLU':
        activation_function = nn.ReLU()
    elif activation_function_string == 'LeakyReLU':
        activation_function = nn.LeakyReLU()
    else: return 'no such activation_function'
    

    outputModelNo = 0
    device = torch.device("cuda:0,1" if torch.cuda.is_available() else "cpu")


    
    

    for epoch in range(epoch_setting):
        
        # training
        for batchidx, (train_input, train_label) in enumerate(train_data_loader):
            train_input = train_input.to(device)
            train_label = train_label.to(device)
            # set module to training mode
            model.train = True
            # forwarding
            optimizer.zero_grad()   # zero the gradient buffers           
            train_output = model(train_input)
            # calculate the loss
            train_label = train_label.long()
            # loss = lossFunction(train_output, train_label)
            loss = cost(train_output, train_label)
            if batchidx % 100 == 0:
                print(loss)
            # backpropageation
            loss.backward()

            # weight updating
            # create your optimizer
            optimizer.step()    # Does the update


        correct_results = 0
        total_results = 0

        

        if epoch % 10 == 0:
            outputModelNo += 1
            print("DeepConvNet with ", activation_function_string)
            torch.save(model.state_dict(), '/home/peterpaker/basic_hw1_restart/save_model50_10_13/save_model_No'+str(outputModelNo)+'.pkl')
            print("---------------------------------------------")
           
    


# use test data to test the model
# def testTheModel:
  
    # define a dict for storing the testing result
    testResultDict = {}
    classNameMapDict = {'3':'highway', '1':'coast', '12':'tallbuilding', '2':'forest', '5':'kitchen', '10':'street', '8':'office', '0':'bedroom', '6':'livingroom', '4':'insidecity', '9':'opencountry', '11':'suburb', '7':'mountain'}
    for batchidx, (testing_input, testing_label) in enumerate(test_data_loader):
            
            testing_input = testing_input.to(device)
            testing_label = testing_label.to(device)
            with torch.no_grad():
                testing_output = model(testing_input)
            _, predicted = torch.max(testing_output, 1)
            
            # print out the prediction result for each test data
            predicted = str(np.around(predicted.cpu().numpy())[0])

            # get the corresponding class name of the prediction result.
            predictedClassName = classNameMapDict[predicted]

            # get testing file path
            testDataName = test_data_loader.dataset.samples[batchidx]

            # extract testing file name from the file path
            testDataName = testDataName[0].rsplit('/', 1)[1]
            testDataName = testDataName.split(".")
            testDataName = testDataName[0]

            # add the predict result and file name to a dict
            testResultDict[testDataName] = predictedClassName
            

            
    # write to csv file      
    with open('PeterCNNOutput.csv', 'w') as f:
       
        outputCSVFileWriter = csv.writer(f)
        # outputCSVFileWriter.writerow(row)

        for key in sorted(testResultDict.keys()) :
            outputCSVFileWriter.writerow([key,testResultDict[key]])
        
            
network('ReLU', epoch_setting = 30, learning_rate = 0.003)



