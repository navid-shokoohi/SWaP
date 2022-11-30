import numpy as np

#-----------------------------------------
# helper functions for getData()
#-----------------------------------------

def sampleData(dataset,x_length,y_length):
    x_data_limit = len(dataset) - (x_length+y_length)
    X = []
    Y = []
    for i in range(x_data_limit):
        # for the inputs
        temp_x = []
        for j in range(x_length):
            temp_x.append(dataset[i+j])
        X.append(temp_x)
        # for the outputs
        temp_y = []
        for j in range(y_length):
            temp_y.append(dataset[i+x_length+j])
        Y.append(temp_y)
    return X,Y
        

    
#-----------------------------------------
# main method to obtain data
#-----------------------------------------

# obtains the datasets -> used for the RNN model
# filename : the string name for the file
# x_length : length of the input(timesteps of the past)
# y_length : length of output(timesteps into future)
# percentage : the percentage of data to use for training and testing
import pandas as pd
import numpy as np


def stackHorizontally(dataset,input_dim=1):
    newSet = []
    for i in range(0,len(dataset)):
        instance = np.hstack(dataset[i])
        newSet.append(instance)
    return np.array(newSet)

#75% - 25% in order
def getData(filename,x_length,y_length,percentage, input_dim=1, columns='C', typeModel=0,limitOutlier=5000):

    df = pd.read_csv(filename, delimiter=',')#,usecols=["Water flow"]
    #print df.columns
    median = df.loc[df['Water flow']<=limitOutlier, 'Water flow'].median()
    df.loc[df['Water flow'] > limitOutlier, 'Water flow'] = np.nan
    df.fillna(median,inplace=True)

    #Selecting columns for the dataset
    listCols = {'C':"Water flow",'D':"Day Number",'H':"Hour Number",'W':"Weekday"}
    listCols = [ listCols[column] for column in columns]

    print listCols
    data = df[listCols]

    #-- seperate training and testing --------
    train_size = int(percentage*len(data))
    #test_size = int(len(data)-train_size)
    
    train_data = np.array(data[:train_size])
    test_data = np.array(data[train_size:-1]) #0 to get only the first column: consumption

    X_Train,Y_Train = sampleData(train_data,x_length,y_length)
    X_Test,Y_Test = sampleData(test_data,x_length,y_length)

    X_Train,Y_Train,X_Test,Y_Test = np.array(X_Train),np.array(Y_Train)[:,:,0,None],np.array(X_Test),np.array(Y_Test)[:,:,0,None]

    #LSTM needs it in 3D array and GCRF in 2D array, that is why X sequences are stacked horizontally, to remove 3rd dimension
    if typeModel==1:

        X_Train = stackHorizontally(X_Train,input_dim)
        Y_Train = stackHorizontally(Y_Train,input_dim=1)
        X_Test = stackHorizontally(X_Test,input_dim)
        Y_Test = stackHorizontally(Y_Test,input_dim=1)

    print len(data), train_size, X_Train.shape, Y_Train.shape, X_Test.shape, Y_Test.shape

    return X_Train,Y_Train,X_Test,Y_Test


#getData('./../Data/processed_data/Hourly/BN.csv',24,12,0.75,input_dim=1, columns='C', typeModel=0,limitOutlier=5000)