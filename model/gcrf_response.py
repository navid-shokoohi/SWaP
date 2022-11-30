import numpy as np
import sys
from sgcrf import SparseGaussianCRF
from sklearn.metrics import mean_squared_error

#import getSamples as gs
from preprocessing.fileprocessor import *
import datetime
import os
import sys

from preprocessing.preprocessor import *
from preprocessing.calculateError import *
from preprocessing.fileprocessor import *

import argparse


parser = argparse.ArgumentParser(description='lstm')
parser.add_argument('--fileName', type=str, default='', #./Building_hourly/LH.csv
                    help='Path where the files are located')
parser.add_argument('--path', type=str, default='Results/Hourly/gcrf', #./Hourly/LH/
                    help='Path where the files are located')
parser.add_argument('--x-length', type=int, default=24, metavar='XL',
                    help='previous time steps (default: 20)')
parser.add_argument('--y-length', type=int, default=12, metavar='YL',
                    help='steps to predict (default: 10)')
parser.add_argument('--minEpoch', type=int, default=10, metavar='ME',
                    help='minimum number of epochs (default: 20)')
parser.add_argument('--lambdaL', type=int, default=0.5, metavar='LA',
                    help='default value of lambda')
parser.add_argument('--thetaT', type=int, default=0.5, metavar='TH',
                    help='default value for theta')
parser.add_argument('--modelPath', type=str, default='', 
                    help='Path to restore the model from')
parser.add_argument('--learningRate', type=float, default=0.01, metavar='LR',
                    help='learning rate')
parser.add_argument('--trainPercentage', type=float, default=0.75, metavar='TP',
                    help='training percentage')
parser.add_argument('--inputCols', type=str, default='C', 
                    help='columns to take as features')
parser.add_argument('--input-dim', type=int, default=1, metavar='TL',
                    help='Number of features to include in X')
parser.add_argument('--output-dim', type=int, default=1, metavar='OD',
                    help='steps to predict (default: 10)')
parser.add_argument('--randomSamp', type=int, default=0, metavar='OD',
                    help='not overlapping windows assigned randomly to training and testing set')


args = parser.parse_args()
lambdaL = args.lambdaL
thetaT = args.thetaT
x_length = args.x_length
y_length = args.y_length
building = args.fileName[-6:-4]#building.csv #args.building
lr = args.learningRate
percentage = args.trainPercentage
epochs = args.minEpoch
input_dim = args.input_dim
filename = args.fileName

name = building+"_"+str(x_length)+"_"+str(y_length)
folder  = datetime.datetime.now().strftime("%y-%m-%d_%H-%M")
save_path_name = os.path.join(args.path,building,folder)
os.makedirs(save_path_name)

assert len(args.inputCols) == args.input_dim

folder = datetime.datetime.now().strftime("%y-%m-%d_%H-%M")

print("Running for:", building)

lossIt = ["trainLoss"]
train_RMSE = ["trainRMSE"]
testing_RMSE = ["testRMSE"]
train_RMSE_avg = ["trainRMSE-avg"] # it did not have title
test_RMSE_avg = ["testRMSE-avg"]
listErrDay0 = ["trainig relError"]

#Accomodating input depending if we are usnig only 1 feature or +2 (day of the week, hour of the day)
X_train, Y_train, X_test, Y_test = getData(filename, x_length, y_length,percentage, args.input_dim,args.inputCols,typeModel=1)

print X_train.shape, Y_train.shape, X_test.shape, Y_test.shape

model = SparseGaussianCRF(lamL=lambdaL, lamT=thetaT,learning_rate=lr, n_iter=1000)  # lamL and lamT are regularization parameters
predictions = []
iterations = []

for i in range(0, epochs):

	model.fit(X_train, Y_train)

	#saving the loss
	loss = np.round(model.lnll[-1],2)
	

	pred_train = model.predict(X_train)
	prediction = model.predict(X_test)
	#predictions.append(prediction)

	rmse_train = RMSE(Y_train, pred_train)
	rmse_days = RMSE(Y_test, prediction)
	maerr = MAE(Y_test,prediction)
	avgRMSE_train = np.round(np.mean(rmse_train),4)
	avgRMSE_test = np.round(np.mean(rmse_days),4)
	relerror0_train, relerror1_train, avg0_train, avg1_train = RelE(pred_train, Y_train, y_length)
	relerror0_test, relerror1_test, avg0_test, avg1_test = RelE(prediction, Y_test, y_length)

	lossIt.append(loss)
	train_RMSE_avg.append(avgRMSE_train)
	test_RMSE_avg.append(avgRMSE_test)
	listErrDay0.append(avg0_train)
	iterations.append(i)

	if i % (epochs/10) ==0:
		print "Loss:", loss , "AVG train RMSE:", avgRMSE_train, "AVG test RMSE:", avgRMSE_test, "RelErrs (train-test)",avg0_train,avg0_test


#------------- SAving files in date foler and directly in lstm/ -----------#
with open(os.path.join(save_path_name,"params.csv"),'w') as f:
    csv_reader = csv.writer(f, delimiter=',')
    csv_reader.writerow([args.inputCols,str(args.input_dim),str(args.x_length)+"_"+str(args.y_length),args.fileName[-6:-4],args.learningRate, avgRMSE_test, avg0_test])


#### Writing output files
writeColumns(os.path.join(save_path_name,name+"_Train_loss_rmse.csv"), zip(iterations,lossIt,train_RMSE_avg,test_RMSE_avg))
writeColumns(os.path.join(save_path_name,name+"_Train_rmse_STEP.csv"),zip(iterations,train_RMSE)) # saving the train RMSE values for every 200th epoch
writeErrResult(os.path.join(save_path_name,name+"_test_rmse.txt"),rmse_days) # y_length
writeErrResult(os.path.join(args.path,name+"_test_rmse.txt"),rmse_days)
writeErrResult(os.path.join(save_path_name,name+"_test_mae.txt"),maerr)
writeErrResult(os.path.join(args.path,name+"_test_mae.txt"),maerr)
writeErrResult(os.path.join(save_path_name,name+"_test_rel0.txt"),listErrDay0)

writetofile(os.path.join(save_path_name,name+"_pred.txt"),np.round(prediction,2))
writetofile(os.path.join(save_path_name,name+"_real.txt"),Y_test)

writetofile(os.path.join(args.path,name+"_pred.txt"),np.round(prediction,2))
writetofile(os.path.join(args.path,name+"_real.txt"),Y_test)

print "----- run complete-------"
print datetime.datetime.now()