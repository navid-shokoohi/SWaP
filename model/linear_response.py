import numpy as np
import sys
#import global_var
#sys.path.append(global_var.project_path+"/model")
#import getSamples as gs
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
#from preprocessing.fileprocessor import *
from preprocessing.preprocessor import *
from preprocessing.calculateError import *
import datetime
import os
import sys
import argparse

top_buildings = ['BN', 'BR', 'DG', 'EB', 'GE', 'JS', 'S2', 'S3', 'SN', 'DE', 'FA', 'C4', 'LH', 'RA']

parser = argparse.ArgumentParser(description='lstm')
parser.add_argument('--x-length', type=int, default=24, metavar='XL', help='previous time steps (default: 20)')
parser.add_argument('--y-length', type=int, default=12, metavar='YL', help='steps to predict (default: 10)')
parser.add_argument('--trainPercentage', type=float, default=0.75, metavar='TP', help='training percentage')
parser.add_argument('--iterations', type=int, default=10, metavar='IT', help='number of iterations for training')
parser.add_argument('--path', type=str, default='./Results/Hourly/linear/', help='Path where the files are located')
parser.add_argument('--fileName', type=str, default='./Data/Hourly/', help='Path where the files are located') #./Building_hourly/LH.csv

args = parser.parse_args()

x_length = args.x_length #20 # the input sequence length
y_length = args.y_length #10 # the output sequence length
percentage = args.trainPercentage
iterations = args.iterations  # number of GCRF models
fileName1 = args.fileName

def linear_regression_model():

	for building in top_buildings:

		#### SET UP FOLDER PATHS AND OUTPUT FILE NAMES, PATHS
		dir_path = args.path+building
		folder = datetime.datetime.now().strftime("%y-%m-%d_%H-%M")
		dir_path+="/"+folder+"/"
		os.makedirs(dir_path)

		datafileName = dir_path + building + "_"+str(x_length)+"_"+str(y_length)
		real_value_path = datafileName+"_real.txt"
		rela_error_path = datafileName+"_relativeError.txt"
		predicted_value_path = datafileName+ "_pred.txt"

		fileName = fileName1+ building + '.csv'

		pred = open(datafileName+ "_pred.txt", "w")
		real = open( datafileName+"_real.txt", "w")
		error = open(datafileName+"_relativeError.txt", "w")
		fh2 = open(datafileName+"_test_rmse.txt", "w")
		fmae = open(datafileName+"_test_mae.txt", "w")

		param_file = open(dir_path+"params_history.txt","w")
		param_file.write("Building : "+building+"\nxlength : "+str(x_length)+"\nyLength : "+str(y_length)+"\n\n")
		param_file.close()


		#### LOADING DATA
		X_train, Y_train, X_test, Y_test = getData(fileName, x_length, y_length,percentage)

		x = [i for i in range(0, x_length)]
		y = [i for i in range(x_length, x_length + y_length)]
		x_days = np.array(x)
		y_days = np.array(y)
		x_days = x_days.reshape((-1, 1))
		y_days = y_days.reshape((-1, 1))

		predictions = []

		# for each test sample, fit a straight line and calculate the MSE

		model = linear_model.LinearRegression()
		counter = 0
		for i in range(0, X_test.shape[0]):
			real_value = Y_test[i, :].reshape((-1, 1))
			model.fit(x_days, X_test[i, :].reshape((-1, 1)))
			prediction = model.predict(y_days)
			for k in range(0,y_length):
				if prediction[k][0] < 0:
					prediction[k][0] = 0
				pred.write(str(round(prediction[k][0],2))+",")

			pred.write("\n")

			for k in range(0, y_length):
				real.write(str(round(real_value[k][0],2))+",")
			real.write("\n")
			counter+=1
			predictions.append(prediction)

		predictions = np.array(predictions).reshape((-1, y_length))


		# Get RMSE over each day, and different rmse calculations
		#rmse_days, rmse1, rmse2 = rmse(predictions, Y_test)
		rmse_days =  RMSE(predictions,Y_test)
		mae_days = MAE(predictions,Y_test)
		relative_error0, relative_error1, avg0,avg1 = RelE(predictions,Y_test,y_length)

		# Write OUTPUT FILES
		for r in rmse_days:
			fh2.write(str(r) + '\n')

		for r in mae_days:
			fmae.write(str(r) + '\n')
		error.write("RE0  , RE1  ,\n")
		for i in range(y_length):
			error.write(str(round(relative_error0[i],2))+" ,"+str(round(relative_error1[i],2))+"\n")
		error.write("\naverage0 : "+str(round(avg0,2))+"\naverage1 : "+str(round(avg1,2)))

		fh2.close()
		fmae.close()
		pred.close()
		real.close()
		error.close()



linear_regression_model()
