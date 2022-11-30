import os
#os.environ['TP_CPP_MIN_LOG_LEVEL'] = '2'
#--
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys # for debugging
import copy
import random
import datetime

from preprocessing.fileprocessor import *
from preprocessing.preprocessor import *
from preprocessing.calculateError import *

from modelGraph import *

from tensorflow.contrib import rnn
from tensorflow.python.ops import variable_scope
from tensorflow.python.framework import dtypes


# obtaining the Data ------------------------

import argparse

parser = argparse.ArgumentParser(description='lstm')
parser.add_argument('--fileName', type=str, default='', #./Building_hourly/LH.csv
                    help='Path where the files are located')
parser.add_argument('--path', type=str, default='./../Results/Hourly/lstm', #./Hourly/LH/
                    help='Path where the files are located')
parser.add_argument('--x-length', type=int, default=24, metavar='XL',
                    help='previous time steps (default: 20)')
parser.add_argument('--y-length', type=int, default=12, metavar='YL',
                    help='steps to predict (default: 10)')
parser.add_argument('--minEpoch', type=int, default=1000, metavar='ME',
                    help='minimum number of epochs (default: 20)')
parser.add_argument('--hiddenLayer', type=int, default=200, metavar='HL',
                    help='number of hidden layers (default: 20)')
parser.add_argument('--numLayers', type=int, default=2, metavar='NL',
                    help='number of layers')
parser.add_argument('--modelPath', type=str, default='', 
                    help='Path to restore the model from')
parser.add_argument('--learningRate', type=float, default=0.01, metavar='LR',
                    help='learning rate')
parser.add_argument('--trainPercentage', type=float, default=0.75, metavar='TP',
                    help='training percentage')
parser.add_argument('--inputCols', type=str, default='C', 
                    help='columns to take as features')
parser.add_argument('--input-dim', type=int, default=1, metavar='ID',
                    help='steps to predict (default: 10)')
parser.add_argument('--output-dim', type=int, default=1, metavar='OD',
                    help='steps to predict (default: 10)')
parser.add_argument('--tanhOut', type=int, default=0, metavar='TL',
                    help='if to include or not tanh layer')
#parser.add_argument('--relu', type=int, default=0, metavar='RE', help='if to include or not relu layer')

args = parser.parse_args()

x_length = args.x_length #20 # the input sequence length
y_length = args.y_length #10 # the output sequence length
fileName = args.fileName
path = args.path
minEpoch = args.minEpoch
hidden_size = args.hiddenLayer #200 # LSTM hidden node size
modelPath = args.modelPath
initial_learning_rate = args.learningRate # learning rate parameter
building = args.fileName[-6:-4]#building.csv #args.building

assert len(args.inputCols) == args.input_dim

percentage = args.trainPercentage #0.75 # the percentage of data used for training
input_dim = args.input_dim # the number of input signals
output_dim = args.output_dim # the number of output signals

name = building+"_"+str(x_length)+"_"+str(y_length)

folder  = datetime.datetime.now().strftime("%y-%m-%d_%H-%M")
save_path_name = os.path.join(args.path,building,folder)
os.makedirs(save_path_name)


save_object_name = name + "_model"
print "Model path ", args.modelPath

X_train_data, Y_train_data, X_test_data, Y_test_data = getData(args.fileName,x_length,y_length,percentage, args.input_dim,args.inputCols)

X_train = np.array(X_train_data)
Y_train = np.array(Y_train_data)
X_test = np.array(X_test_data)
Y_test = np.array(Y_test_data)

print X_train.shape, Y_train.shape, X_test.shape, Y_test.shape

#-----------------------------------------------------
# un-guided training method
#loss_t = 300 # needs to be some random value less than LOSS_LIMIT
ep = 0;
avg_rmse_lim = 3
LOSS_LIMIT = avg_rmse_lim * avg_rmse_lim
CONTINUE_FLAG = True
EPOCH_LIMIT = args.minEpoch#50#20000#50000
MIN_EPOCH_LIM = args.minEpoch#50

iteration = ["iteration"]
train_loss = ["trainLoss"]
train_RMSE = ["trainRMSE"]
testing_RMSE = ["testRMSE"]
train_RMSE_avg = ["trainRMSE-avg"] # it did not have title
test_RMSE_avg = ["testRMSE-avg"]
test_rel0 = ["testRel0"]
test_rel1 = ["testRel1"]
past_loss_values = []
epoch_range = 5

#---------- RESTORING SAVED MODEL ----------------#
rnn_model =graph(args,feed_previous=True) #un-guided training model
temp_saver = rnn_model['saver']()

if args.modelPath == '':
    print("No previous model used")
    #tf.reset_default_graph()
    init = tf.global_variables_initializer()
# -------------- TRAINING ---------------#
with tf.Session() as sess:
    if args.modelPath != '':
        temp_saver.restore(sess, os.path.join(args.modelPath, save_object_name))#name+'_model'
    else:
        init.run()
    while CONTINUE_FLAG:
        ### Feeding data into tensors
        # enc_inpt represents the encoder input
        feed_dict = {       rnn_model['enc_inp'][t]:X_train[:,t].reshape(-1,input_dim) for t in range(x_length)         }
        # target_seq represents desired decoder outputs
        feed_dict.update({  rnn_model['target_seq'][t]:Y_train[:,t,0].reshape(-1,output_dim) for t in range(y_length)   })
        # dec_inp_init represents decoder inputs
        feed_dict.update({  rnn_model['dec_inp_init'][t]:Y_train[:,t].reshape(-1,input_dim) for t in range(y_length)    })
 
        train_t,loss_tr,out_t = sess.run([rnn_model['train_op'],rnn_model['loss'],rnn_model['reshaped_outputs']],feed_dict)

        if ep % (EPOCH_LIMIT/100) == 0:

            temp_output = np.reshape(out_t,(y_length,-1))
            temp_output = temp_output.transpose()
            temp_y_found = temp_output.tolist()
            temp_err = RMSE(Y_train_data[:,:,0],temp_y_found)

            train_loss.append(loss_tr)
            train_RMSE.append(temp_err)
            iteration.append(ep)
            rmse_avg = np.round(np.mean(temp_err),2)
            train_RMSE_avg.append(rmse_avg)

            #-------------------- STATE LOGGER--------------------------------
            # log state of identified values every 2000 epochs
            if ep % (EPOCH_LIMIT/5) == 0:
                temp_saver = rnn_model['saver']()
                save_path = temp_saver.save(sess,os.path.join(save_path_name,save_object_name))                
            #-----------------------------------------------------------------

            #------------------ Testing ---------------
            feed_dict2 = {rnn_model['enc_inp'][t]:X_test[:,t].reshape(-1,input_dim) for t in range(x_length)}
            Y_temp = np.zeros((len(X_test),y_length), dtype=np.float)
            feed_dict2.update({rnn_model['target_seq'][t]:Y_temp[:,t].reshape(-1,output_dim) for t in range(y_length)})
            loss_tst, out_t = sess.run([rnn_model['loss'],rnn_model['reshaped_outputs']],feed_dict2)
            matrix = np.reshape(out_t,(y_length,-1))
            matrix = matrix.transpose()
            Y_found = matrix.tolist()
            err = RMSE(Y_test[:,:,0],Y_found)
            listErrDay0, listErrDay1, avgERR0, avgERR1 = RelE(Y_found,Y_test[:,:,0],args.y_length)

            rmse_avg_test = np.round(np.mean(err),4)
            testing_RMSE.append(loss_tst)
            test_rel0.append(avgERR0)
            test_rel1.append(avgERR1)

        if ep % (EPOCH_LIMIT/10) == 0:
            print ep,"Train loss:", loss_tr, "Train rmse:",rmse_avg, "Test loss:",loss_tst,"Testing RMSE/RelE0/RelE1:", rmse_avg_test, np.round(np.mean(avgERR0),2), np.round(np.mean(avgERR1),2)
            #print "      rmse all ", np.round(temp_err,2)

        #-- condition to stop training - condition to keep track of past losses
        if ep < epoch_range:
            past_loss_values.append(loss_tr)
        else:
            past_loss_values.pop(0)
            past_loss_values.append(loss_tr)
        # increase the epoch count
        ep += 1
        #-- find if the entire range of previous losses are below a threshold
        count = 0
        for val in past_loss_values:
            if val < LOSS_LIMIT:
                count += 1
        #-- stopping condition for training
        if (count >= epoch_range or ep >= EPOCH_LIMIT) and ep>= MIN_EPOCH_LIM:
            print(count,ep)
            CONTINUE_FLAG = False

    print "--- training complete ---"

    save_path = temp_saver.save(sess,os.path.join(save_path_name,save_object_name))
    print "--- session saved ---"

    print "--- testing started ---"
    feed_dict2 = {rnn_model['enc_inp'][t]:X_test[:,t].reshape(-1,input_dim) for t in range(x_length)}
    Y_temp = np.zeros((len(X_test),y_length), dtype=np.float)
    feed_dict2.update({rnn_model['target_seq'][t]:Y_temp[:,t].reshape(-1,output_dim) for t in range(y_length)})
    loss_t, out_t = sess.run([rnn_model['loss'],rnn_model['reshaped_outputs']],feed_dict2)
    #(time,batch,dim)
    matrix = np.reshape(out_t,(y_length,-1))
    matrix = matrix.transpose()
    Y_found = matrix.tolist()
    err = RMSE(Y_test[:,:,0],Y_found)
    maerr = MAE(Y_test[:,:,0],Y_found)
    listErrDay0, listErrDay1, avgERR0, avgERR1 = RelE(Y_found,Y_test[:,:,0],args.y_length)
    print " loss: ", loss_t
    print "test error: ",err

with open(os.path.join(save_path_name,"params.csv"),'w') as f:
    csv_reader = csv.writer(f, delimiter=',')
    csv_reader.writerow([args.inputCols,str(args.input_dim),str(args.x_length)+"_"+str(args.y_length),args.fileName[-6:-4],args.learningRate, args.hiddenLayer, rmse_avg_test, avgERR0, avgERR1])

#------------- SAving files in date foler and directly in lstm/ -----------#
writeColumns(os.path.join(save_path_name,name+"_Train_loss_rmse.csv"), zip(iteration,train_loss,train_RMSE_avg))
writeColumns(os.path.join(save_path_name,name+"_Train_rmse_STEP.csv"),zip(iteration,train_RMSE)) # saving the train RMSE values for every 200th epoch
writeErrResult(os.path.join(save_path_name,name+"_test_rmse.txt"),err) # y_length
writeErrResult(os.path.join(args.path,name+"_test_rmse.txt"),err)
writeErrResult(os.path.join(save_path_name,name+"_test_mae.txt"),maerr)
writeErrResult(os.path.join(args.path,name+"_test_mae.txt"),maerr)
writeErrResult(os.path.join(save_path_name,name+"_test_loss.txt"),[loss_t])
writeErrResult(os.path.join(save_path_name,name+"_test_rel0.txt"),listErrDay0)
writeErrResult(os.path.join(save_path_name,name+"_test_rel1.txt"),listErrDay1)


#np.save(os.path.join(save_path_name,name+"_pred"),Y_found)
#np.save(os.path.join(save_path_name,name+"_real"),Y_test_data)
writetofile(os.path.join(save_path_name,name+"_pred.txt"),np.round(Y_found,2))
writetofile(os.path.join(save_path_name,name+"_real.txt"),Y_test[:,:,0])

writetofile(os.path.join(args.path,name+"_pred.txt"),np.round(Y_found,2))
writetofile(os.path.join(args.path,name+"_real.txt"),Y_test[:,:,0])

print "----- run complete-------"
print datetime.datetime.now()    
    
    
    
