DATA

    14 buildings: 6 Academic buildings, 6 Residenc Halls, 1 Dining Hall and 1 Gym
    The split for training and testing correspond to 75% and 25%

PREPROCESSING

    Dates: From August 1st to December 8th, we also removed Thanksgiving period: November 20th 2018 to November 26th 2018
    Missing values were filled with a linear regressiong Pattern. Ejem 1, - , - , - , 5 was filled with 1, 2, 3, 4, 5
    Outliers were removed when over 2000 limit

CODE

    LSTM code was taken from the code in this paper "A Deep Learning Model for Wireless Channel Quality Prediction" by J. D. Herath and A. Seetharam and A. Ramesh
    GCRF code was taken from "NYCER: A Non-Emergency Response Predictor for NYC using Sparse Gaussian Conditional Random Fields" by David DeFazio and Arti Ramesh
    Libraries used for ARIMA model: pyramid (if will be renamed to pmdarima)
    For LSTM model with tensorflow, create a conda environment from tf_waterPrediction.yml

RUN

    Set the PYTHONPATH variable with the current directory:
        export PYTHONPATH=$PYTHONPATH:/home/gissella/Documents/Research/WaterQuality/water-consumption-prediction
    The parameters that you can choose to run GCRF or LSTM are:
        fileName: location for input file
        input-dim: number of variables at each time step
        inputCols: C (consumption), D (day of the week), W (week of the day) H (hour of the day)
        path: path for the output location
        x-length
        y-length
    You can run LSTM: python model/seq2seq_unguided_LSTM.py --fileName Data/processed_data/Hourly/BN.csv --input-dim 3 --inputCols CDH --path Results/Hourly/lstm --minEpoch 1000 --hiddenLayer 200 --numLayers 1 --x-length 24
    You can run the test of LSTM with the file model/seq2seq_unguided_LSTM_test.py
    You can run GCRF: python model/gcrf_response.py --fileName Data/processed_data/Hourly/LH.csv --input-dim 3 --inputCols CHW --path Results/Hourly/gcrf
