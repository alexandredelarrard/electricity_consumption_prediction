# Prediction for different time lags of the electricity consumption

This directory aims to give a strategy to predict electricity consumption based on GBM algorithms. 

# 1) Data Preparation
This project aimed to predict electricity consumption in a European city for 3 different time scale: 
  - Hourly predictions for the next day (J+1) between 00h00 and 23h30 basd on historical lags between past and J 9h00
  - Hourly predictions for the next 14 days (J+14) on historical lags between past and J 9h00
  - Dayly predictions for the next three years 
  
The forth model called "Part climatique" tries to predict the influence of the temperature fluctuation into the elecrticity consumption. It is based on the assumption that at 16 °C temperature does not influence electricity consumption (no heating or cooling devices).

All models were using information given by the data furnisher that are : 
  - Calendar dates
  - Temperature with points every three hours
  - Electricity consumption lags with points every half an hour
  - Bank holidays = "jours fériés"
  
 Additionnal data were added and seemed to have a significant signal:
  - Sun rise and down 
  - Number of hours with sun
  - Probability to be at work (specifically for days between week-ends and bank holidays)
  
 The data preparation is basically the same for all first three models. It includes a selection of temperature lags (no longer significant after 3 days of lags), consumption lags (up to 14 days in the past), creation of holiday feature engineering, addition of sun descriptors. 
 
# 2) Modelling & Results

The modelling used the same strategy for J+1 and J+14 models with different lag selection and fine tuning. Both of these models are based on xgboost regression with one model for each hour. Such a strategy gave a great gain in RMSE/MAPE reduction in comparison with a global GBM on all the dataset. 
 
 Results are the following ones for J+1 model: Model J+1 (predict J+1 + [0h00, 23h30] from past points up to J + 9h) :
 - 01/2015  RMSE total : 1360.9273697 ; MAPE total : 0.0173845828551
 - 02/2015  RMSE total : 1223.22371606 ; MAPE total : 0.0148847618752
 - 03/2015  RMSE total : 1358.83369245 ; MAPE total : 0.0187626411252
 - 04/2015  RMSE total : 1020.20744528 ; MAPE total : 0.0140366121832
 - 05/2015  RMSE total : 1022.71998352 ; MAPE total : 0.0151548063196
 - 06/2015  RMSE total : 1264.40266626 ; MAPE total : 0.0169174533141
 - 07/2015  RMSE total :  2143.19048742 ; MAPE total : 0.0248809474292
 
A visualisation of predctions made for the J+14 model can be found on https://www.youtube.com/watch?v=A9YDZ0j-XaM

The A+3 model which aims to predict the dayly next three years consumption was performed through two different models. The first one predicted the long term trend which is highly correlated with the economic and demographic growth of the area, evven if such information were missing. The second part of the model was a GBM applied to the difference between hourly data and the predicted trend. Such a two times strategy needs to take care of overfitting.

Model A+3 (predict the next 3 years dayly) :
  - 2009-01-01  2011-12-31 : RMSE total = 3949.31555592;   MAPE totale = 4.83742720779
  - 2012-01-01  2014-12-31 : RMSE total = 3588.95910157;   MAPE totale = 4.15100123566
  - 2013-01-01  2015-12-31 : RMSE total = 3289.88520412;   MAPE totale = 3.8662331455
  
The climatic deduction aims at deducing the impact of temperature on the electricity consumption for the last 15 days. It was deduced by the differenciation between real values for the last 15 days (observed) and the prediction from the J+14 model with a fixed temperature of 16°C. The final result can be observed on the figure below:

![alt text](https://github.com/alexandredelarrard/electricity_consumption_prediction/blob/master/output/Part_climatique/part_climatique.png)

  
