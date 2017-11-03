# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 10:36:50 2016

@author: alexandre
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 07 21:13:53 2016
@author: alexandre
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
import xgboost as xgb
import operator
np.random.seed(1304)
warnings.filterwarnings('ignore')

class J_1_Prediction(object):
    
    def __init__(self, directory_data, directory_modele, date_debut, train_test, build_data):
        
        self.data = pd.DataFrame()
        self.directory_data = directory_data
        self.directory_modele = directory_modele
        self.RMSE = 0
        self.MAPE = 0
        
        #### 
        if build_data.upper()== "YES":
            os.chdir(self.directory_modele + "Scripts/")   
            from DataprepJ1 import DataPreparation as dp
            tt = dp(self.directory_data + "Data/") 
            self.data = tt.conso
        else:
            if build_data.upper()== "NO":
                self.data = pd.read_csv(self.directory_data + "Data/prepared/data_prepared_pred_conso.csv" , sep= "," , header =0)
            else:
                print("need yes or no values to know if the dataset has to be built or not")
        
        ### hyper parameters of the algorithm 
        self.Y_label = "Consommation"
        self.loss = "ls"
        self.learning_rate =  np.array(np.ones(24)*0.02)
        self.depth = np.array(np.ones(24)*5)
        self.learning_rate[0:2] =0.016
        self.learning_rate[2:5] =0.017
        self.learning_rate[15] = 0.03
        self.learning_rate[17] = 0.025
        self.learning_rate[18] = 0.03
        self.learning_rate[19] = 0.025
        self.learning_rate[20] = 0.028
        
        ## space to train and test the algorithm
        self.date_debut = date_debut
        
        if train_test.upper() =="TRAIN":
            self.sortie = self.GBM()
            
        else: 
            if train_test.upper() =="TEST":
                 self.sortie = self.predicions_1jours()
                
            else: 
                print("need train or test values")
        
    def evalerror(self, yhat, y):
        y = np.exp(y.get_label())
        yhat = np.exp(yhat)
        return  "rmse", np.sqrt(np.mean((y - yhat)**2))
        
    def plots(self, data, clf):
    
        def ceate_feature_map(data, Y_label):
            
            features = data.drop(Y_label,axis=1).columns
            outfile = open(self.directory_data + 'Output/xgb.fmap', 'w')
            i = 0
            for feat in features:
                outfile.write('{0}\t{1}\tq\n'.format(i, feat))
                i = i + 1
        
            outfile.close()
            
        if not os.path.exists(self.directory_data + "Output"):
              os.makedirs(self.directory_data + "Output")
              
        ceate_feature_map(data, self.Y_label)
          
        importance = clf.get_fscore(fmap= self.directory_data + 'Output/xgb.fmap')
        importance = sorted(importance.items(), key=operator.itemgetter(1))
        
        df = pd.DataFrame(importance, columns=['feature', 'fscore'])
        df['fscore'] = df['fscore'] / df['fscore'].sum()
        
        plt.figure()
        df.plot()
        df.plot(kind='barh', x='feature', y='fscore', legend=False,figsize=(8, 18))
        plt.title('XGBoost Feature Importance')
        plt.xlabel('relative importance')
        plt.gcf().savefig(self.directory_data + 'Output/feature_importance_xgb.png')
        plt.close()
        
        return df
           
    def GBM(self):   
            
        datas = self.data.copy()
        
        ### test = split train test from begining date
        
        test = datas[pd.to_datetime(datas.Date) == pd.to_datetime(self.date_debut)].copy()
        train = datas[(datas.Date < self.date_debut)].copy()
        test['Predictions'] = 0

        for h in range(24):
                    
                train1 = train[ (train['Heure']< (h+1))&(train['Heure']>=h)].copy()                     
                test1 = test[(test['Heure']< (h+1))&(test['Heure']>=h)].copy()                   
                
                train1 = train1.drop(['Heure', "Date"], axis=1)
                test1 = test1.drop(['Heure', "Date"], axis=1)
                
                y_train = np.log(train1[self.Y_label])        
                train_i_M = xgb.DMatrix(train1.drop(self.Y_label, axis=1), label=y_train)
                    
                y_test = np.log(test1[self.Y_label])        
                test_i_M = xgb.DMatrix(test1.drop([self.Y_label,'Predictions'] , axis=1), label=y_test)      
                    
                ####### fitting
                params = {"objective": "reg:linear",
                              "booster" : "gbtree",
                              "eta": self.learning_rate[h],
                              "max_depth": self.depth[h],
                              "subsample": 0.86,
                              "colsample_bytree": 0.9,
                              "colsample_bylevel" : 0.9,
                              "silent": 1,
                              "min_child_weight": 1,
                              "seed": 1301}
                
                clf =xgb.train(params, train_i_M, 1200)   
                prediction_bis = clf.predict(test_i_M)   
                
                # save model    
                if not os.path.exists(self.directory_data + "Trained_models/"):
                    os.makedirs(self.directory_data + "Trained_models/")
                clf.save_model(self.directory_data +  "Trained_models/model_"+ str(h) + '.model')

                test['Predictions'][test['Heure'] == h]  = np.exp(prediction_bis)
        
        test = test.reset_index(drop= True)
        print("_"*80)
        print("Date: "+ str(test["Date"].value_counts().index[0]) +"; MAPE = " + str(np.mean(abs(test[self.Y_label]-  test['Predictions'])/test[self.Y_label] )) + "; RMSE = " + str(np.sqrt(((test[self.Y_label]- test['Predictions'])**2).mean())))
        self.MAPE = np.mean(abs(test[self.Y_label]-  test['Predictions'])/test[self.Y_label] )
        self.RMSE = np.sqrt(((test[self.Y_label]-test['Predictions'])**2).mean())
        daterange = pd.date_range(pd.to_datetime(self.date_debut), pd.to_datetime(self.date_debut) + pd.DateOffset(1), freq= "H")
            
        test.index = daterange[:-1]
                
        if not os.path.exists(self.directory_data + 'Output/'):
                    os.makedirs(self.directory_data + 'Output/')
        try:        
            plt.figure()
            test[['Predictions', self.Y_label]].plot(figsize= (10,6))
            plt.title("Predictions entre J=" + str(self.date_debut) + " et J+ 1")
            plt.xlabel('Jours')
            plt.ylabel('Consommation')
            plt.gcf().savefig(self.directory_data + 'Output/1day_prediction_' + str(self.date_debut) +'.png')
                    
        except Exception:
                print("prediction mode online")
                plt.figure()
                test[['Predictions']].plot(figsize= (10,6))
                plt.title("Predictions entre J=" + str(self.date_debut) + " et J+ 1")
                plt.xlabel('Jours')
                plt.ylabel('Consommation')
                plt.gcf().savefig(self.directory_data + 'Output/1day_prediction_' + str(self.date_debut) +'.png')
                pass
            
        return test[["Date", "Heure" , "Predictions"]]
            
            
    def predicions_1jours(self):
            
            liste_jours= ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
            print("prediction de "+ liste_jours[(pd.to_datetime(self.date_debut, format='%Y-%m-%d')).dayofweek] + " : " + str(pd.to_datetime(self.date_debut, format='%Y-%m-%d')))
            datas = self.data.copy()
        
            test = datas[ pd.to_datetime(datas.Date) == pd.to_datetime(self.date_debut).to_datetime()].copy()
            test['Predictions'] = 0
            
            for h in range(0,24):
                test1 = test[(test['Heure']< (h+1))&(test['Heure']>=h)].copy()                   
                
                clf = xgb.Booster({'nthread':4}) 
                clf.load_model(self.directory_data + "Trained_models/model_"+  str(h) +".model") 
                    
                prediction_bis = clf.predict(xgb.DMatrix(test1.drop([self.Y_label, 'Predictions', 'Heure', 'Date'] , axis=1))) 
                test['Predictions'][test['Heure'] == h] = np.exp(prediction_bis)
            
            test = test.reset_index(drop= True)
            print("_"*80)
            print("Date: "+ str(test["Date"].value_counts().index[0]) +"; MAPE = " + str(np.mean(abs(test[self.Y_label]-  test['Predictions'])/test[self.Y_label] )) + "; RMSE = " + str(np.sqrt(((test[self.Y_label]- test['Predictions'])**2).mean())))
        
            self.RMSE = np.sqrt(((test[self.Y_label]-test['Predictions'])**2).mean())
            self.MAPE = np.mean(abs(test[self.Y_label]-  test['Predictions'])/test[self.Y_label] )
            daterange = pd.date_range(pd.to_datetime(self.date_debut), pd.to_datetime(self.date_debut) + pd.DateOffset(1), freq= "H")
            
            test.index = daterange[:-1]
                
            if not os.path.exists(self.directory_data + 'Output/'):
                    os.makedirs(self.directory_data + 'Output/')
            try:    
                plt.figure()
                test[['Predictions', self.Y_label]].plot(figsize= (10,6))
                plt.title("Predictions entre J=" + str(self.date_debut) + " et J+ 1")
                plt.xlabel('Jours')
                plt.ylabel('Consommation')
                plt.gcf().savefig(self.directory_data + 'Output/1day_prediction_' + str(self.date_debut) +'.png')
                    
            except Exception:
                print("prediction mode online")
                plt.figure()
                test[['Predictions']].plot(figsize= (10,6))
                plt.title("Predictions entre J=" + str(self.date_debut) + " et J+ 1")
                plt.xlabel('Jours')
                plt.ylabel('Consommation')
                plt.gcf().savefig(self.directory_data + 'Output/1day_prediction_' + str(self.date_debut) +'.png')
                pass
            
            return test[["Date", "Heure" , "Predictions"]]
            
##### Main
import calendar          
daterange = pd.date_range(pd.to_datetime("2015-01-01"), pd.to_datetime("2016-01-01"), freq= "D")
directory = 'C:/Users/Alexandre/Documents/Antonin project/Monaco Predictions/Modele J+1/'   
mape_mois = 0
rmse_mois=0

for day in daterange:
    
    if day.date().day == 1:
        tt = J_1_Prediction(directory_data = directory, directory_modele= directory, date_debut = str(day.date()), train_test = "train", build_data = "no")
        rmse_mois =rmse_mois+ tt.RMSE
        mape_mois =mape_mois+ tt.MAPE
        end = calendar.monthrange(day.date().year,day.date().month)
        print(end[1])
        
    if day.date().day != 1:
        tt = J_1_Prediction(directory_data = directory, directory_modele= directory, date_debut = str(day.date()), train_test = "test", build_data = "no")
        rmse_mois = rmse_mois+ tt.RMSE
        mape_mois =mape_mois+ tt.MAPE
        
    if day.date().day==end[1]:  
        print("RMSE total du mois " + str(day.date().month) + "  : "+str(rmse_mois/end[1])+ " ; MAPE total : " + str(mape_mois/end[1]))
        mape_mois = 0
        rmse_mois=0