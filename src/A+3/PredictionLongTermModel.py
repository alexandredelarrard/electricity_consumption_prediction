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
from pandas.stats.api import ols
from sklearn import linear_model
from dateutil.relativedelta import relativedelta

np.random.seed(1304)
warnings.filterwarnings('ignore')

class LT_Prediction(object):
    
    def __init__(self, directory_data, directory_modele, date_debut, train_test, build_data):
        
        self.data =  pd.DataFrame()
        self.directory_data = directory_data
        self.directory_modele = directory_modele
        self.passe = 3
        ####  Option to build the data or not
        if build_data.upper()== "YES":
            os.chdir(self.directory_modele + "Scripts/")   
            from DataprepLongTermModel import DataPreparation as dp
            tt = dp(self.directory_data + "Data/") 
            self.data = tt.conso
            
        else:
            if build_data.upper()== "NO":
              self.data = pd.read_csv(directory + "Data/prepared/data_prepared.csv" , sep= ";" , header =0)
            else:
                print("need yes or no values to know if the dataset has to be built or not")
        
        ### hyper parameters of the algorithm 
        self.Y_label = "Consommation"
        self.loss = "ls"

        ### train and test the algorithm
        self.date_debut = date_debut 
        
        if train_test.upper() =="TRAIN":
            
            self.TrendLT()                 
            test = self.GBM()

            daterange = pd.date_range(pd.to_datetime(self.date_debut), pd.to_datetime(self.date_debut) + relativedelta(years=3) + relativedelta(days=7), freq= "H")
            test.index = daterange[:len(test)]
            test[["Predictions" , self.Y_label]].plot(figsize=(12,7))
            self.sortie = test[["Date", "Heure", "Predictions"]]
            
            RMSE_tot = np.sqrt(((test[self.Y_label]- test['Predictions'])**2).mean())
            mape_totale = np.mean(abs(test[self.Y_label]- test['Predictions'])/test[self.Y_label] )
            self.KPI = RMSE_tot 
            print("_"*80)
            print("RMSE total = " + str(RMSE_tot) + ";   MAPE totale = " + str(mape_totale*100))
            
        elif train_test.upper() =="TEST":
                 print("predictions J+1 à A+3")
                 self.sortie = self.predicions_LT()
                
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
        
    def TrendLT(self):
        
        datas = self.data.copy()
        datas = datas[pd.to_datetime(datas.Date) < pd.to_datetime(self.date_debut) - relativedelta(years=self.passe)]
        datas["annee_carre"] = datas["annee"]**2        

#        model =  ols(y= datas[self.Y_label] , x = datas[[ 'annee', 'annee_carre',  "mois_1.0", "mois_2.0", "mois_3.0", "mois_4.0", "mois_5.0", "mois_6.0", "mois_7.0", "mois_8.0", "mois_9.0", "mois_10.0", "mois_11.0", "mois_12.0"]][datas.Date <= self.date_debut ], datas[self.Y_label][datas.Date <= self.date_debut])
#        print(model)
        lr = linear_model.LinearRegression()
        fitted = lr.fit(datas[['annee' , 'annee_carre']][datas.Date < self.date_debut ], datas[self.Y_label][datas.Date < self.date_debut])
        self.data['annee_carre'] = self.data["annee"]**2
        
        plt.figure(figsize= (12,10))
        results_LT = fitted.predict(self.data[['annee' ,  'annee_carre']])
        plt.plot(figsize=(12,8))
        plt.plot(results_LT, alpha=0.5)
        plt.plot(self.data[self.Y_label], alpha=0.5)
        plt.show()
        
        ##### add long term consumption prediction
        self.data["Consommation_linear_reg"] = results_LT        
#        del self.data['annee_carre']
#        
#        ### add average consumption per day of year of 3,4,5,6 years
#        for years in [6]:
#            self.data["Consommation_mean_A-"+ str(years)] =0
#            print(years)
#            group_3y = self.data[["jour_annee", "Consommation"]][(pd.to_datetime(self.data.Date) >= pd.to_datetime(self.date_debut) - relativedelta(years=years))&(pd.to_datetime(self.data.Date) < pd.to_datetime(self.date_debut))].groupby("jour_annee").mean()
#            group_3y.rename(columns= {"Consommation" : "Consommation_mean_A-"+ str(years)}, inplace= True)
#            group_3y["jour_annee"] = group_3y.index
#            group_3y = group_3y.reset_index(drop=True)
#            
#            datas_train = self.data[(pd.to_datetime(self.data.Date) >= pd.to_datetime(self.date_debut) - relativedelta(years=self.passe + years))&(datas.Date< str(pd.to_datetime(pd.to_datetime(self.date_debut).to_datetime() - relativedelta(years=self.passe))))]
#            datas_train["Consommation_mean_A-" + str(years)] = 0
#            
#            for i in range(years):
#                for jour in range(367):
#                    me_jr = datas_train.loc[datas_train["jour_annee"] == jour, "Consommation"]
#                    me_jrs = np.sqrt(me_jr.loc[len(me_jr)-years:].min())
#                    me_jrm = np.sqrt(me_jr.loc[len(me_jr)-years:].max())
#                    self.data.loc[self.data["jour_annee"] == jour, "Consommation_min_A-" + str(years)] = float(me_jrs)
#                    self.data.loc[self.data["jour_annee"] == jour, "Consommation_max_A-" + str(years)] = float(me_jrm)       
   
    def GBM(self):   
        
        rmse_totale = 0
        self.learning_rate = [0.005, 0.006, 0.006, 0.007, 0.007, 0.007, 0.008, 0.008, 0.008,
                             0.006, 0.006, 0.006, 0.006, 0.006, 0.006, 0.006, 0.006,
                             0.006, 0.006, 0.006, 0.006, 0.006, 0.006, 0.006]
  
        datas = self.data.copy()
            
        datas = datas[(datas.Date>= str(pd.to_datetime(pd.to_datetime(self.date_debut).to_datetime() - relativedelta(years=self.passe))))]
            
        ### test = split train test from begining date
        train = datas[datas.Date < self.date_debut ].copy()
        test = datas[(datas.Date >= self.date_debut) & (datas.Date< str(pd.to_datetime(pd.to_datetime(self.date_debut).to_datetime() + relativedelta(years=self.passe))) ) ].copy()
        train = train.drop(["annee"],axis=1)   
        test= test.drop(["annee"],axis=1)  
        
        test['Predictions'] = 0

        for h in range(0,24):
                    
                train1 = train[ (train['Heure']< (h+1))&(train['Heure']>=h)].copy()                     
                test1 = test[(test['Heure']< (h+1))&(test['Heure']>=h)].copy() 
                        
                train1 = train1.drop(['Heure',"Date"], axis=1)
                test1 = test1.drop(['Heure',"Date"], axis=1)
                
                y_train = np.log(train1[self.Y_label])        
                train_i_M = xgb.DMatrix(train1.drop(self.Y_label, axis=1), label=y_train)
                    
                y_test = np.log(test1[self.Y_label])        
                test_i_M = xgb.DMatrix(test1.drop([self.Y_label,'Predictions'] , axis=1), label=y_test)      
                    
                ####### fitting
                params = {"objective": "reg:linear",
                              "booster" : "gbtree",
                              "eta": self.learning_rate[h],
                              "max_depth": 4,
                              "subsample": 0.8,
                              "colsample_bytree": 0.8,
                              "colsample_bylevel" : 0.8,
                              "silent": 1,
                              "eval_metric":"rmse",
                              "seed": 1301}
                              
                clf =xgb.train(params, train_i_M, 1200)   
                prediction_bis = clf.predict(test_i_M) 
                
                # save model    
                if not os.path.exists(self.directory_data + "Trained_models"):
                    os.makedirs(self.directory_data + "Trained_models")
                    
                clf.save_model(self.directory_data +  "Trained_models/modelLT_"+ str(h) + '.model')
                    
                #### calculus error MAPE/RMSE
                rmse =  np.sqrt(((test1[self.Y_label]- np.exp(prediction_bis))**2).mean())
                mape = np.mean(abs(test1[self.Y_label]- np.exp(prediction_bis))/test1[self.Y_label] )
                print("Heure : %.2f, error rmse: %.3f, erreur mape: %.3f " %(h, rmse, mape*100))
                
                test['Predictions'][test['Heure'] == h]  = np.exp(prediction_bis)
        
        test = test.reset_index(drop= True)
        
        rmse_totale =  np.sqrt(((test[self.Y_label]- test['Predictions'])**2).mean())
        self.KPI = rmse_totale
               
       ### Plot importance error
        self.plots(train1, clf)
     
        return test
        
    def predicions_LT(self):
        
        print(" test phase: prediction de 14 J futurs par pas horaire \n")
        
        datas = self.data.copy()
        
        test = datas[(datas.Date >= self.date_debut) & (datas.Date< str(pd.to_datetime(pd.to_datetime(self.date_debut).to_datetime() + relativedelta(years=self.passe))) ) ].copy()
        test['Predictions'] = 0
        
        test = test.drop(["annee"], axis=1)
            
        for h in range(0,24):
                test1 = test[(test['Heure']< (h+1))&(test['Heure']>=h)].copy()                   
                test1 = test1.drop('Heure', axis=1)
                
                clf = xgb.Booster({'nthread':4}) #init model
                clf.load_model(self.directory_data + "Trained_models/modelLT_"+  str(h) +".model") 
                    
                prediction_bis = clf.predict(xgb.DMatrix(test1.drop([self.Y_label, 'Predictions', "Date"] , axis=1))) 
                test['Predictions'][test['Heure'] == h] = np.exp(prediction_bis)
        
        daterange = pd.date_range(pd.to_datetime(self.date_debut), pd.to_datetime(self.date_debut) + relativedelta(years=self.passe), freq= "H")
        test = test[:-23]
        test.index = daterange
        
        try:
            test[["Predictions", self.Y_label]].plot(figsize = (12,7))
            
        except Exception:
            print("prediction mode online")
            
            test[["Predictions"]].plot(figsize = (12,7))
            pass
        
        return test
        
#### Main 
directory = 'C:/Users/Alexandre/Documents/Antonin project/Monaco Predictions/Modele long terme/'        
tt = LT_Prediction(directory_data= directory, directory_modele= directory, date_debut = "2009-01-01", train_test = "train", build_data = "no")
