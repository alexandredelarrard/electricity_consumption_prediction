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
import gc
from dateutil.relativedelta import relativedelta
import datetime
import cPickle as pickle

np.random.seed(1304)
warnings.filterwarnings('ignore')

class J_14_Prediction(object):
    
    def __init__(self, directory_data, directory_modele , date_debut, train_test, build_data, nombre_jours_futurs=14):
        
        self.nombre_jours_futurs = nombre_jours_futurs + 1
        self.data = {}
        self.resultats_KPI=  pd.DataFrame([] , columns= ["Date", "jour semaine", "RMSE" , "MAPE"], index = range(1, self.nombre_jours_futurs))
        self.results = []
        self.directory_data = directory_data
        self.directory_modele = directory_modele
        
        #### 
        if build_data.upper()== "YES":
            os.chdir(self.directory_modele + "Scripts/")   
            from DataprepModele14J import DataPreparation as dp
            tt = dp(self.directory_data + "Data/", self.nombre_jours_futurs) 
            self.data = tt.prepared
            
        else:
            if build_data.upper()== "NO":
                for i in range(1,self.nombre_jours_futurs):
                    self.data[i] =  pickle.load(open(self.directory_data + "Data/prepared/data_prepared_pred_conso_" + str(i) +".pkl", "rb"))
            else:
                print("need yes or no values to know if the dataset has to be built or not")
        
        ### hyper parameters of the algorithm 
        self.Y_label = "Consommation"
        self.loss = "ls"
        self.learning_rate =  np.array(pd.DataFrame([np.ones(24)*0.02] , columns= range(24), index = range(1, self.nombre_jours_futurs)))
        self.learning_rate[:,0:2] =0.016
        self.learning_rate[:,2:5] =0.017
        self.learning_rate[:,15] = 0.03
        self.learning_rate[:,17] = 0.025
        self.learning_rate[:,18] = 0.03
        self.learning_rate[:,19] = 0.025
        self.learning_rate[:,20] = 0.028
        
        ## space to train and test the algorithm
        
        if train_test.upper() =="TRAIN":
            self.sortie = self.Apprentissage_14Jours()
            
        else: 
            if train_test.upper() =="TEST":
                 self.sortie = self.predicions_14jours()
                
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
    
    def Apprentissage_14Jours(self):
        
        reponse = pd.DataFrame()
        for i in range(1, self.nombre_jours_futurs):
            print("\n")
            print("prediction du jour J + " + str(i))
            rep = self.GBM(i)
            reponse = pd.concat([reponse, rep], axis=0)
            gc.collect()
            
        daterange = pd.date_range(pd.to_datetime(self.date_debut), pd.to_datetime(self.date_debut) + pd.DateOffset(self.nombre_jours_futurs -1), freq= "H")
        
        try: 
            reponse["Consommation"] = self.data.loc[(self.data.Date >= self.date_debut) & (self.data.Date< str(pd.to_datetime(pd.to_datetime(self.date_debut).to_datetime() + relativedelta(days=self.nombre_jours_futurs-1))) ), "Consommation" ].reset_index(drop=True)
            
        except Exception:
            print("prediction mode online")
            pass
        
        reponse.index= daterange[:-1]
        self.results = reponse
        
        plt.figure()
        reponse.plot(figsize= (10,6))
        plt.title("Predictions entre J=" + str(self.date_debut) + " et J+ " + str(self.nombre_jours_futurs -1))
        plt.xlabel('Jours')
        plt.ylabel('Consommation')
        plt.ylim([40000,90000])
        plt.savefig(self.directory_data + 'Output/14days_prediction_' + str(self.date_debut) +'.png')
        
        
    def GBM(self, jour):   
            
        datas = self.data[jour].copy()
        del datas.annee
        
        ### test = split train test from begining date
        test = datas[ pd.to_datetime(datas.Date) == pd.to_datetime(self.date_debut).to_datetime() + datetime.timedelta(days=jour-1)  ].copy()
        train = datas[(datas.Date < self.date_debut)].copy()
        test['Predictions'] = 0

        for h in range(0,24):
                    
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
                              "eta": self.learning_rate[jour-1,h],
                              "max_depth": 5,
                              "subsample": 0.86,
                              "colsample_bytree": 0.9,
                              "colsample_bylevel" : 0.9,
                              "silent": 1,
                              "seed": 1301}
                              
                clf =xgb.train(params, train_i_M, 1200)   
                prediction_bis = clf.predict(test_i_M)   
                
                # save model    
                if not os.path.exists(self.directory_data + "Trained_models/jour_" + str(jour)):
                    os.makedirs(self.directory_data + "Trained_models/jour_" + str(jour))
                clf.save_model(self.directory_data +  "Trained_models/jour_" + str(jour) + "/model_"+ str(h) + '.model')
                    
                #### calculus error MAPE/RMSE
                rmse =  np.sqrt(((test1[self.Y_label]- np.exp(prediction_bis))**2).mean())
                mape = np.mean(abs(test1[self.Y_label]- np.exp(prediction_bis))/test1[self.Y_label] )
                
                print("Heure : %.2f, error rmse: %.3f, erreur mape: %.3f" %(h, rmse, mape))
                
                test['Predictions'][test['Heure'] == h]  = np.exp(prediction_bis)
        
        test = test.reset_index(drop= True)
        print("_"*80)
        print("Date: "+ str(test["Date"].value_counts().index[0]) +"; MAPE = " + str(np.mean(abs(test[self.Y_label]-  test['Predictions'])/test[self.Y_label] )) + "; RMSE = " + str(np.sqrt(((test[self.Y_label]- test['Predictions'])**2).mean())))
        
        #### results display
        self.resultats_KPI["Date"][jour]  = (pd.to_datetime(self.date_debut) + pd.DateOffset(jour))
        self.resultats_KPI["jour semaine"][jour] = (pd.to_datetime(self.date_debut) + pd.DateOffset(jour)).dayofweek
        self.resultats_KPI["RMSE"][jour] = np.sqrt(((test[self.Y_label]- test['Predictions'])**2).mean())
        self.resultats_KPI["MAPE"][jour] = np.mean(abs(test[self.Y_label]-  test['Predictions'])/test[self.Y_label] )
        
        return test['Predictions']
        self.plots(train1, clf, jour)
        
    def predicions_14jours(self):
        
        print("\n")
        print("_"*80)
        print(" test phase: prediction de " + str(self.nombre_jours_futurs-1) + " J futurs par pas horaire \n")
        print("_"*80)
        
        reponse = pd.DataFrame([])
        
        liste_jours= ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
        
        for jour in range(1,self.nombre_jours_futurs):
            print("prediction de "+ liste_jours[(pd.to_datetime(self.date_debut, format='%Y-%m-%d') + pd.DateOffset(jour-1)).dayofweek] + " : " + str(pd.to_datetime(self.date_debut, format='%Y-%m-%d') + pd.DateOffset(jour-1)))
            datas = self.data[jour].copy()
        
            test = datas[ pd.to_datetime(datas.Date) == pd.to_datetime(self.date_debut).to_datetime() + datetime.timedelta(days=jour-1)  ].copy()
            test['Predictions'] = 0
            
            for h in range(0,24):
                test1 = test[(test['Heure']< (h+1))&(test['Heure']>=h)].copy()                   
                test1 = test1.drop(['Heure', 'Date'], axis=1)
                
                clf = xgb.Booster({'nthread':4}) 
                clf.load_model(self.directory_data + "Trained_models/jour_" + str(jour) +"/model_"+  str(h) +".model") 
                    
                prediction_bis = clf.predict(xgb.DMatrix(test1.drop([self.Y_label, 'Predictions'] , axis=1))) 
                test['Predictions'][test['Heure'] == h] = np.exp(prediction_bis)
            
            reponse = pd.concat([reponse, test['Predictions']],axis=0)
            
        reponse.rename(columns={0: "Predictions"}, inplace = True)
        reponse = reponse.reset_index(drop=True)
        daterange = pd.date_range(pd.to_datetime(self.date_debut), pd.to_datetime(self.date_debut) + pd.DateOffset(self.nombre_jours_futurs -1), freq= "H")
        
        try: 
            reponse["Consommation"] = datas.loc[(datas.Date >= self.date_debut) & (datas.Date< str(pd.to_datetime(pd.to_datetime(self.date_debut).to_datetime() + relativedelta(days=self.nombre_jours_futurs-1))) ), "Consommation" ].reset_index(drop=True)
            
        except Exception:
            print("prediction mode online")
            pass
        
        reponse.index= daterange[:-1]
        self.results = reponse
        
        plt.figure()
        reponse.plot(figsize= (10,6))
        plt.title("Predictions entre J=" + str(self.date_debut) + " et J+ " + str(self.nombre_jours_futurs -1))
        plt.xlabel('Jours')
        plt.ylabel('Consommation')
        plt.ylim([40000,90000])
        plt.savefig(self.directory_data + 'Output/14days_prediction_' + str(self.date_debut) +'.png')
        
#### Main
directory = 'C:/Users/Alexandre/Documents/Antonin project/Monaco Predictions/Modele J+14/'        
tt = J_14_Prediction(directory_data = directory, directory_modele = directory, date_debut = "2015-01-01", train_test = "train", build_data = "no", nombre_jours_futurs =14)
for j in range(1,32):
    tt = J_14_Prediction(directory_data = directory, directory_modele = directory, date_debut = "2015-01-" + str(j), train_test = "test", build_data = "no", nombre_jours_futurs =14)
    
tt = J_14_Prediction(directory_data = directory, directory_modele = directory, date_debut = "2015-02-01", train_test = "train", build_data = "no", nombre_jours_futurs =14)
for j in range(1,29):
    tt = J_14_Prediction(directory_data = directory, directory_modele = directory, date_debut = "2015-02-" + str(j), train_test = "test", build_data = "no", nombre_jours_futurs =14)

tt = J_14_Prediction(directory_data = directory, directory_modele = directory, date_debut = "2015-03-01", train_test = "train", build_data = "no", nombre_jours_futurs =14)
for j in range(1,32):
    tt = J_14_Prediction(directory_data = directory, directory_modele = directory, date_debut = "2015-03-" + str(j), train_test = "test", build_data = "no", nombre_jours_futurs =14)

tt = J_14_Prediction(directory_data = directory, directory_modele = directory, date_debut = "2015-04-01", train_test = "train", build_data = "no", nombre_jours_futurs =14)
for j in range(1,31):
    tt = J_14_Prediction(directory_data = directory, directory_modele = directory, date_debut = "2015-04-" + str(j), train_test = "test", build_data = "no", nombre_jours_futurs =14)

tt = J_14_Prediction(directory_data = directory, directory_modele = directory, date_debut = "2015-05-01", train_test = "train", build_data = "no", nombre_jours_futurs =14)
for j in range(1,32):
    tt = J_14_Prediction(directory_data = directory, directory_modele = directory, date_debut = "2015-05-" + str(j), train_test = "test", build_data = "no", nombre_jours_futurs =14)
    
tt = J_14_Prediction(directory_data = directory, directory_modele = directory, date_debut = "2015-06-01", train_test = "train", build_data = "no", nombre_jours_futurs =14)
for j in range(1,31):
    tt = J_14_Prediction(directory_data = directory, directory_modele = directory, date_debut = "2015-06-" + str(j), train_test = "test", build_data = "no", nombre_jours_futurs =14)
    