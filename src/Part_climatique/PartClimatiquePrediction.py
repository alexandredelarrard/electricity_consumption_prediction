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
from sklearn.linear_model import ElasticNet
from sklearn.externals import joblib

np.random.seed(1304)
warnings.filterwarnings('ignore')

class PartClimatiquePrediction(object):
    
    def __init__(self, directory_data, directory_modele, date_debut, date_fin, temperature_reference, train_test, build_data, number_iteration= 500, depth =5 , learning_rate=0.05):
        
        #### 
        if build_data.upper() =="YES":
            os.chdir(directory_modele + "Scripts/")       
            from DataprepPartClimatique import DataPreparation as dp
            
            self.data = dp(directory_data + "Data/").conso
            
        elif build_data.upper() =="NO":
            ### global parameters with output to predict        
            self.data = pd.read_csv(directory_data + "Data/prepared/data_prepared_pred_conso.csv" , sep= "," , header =0)
        
        else: 
            print("need train or test values")
        
        ### hyper parameters of the algorithm 
        self.Y_label = "Consommation"
        self.number_iteration = number_iteration  
        self.depth = depth
        self.learning_rate = learning_rate
        self.mode= "gbm"
        self.directory_data = directory_data
        self.directory_modele = directory_modele
        
        ### space to train and test the algorithm
        self.date_debut = date_debut
        self.date_fin = date_fin
        self.temp_ref = temperature_reference
        
        if train_test =="train":
            self.sortie = self.GBM()
            
        elif train_test =="test":
            self.sortie = self.predicions_temperature()
                
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
        
        ### test = between first and last day
        test = datas[(pd.to_datetime(datas.Date) >= pd.to_datetime(self.date_debut)) &(pd.to_datetime(datas.Date) < pd.to_datetime(self.date_fin)) ].copy()
        train = datas[(datas.Date < self.date_fin)].copy()
        test['Predictions'] = 0
        
        for h in range(0,24):
                    
                train1 = train[ (train['Heure']< (h+1))&(train['Heure']>=h)].copy()                     
                test1 = test[(test['Heure']< (h+1))&(test['Heure']>=h)].copy()                   
                
                train1 = train1.drop(['Heure', "Date"], axis=1)
                test1 = test1.drop(['Heure', "Date"], axis=1)
                
                if self.mode=="gbm":
                
                    y_train = np.log(train1[self.Y_label])        
                    train_i_M = xgb.DMatrix(train1.drop(self.Y_label, axis=1), label=y_train)
                    
                    y_test = np.log(test1[self.Y_label])        
                    test_i_M = xgb.DMatrix(test1.drop([self.Y_label,'Predictions'] , axis=1), label=y_test)      
                    
                    ####### fitting
                    params = {"objective": "reg:linear",
                              "booster" : "gbtree",
                              "eta": self.learning_rate,
                              "max_depth": self.depth,
                              "subsample": 0.85,
                              "colsample_bytree": 0.85,
                              "colsample_bylevel" : 0.85,
                              "silent": 1,
                              "seed": 1301,
                              "min_child_weight": 1,
                              }
                              
                    n_estimators=self.number_iteration
                    watchlist = [(train_i_M, 'train'), (test_i_M, 'eval')]
                    
                    clf =xgb.train(params, train_i_M, n_estimators, evals=watchlist, \
                    early_stopping_rounds=100, feval= self.evalerror, verbose_eval=False)
                    self.clf = clf       
                    prediction_bis = clf.predict(xgb.DMatrix(test1.drop([self.Y_label,'Predictions'] , axis=1))) 
                    
                    # save model  
                    if not os.path.exists(self.directory_data + "Trained_models"):
                        os.makedirs(self.directory_data + "Trained_models")
                        
                    clf.save_model(self.directory_data +  "Trained_models/model_"+ str(h) + '.model')
                    
                if self.mode=="l1-l2":
                    enet = ElasticNet(alpha=0.0, l1_ratio= 0.0)
                    clf  = enet.fit(train1.drop([self.Y_label], axis=1), np.log(train1[self.Y_label]))
                    prediction_bis = clf.predict(test1.drop([self.Y_label, 'Predictions'], axis=1)) 
                    self.clf = clf
                    
                    # save model
                    joblib.dump(clf, self.directory_data +  "Trained_models/model_"+ str(h) + ".pkl")
                    
                rmse =  np.sqrt(((test1[self.Y_label]- np.exp(prediction_bis))**2).mean())
                mape = np.mean(abs(test1[self.Y_label]- np.exp(prediction_bis))/test1[self.Y_label] )
                print("Heure : %.2f, error rmse: %.3f, erreur mape: %.3f" %(h, rmse, mape))
                
                test['Predictions'][test['Heure'] == h]  = np.exp(prediction_bis)
       
        #### importance variables
        self.plots(test1, clf)
        test[["Predictions", self.Y_label]].plot(figsize=(15,8), title="Consommation entre " + str(self.date_debut) +" et " + str(self.date_fin ))
        
        return self.predicions_temperature()
        
    def predicions_temperature(self):
        
        print(" test phase: prediction de la part climatique par pas horaire \n")
        datas = self.data.copy()
        
        colnames = [x for x in list(self.data.columns) if (x.split("_")[0] =="temperature")]
        for i in colnames:
            datas[i] = self.temp_ref
                     
        test = datas[(pd.to_datetime(datas.Date) >= pd.to_datetime(self.date_debut)) &(pd.to_datetime(datas.Date) < pd.to_datetime(self.date_fin)) ].copy()
        test['Consommation 16 degre'] = 0
        
        for h in range(0,24):
            
            if self.mode == "gbm":
                clf = xgb.Booster({'nthread':4}) #init model
                clf.load_model(self.directory_data + "Trained_models/model_"+ str(h) +".model") 
                
                prediction_bis = clf.predict(xgb.DMatrix(test.drop([self.Y_label,'Consommation 16 degre', "Date", "Heure"] , axis=1))) 
            
            if self.mode == "l1-l2":
                clf = joblib.load(self.directory_data + "Trained_models/model_"+ str(h) +".pkl")
                prediction_bis = clf.predict(test.drop([self.Y_label, "Heure",'Consommation 16 degre'] , axis=1)) 
                
            test['Consommation 16 degre'][test['Heure'] == h] = np.exp(prediction_bis)

        test["Part climatique"] =  test[self.Y_label] - test['Consommation 16 degre']
        test["Part climatique"][test["Part climatique"]<0] =0
        test.index = test.Date
           
        plt.figure()
        test[["Consommation 16 degre","Part climatique", self.Y_label]].plot(figsize=(17,10))
        plt.title("Consommation entre " + str(self.date_debut) +" et " + str(self.date_fin))
        plt.xlabel('Date')
        plt.ylabel('Consommation')
        plt.gcf().savefig(self.directory_data + 'Output/feature_importance_xgb.png')
        plt.close()
        
        return test[["mois", "jour_mois" , "Heure",  self.Y_label, "Consommation 16 degre", "Part climatique"]]

### Main 
directory = 'C:/Users/Alexandre/Documents/Antonin project/Monaco Predictions/Part climatique/'        
tt = PartClimatiquePrediction(directory_data = directory, directory_modele = directory, date_debut = "2015-01-01", date_fin = '2016-01-01', temperature_reference = 16.0, train_test ="train", build_data = "yes")