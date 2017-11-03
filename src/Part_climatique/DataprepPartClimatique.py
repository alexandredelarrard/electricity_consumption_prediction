# -*- coding: utf-8 -*-
"""
Created on Thu Jul 07 19:48:55 2016

@author: alexandre

this dataprep intend to predict consumption of electricity for a 15 h lag minimum. 
3 horaire Temperature data, calandaire + Var client, historique consommation par pas de 10 minutes, moyennés à l'horaire, lever et coucher du solail.  

"""


import numpy as np
import pandas as pd
import os
import warnings
import glob
import time

warnings.filterwarnings('ignore')

class DataPreparation(object):
   
    def __init__(self, directory):
        
        os.chdir(directory)
        self.names_data =[]
        self.conso = pd.DataFrame()
        self.directory = directory
        self.data ={}
        
        #### data prep functions
        self.Import()
        self.PreparationElectricite()
        
    def Import(self):    
        
        files = glob.glob("*.txt")
        
        liste_date=pd.DataFrame([["Historique_consommation_SMEG_",3000, 3],["Sun_SMEG_",3000,3],["Temp_SMEG_",3000,3],["TypeJours_SMEG_",3000,3]], columns=['Type_data', 'Min_year', 'Max_year']) 
        self.names_data=["Historique_consommation_SMEG_", "Sun_SMEG_","Temp_SMEG_","TypeJours_SMEG_"]
        
        ###### check if all dataset have the same time length (number of years)
        ###### keep only years where all data are available
        
        for f in files:
            f = f.replace('.txt' ,'')
            year = int(f.split('_', 3)[len(f.split('_', 3))-1])
            if int(year) > int(np.array(liste_date[liste_date["Type_data"] == f[:-4]]['Max_year'])[0]):
                liste_date.ix[liste_date["Type_data"] ==f[:-4],'Max_year'] = year
            
            if int(year) < int(np.array(liste_date[liste_date["Type_data"] == f[:-4]]['Min_year'])[0]):
                liste_date.ix[liste_date["Type_data"] == f[:-4],'Min_year'] = year
        
        end_year = min(liste_date['Max_year'])
        start_year = max(liste_date['Min_year']) 
        
        print(" Apprentissage commence en année " + str(start_year) + "; s'arrête en année " + str(end_year))
        print("_"*80)
        print(liste_date)
        print("_"*80)
        
        if end_year < max(liste_date['Max_year']):
            print("real maximum year is " +  str(max(liste_date['Max_year'])))
            
        if start_year > min(liste_date['Min_year']):
            print("real minimum year is " +  str(min(liste_date['Min_year'])))
            
        #### initialization of dictonnary of datasets
        for i in self.names_data:
            self.data[i] = pd.DataFrame()        
        
        for i in range(start_year, end_year +1):
            for j in self.names_data:
                db = pd.read_csv(self.directory + str(j) + str(i) + '.txt', sep="  ", header=None)                
                self.data[j] = pd.concat([self.data[j], db] , axis=0)
                self.data[j] = self.data[j].reset_index(drop=True)
    
    def DirectoryCreation(self, pathDir):
        if not os.path.exists(pathDir):
            os.makedirs(pathDir)
        return None    
        
    def PreparationElectricite(self):
    
        lj = ["Jour" in i for i in self.names_data]
        index_jour = lj.index(True)
        
        lt = ["Temp" in i for i in self.names_data]
        index_temp = lt.index(True)
        
        lm = ["Sun" in i for i in self.names_data]
        index_sun = lm.index(True)
        
        lr = ["Histo" in i for i in self.names_data]   
        index_ref = lr.index(True)
        
        ########  if number of cunsumption columns  = self.step_size and I want to mean them into
        ########  step_mean number of columns: ex: 6 columns and I want a half hour prediction -> 6 columns into 2        
        
        self.data[self.names_data[index_ref]][2] = self.data[self.names_data[index_ref]].loc[:, list(np.arange(2,3+ 6))].mean(axis=1)
        self.data[self.names_data[index_ref]] = self.data[self.names_data[index_ref]].drop(list(np.arange(3,len(self.data[self.names_data[index_ref]].columns))),axis=1)
      
        self.data[self.names_data[index_ref]].rename(columns={0: 'Date', 1: 'Heure', 2: 'Consommation'}, inplace=True)
    
         ##### gestion de la date  
        self.data[self.names_data[index_ref]]['Heure'] = self.data[self.names_data[index_ref]]['Heure'].str.replace(':3',':5')
        self.data[self.names_data[index_ref]]['Heure'] = self.data[self.names_data[index_ref]]['Heure'].str.replace(':','.').astype(float)
        self.data[self.names_data[index_ref]].loc[self.data[self.names_data[index_ref]]['Heure'] ==24.0, 'Heure']  = 0.0        
        
        self.data[self.names_data[index_ref]]["Date"] = pd.to_datetime(self.data[self.names_data[index_ref]]["Date"], format = '%d/%m/%Y')      
        self.data[self.names_data[index_ref]]['annee'] = self.data[self.names_data[index_ref]]['Date'].dt.year
        self.data[self.names_data[index_ref]]['mois'] = self.data[self.names_data[index_ref]]['Date'].dt.month
        self.data[self.names_data[index_ref]]['jour_mois'] = self.data[self.names_data[index_ref]]['Date'].dt.day  
        self.data[self.names_data[index_ref]]['jour_annee']= self.data[self.names_data[index_ref]]["Date"].dt.dayofyear
        self.data[self.names_data[index_ref]]['jour_semaine']= self.data[self.names_data[index_ref]]["Date"].dt.dayofweek
    
        #####################" ajout columns sun rise and down  a bdd consommation + gestion jour semaine
        self.data[self.names_data[index_sun]].rename(columns={0:'Date', 1:'Rise', 2:'Down', 3:'Last'},inplace=True)
        
        self.data[self.names_data[index_sun]]['Date'] =   pd.to_datetime(self.data[self.names_data[index_sun]]['Date'] , format = '%Y-%m-%d')
        self.data[self.names_data[index_sun]]['Rise'] = self.data[self.names_data[index_sun]]['Rise'].str.replace(':','.').astype(float)
        self.data[self.names_data[index_sun]]['Down'] = self.data[self.names_data[index_sun]]['Down'].str.replace(':','.').astype(float)
        self.data[self.names_data[index_sun]]['Last'] = self.data[self.names_data[index_sun]]['Last'].str.replace(':','.').astype(float)
    
        self.data[self.names_data[index_ref]] = pd.merge(self.data[self.names_data[index_ref]], self.data[self.names_data[index_sun]] , on = 'Date')
        self.data[self.names_data[index_ref]]  = self.data[self.names_data[index_ref]].drop_duplicates(["Date" , "Heure"])
        
        #####################" ajout columns index_jour a bdd consommation
       
        self.data[self.names_data[index_ref]]['Aout'] =  self.data[self.names_data[index_jour]][2]
        self.data[self.names_data[index_ref]]['Noel'] =  self.data[self.names_data[index_jour]][3]
        self.data[self.names_data[index_ref]]['Feriers'] =  self.data[self.names_data[index_jour]][4]
        self.data[self.names_data[index_ref]]['Vacances'] =  self.data[self.names_data[index_jour]][5]
        self.data[self.names_data[index_ref]]['Var'] =  self.data[self.names_data[index_jour]][6]
        self.data[self.names_data[index_ref]]['temperature'] =  self.data[self.names_data[index_temp]][2].reset_index(drop=True)
        
        self.data[self.names_data[index_ref]] = self.data[self.names_data[index_ref]].reset_index(drop=True)
        
        #### gestion pont et proba d'être en vacances
        
        self.data[self.names_data[index_ref]]["Proba_out_of_work"] = 0
        self.data[self.names_data[index_ref]].loc[self.data[self.names_data[index_ref]]["jour_semaine"] == 6,"Proba_out_of_work" ]= 100
        self.data[self.names_data[index_ref]].loc[self.data[self.names_data[index_ref]]["jour_semaine"] == 5, "Proba_out_of_work" ] = 80
        self.data[self.names_data[index_ref]].loc[self.data[self.names_data[index_ref]]['Noel']>0, "Proba_out_of_work"] = self.data[self.names_data[index_ref]]['Noel']
        self.data[self.names_data[index_ref]].loc[self.data[self.names_data[index_ref]]['Aout']>0, "Proba_out_of_work"] = self.data[self.names_data[index_ref]]['Aout']
        self.data[self.names_data[index_ref]].loc[self.data[self.names_data[index_ref]]['Feriers']>0, "Proba_out_of_work"] = 100
        self.data[self.names_data[index_ref]].loc[self.data[self.names_data[index_ref]]['Vacances']>0, "Proba_out_of_work"] = self.data[self.names_data[index_ref]]['Vacances']
        
        #### Creation de la variable j-1 | j | j+1 d'un jour hors travail
      
        feriers = self.data[self.names_data[index_ref]]["Proba_out_of_work"].copy()
        feriers[feriers>0] = 1
        local_lag_ferier = feriers.shift(-24)
        local_plus_lag_ferier = feriers.shift(24)
        
        ##### 50% de chance de ne pas travailler pour un pont
        z = local_lag_ferier  + local_plus_lag_ferier + feriers 
        self.data[self.names_data[index_ref]].loc[z==1,"Proba_out_of_work"] = 50
        
        self.data[self.names_data[index_ref]]["Pont"] = 0
        self.data[self.names_data[index_ref]].loc[z==1,"Pont"]=1
        
        ####   Creation nouvelles base de donnée avec imbrication des horaires non entier ##########        
        self.data[self.names_data[index_ref]] = self.data[self.names_data[index_ref]].drop(self.data[self.names_data[index_ref]].columns[list(np.arange(3, 1 + 2))], axis=1)
        
        self.data[self.names_data[index_ref]] = self.data[self.names_data[index_ref]].sort(columns=['annee', 'jour_annee', 'Heure'])
        self.data[self.names_data[index_ref]] = self.data[self.names_data[index_ref]].reset_index(drop=True)
        
        self.PreparationLag(self.data[self.names_data[index_ref]])
        print(list(self.conso.columns))
        print("\n")
        
        self.DirectoryCreation(self.directory  + "prepared\\")
        
        print(" jour +1 data prep has ended successfully ; saved in" + self.directory  + "\n")        
        self.conso.to_csv(self.directory + "/prepared/data_prepared_pred_conso.csv", index= False)
        
    def PreparationLag(self, conso):
        
        ##### gestion des lags de temperature
        ##### 2 cas de figure: températures trihoraires ou horaire ou températures quotidiennes.....
           
        for i in range(1,8):
            conso['temperature_Lag'+ str(i*3)] =conso['temperature'].shift(3*i)
        
        conso['temperature_Lag'+ str(48)] = conso['temperature'].shift(48)
        conso['temperature_Lag'+ str(51)] = conso['temperature'].shift(51) 
        conso['temperature_Lag'+ str(96)] = conso['temperature'].shift(96)
        conso['temperature_Lag'+ str(99)] = conso['temperature'].shift(99)    
           
        self.conso = conso.sort(['annee','jour_annee']).reset_index(drop=True)
    
        ListDf=[]
        for i in ["mois", "jour_semaine"]:
            ListDf.append(pd.get_dummies(self.conso[i].astype(str),prefix=i))
        self.conso = pd.concat([self.conso, pd.concat(ListDf, axis=1)], axis=1)
        
        #### is there nan
        for i in self.conso.columns:
            if np.any(pd.isnull(self.conso[i])) ==True:
                self.conso = self.conso[~pd.isnull(self.conso[i])]
#        
#t0 = time.time()
#directory = 'C:/Users/Alexandre/Documents/Antonin project/Monaco Predictions/Part climatique/'        
#dd =  DataPreparation(directory+ "Data/") 
#print("execution time : " + str(time.time() - t0) ) 