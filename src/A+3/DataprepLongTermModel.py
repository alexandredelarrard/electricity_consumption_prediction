# -*- coding: utf-8 -*-
"""
Created on Thu Jul 07 19:48:55 2016

@author: alexandre

this dataprep intend to predict consum
"""
import numpy as np
import pandas as pd
import os
import warnings
import glob
from datetime import date
import datetime
import gc

from pandas.stats.api import ols
from sklearn import linear_model
from scipy.signal import savgol_filter

warnings.filterwarnings('ignore')

class DataPreparation(object):
   
    def __init__(self, directory):
        
        os.chdir(directory)
        self.names_data =[]
        self.directory = directory
        self.data ={}
        self.conso = pd.DataFrame()

        #### data prep functions
        self.Import()
        self.PreparationElectricite() 
#        self.Temperature_modeling()
        self.conso.to_csv(self.directory + "/prepared/data_prepared.csv", index = False,sep=";")
        
    def datepaques(self, an):
        """Calcule la date de Pâques d'une année donnée an (=nombre entier)"""
        a=an//100
        b=an%100
        c=(3*(a+25))//4
        d=(3*(a+25))%4
        e=(8*(a+11))//25
        f=(5*a+b)%19
        g=(19*f+c-e)%30
        h=(f+11*g)//319
        j=(60*(5-d)+b)//4
        k=(60*(5-d)+b)%4
        m=(2*j-k-g+h)%7
        n=(g-h+m+114)//31
        p=(g-h+m+114)%31
        jour=p+1
        mois=n
        
        return jour, mois, an 
        
    def Import(self): 
        def IsLeapYear(year):
            return (year & 3 == 0)&((year % 100 != 0)|(year % 400==0))
        
        files = glob.glob("*.txt")
        liste_type=[]
        
        for f in files:
            f = f.split("SMEG_")[:-1]
            liste_type.append(f[0] + "SMEG_")
        
        self.names_data= list(set(liste_type))
        liste_date = []
        
        for i in range(len(self.names_data)):
            liste_date.append([self.names_data[i] , 3000, 3])
        
        liste_date=pd.DataFrame(liste_date, columns=['Type_data', 'Min_year', 'Max_year']) 
        
        ###### check if all dataset have the same time length (number of years)
        ###### keep only years where all data are available
        
        for f in files:
            f = f.replace('.txt' ,'')
            year = int(f.split('_', 3)[len(f.split('_', 3))-1])
            if int(year) > int(np.array(liste_date[liste_date["Type_data"] == f[:-4]]['Max_year'])[0]):
                liste_date.ix[liste_date["Type_data"] ==f[:-4],'Max_year'] = year
            
            if int(year) < int(np.array(liste_date[liste_date["Type_data"] == f[:-4]]['Min_year'])[0]):
                liste_date.ix[liste_date["Type_data"] == f[:-4],'Min_year'] = year
        
        self.end_year = min(liste_date['Max_year'])
        self.start_year = max(liste_date['Min_year']) 
        
        print(" Apprentissage commence en année " + str(self.start_year) + "; s'arrête en année " + str(self.end_year))
        print("_"*80)
        print(liste_date)
        print("_"*80)
        
        #### creation database de consommation pour le futur
        if liste_date.loc[liste_date["Type_data"] == "Historique_consommation_SMEG_", 'Max_year'].values[0]<date.today().year  +3:
            for  i in range(1, date.today().year  +3 - liste_date.loc[liste_date["Type_data"] == "Historique_consommation_SMEG_", 'Max_year'].values[0] +1):
                
                ## si annee bissextile
                if IsLeapYear(date.today().year +1) == True:
                    rangedate = pd.date_range(str(date.today().year +i) + "/01/01", periods=8784, freq='H')
                
                else:
                    rangedate = pd.date_range(str(date.today().year +i) + "/01/01", periods=8760, freq='H')
            
                rangedate =rangedate.map(lambda t: t.strftime('%d/%m/%Y  %H:%M'))
                Historique_consommation_1 = pd.DataFrame(rangedate)
                
                for j in range(6):
                    Historique_consommation_1["0" + str(j)] = 0
                    
                Historique_consommation_1.to_csv(self.directory + "Historique_consommation_SMEG_" + str(date.today().year + i) + ".txt", sep=" ", index= False, header=False)
            
            self.end_year = self.end_year +3
            
        if self.end_year < max(liste_date['Max_year']):
            print("real maximum year is " +  str(max(liste_date['Max_year'])))
            
        if self.start_year > min(liste_date['Min_year']):
            print("real minimum year is " +  str(min(liste_date['Min_year'])))
            
        #### initialization of dictonnary of datasets
        for i in self.names_data:
            self.data[i] = pd.DataFrame()        
        
        for i in range(self.start_year, self.end_year +1):
            for j in self.names_data:
                sep = '\s+'
                if j in ["Sun_SMEG_", "Temp_SMEG_"]:
                    sep = ";"
                db = pd.read_csv(self.directory + str(j) + str(i) + '.txt', sep=sep, header=None)                
                self.data[j] = pd.concat([self.data[j], db] , axis=0)
                self.data[j] = self.data[j].reset_index(drop=True)
         
    def PreparationElectricite(self):
#
#        lj = ["Jour" in i for i in self.names_data]
#        index_jour = lj.index(True)
        
        lt = ["Temp" in i for i in self.names_data]
        index_temp = lt.index(True)
        
        lm = ["Sun" in i for i in self.names_data]
        index_sun = lm.index(True)
        
        lr = ["Histo" in i for i in self.names_data]   
        index_ref = lr.index(True)
        
        data_part1 = self.data[self.names_data[index_ref]].loc[~pd.isnull(self.data[self.names_data[index_ref]][7]),:]
        data_part2 = self.data[self.names_data[index_ref]].loc[pd.isnull(self.data[self.names_data[index_ref]][7]), :]

        ##### partie avec données toutes les 10 min
        if len(data_part1)>0:
            print("longueur data part 1 " + str(len(data_part1)) )
            data_part1[2] =  data_part1.loc[:, list(np.arange(2,3+ 5))].mean(axis=1)
            data_part1 =  data_part1.drop( list(np.arange(3,3+ 5)) , axis= 1)
            data_part1.rename(columns={0: 'Date', 1: 'Heure', 2: 'Consommation'}, inplace=True)
        else:
            print("longueur data part 1 =0")
        
        ###### partie avec données toutes les demies heures
        if len(data_part2)>0:       
            print("longueur data part 2 " + str(len(data_part2)) )
            cols= []
            for i in data_part2.columns:
                if np.any(pd.isnull(data_part2[i])) ==False:
                    cols.append(i)
        
            data_part2 = data_part2[cols]
            
            data_part2.rename(columns={0: 'Date', 1: 'Heure', 2: 'Consommation'}, inplace=True)
            final_value = data_part2['Consommation'][len(data_part2)-1]
    
            conso_lag1=  data_part2['Consommation'].shift(-1)
            
            data_part2['Consommation'] = (data_part2['Consommation'] +  conso_lag1)/float(2)
            data_part2 = data_part2.loc[data_part2['Heure'].str[3:5] == "00", :]
            data_part2.loc[ data_part2['Heure'] == "24:00" , 'Heure'] = "00:00"
            data_part2.loc[data_part2.index == max(data_part2.index), 'Consommation']= (final_value + data_part1['Consommation'].reset_index(drop=True)[0])/float(2)
        else:
            print("longueur data part 2 =0")
            
        if (len(data_part2)>0)&(len(data_part1)>0):
            self.data[self.names_data[index_ref]] = pd.concat([data_part2 ,data_part1], axis= 0)
        else:
            if (len(data_part2)==0):
                self.data[self.names_data[index_ref]] =data_part1
            else:
                self.data[self.names_data[index_ref]] =data_part2
                
        #########################################
        ##### gestion de la date  
        self.data[self.names_data[index_ref]]['Heure'] = self.data[self.names_data[index_ref]]['Heure'].str.replace(':3',':5')
        self.data[self.names_data[index_ref]]['Heure'] = self.data[self.names_data[index_ref]]['Heure'].str.replace(':','.').astype(float)
        self.data[self.names_data[index_ref]]['Heure'][self.data[self.names_data[index_ref]]['Heure'] ==24.0]  = 0.0        
        
        self.data[self.names_data[index_ref]]['annee'] = self.data[self.names_data[index_ref]]['Date'].str[6:10].astype(float)
        self.data[self.names_data[index_ref]]['mois'] = self.data[self.names_data[index_ref]]['Date'].str[3:5].astype(float)
        self.data[self.names_data[index_ref]]['jour_mois'] = self.data[self.names_data[index_ref]]['Date'].str[:2].astype(float)        
          
        self.data[self.names_data[index_ref]]["Date"] = pd.to_datetime(self.data[self.names_data[index_ref]]["Date"], format = '%d/%m/%Y')      
        self.data[self.names_data[index_ref]]['jour_annee']= self.data[self.names_data[index_ref]]["Date"].dt.dayofyear
        
        #########################################
        # ######## sun check
        
        self.data[self.names_data[index_sun]].rename(columns={0:'Date', 1:'Rise', 2:'Down', 3:'Last'},inplace=True)
        date = pd.to_datetime(self.data[self.names_data[index_sun]]['Date'])
        self.data[self.names_data[index_sun]]['jour_semaine'] = date.dt.dayofweek
                
        self.data[self.names_data[index_sun]]['Date'] = self.data[self.names_data[index_sun]]['Date'].str[8:10] + "/" + self.data[self.names_data[index_sun]]['Date'].str[5:7] + "/" + self.data[self.names_data[index_sun]]['Date'].str[0:4]
        self.data[self.names_data[index_sun]]['Date'] = pd.to_datetime(self.data[self.names_data[index_sun]]['Date'] , format = '%d/%m/%Y')
        self.data[self.names_data[index_sun]]['Rise'] = self.data[self.names_data[index_sun]]['Rise'].str.replace(':','.').astype(float)
        self.data[self.names_data[index_sun]]['Down'] = self.data[self.names_data[index_sun]]['Down'].str.replace(':','.').astype(float)
        self.data[self.names_data[index_sun]]['Last'] = self.data[self.names_data[index_sun]]['Last'].str.replace(':','.').astype(float)
    
        self.data[self.names_data[index_ref]] = pd.merge(self.data[self.names_data[index_ref]], self.data[self.names_data[index_sun]] , on = 'Date')
        self.data[self.names_data[index_ref]]  = self.data[self.names_data[index_ref]].drop_duplicates(["Date" , "Heure"])
        
        #########################################
        # ######## temperature check
        self.data[self.names_data[index_temp]].rename(columns={0:'Date', 1:'Temperature'},inplace=True)
        self.data[self.names_data[index_temp]]['Date'] = pd.to_datetime(self.data[self.names_data[index_temp]]['Date'], format = "%Y-%m-%d")
        self.data[self.names_data[index_ref]]['Date'] = pd.to_datetime(self.data[self.names_data[index_ref]]['Date'], format = "%d/%m/%Y")
        self.data[self.names_data[index_ref]] = pd.merge(self.data[self.names_data[index_ref]], self.data[self.names_data[index_temp]] , on = 'Date')
              
        
        for i in range(1,5):
            self.data[self.names_data[index_ref]] = pd.concat([self.data[self.names_data[index_ref]], pd.DataFrame(self.data[self.names_data[index_ref]]['Temperature'].shift(24*i)).rename(columns={'Temperature':'temperature_Lag'+ str(i)})],axis=1)
        
        #### Date management part 2 avec jours feriers
        
            ### jf classiques 
        self.data[self.names_data[index_ref]]["jf_dates"] = 0
        self.data[self.names_data[index_ref]].loc[(self.data[self.names_data[index_ref]]["jour_mois"].isin([1,27])) & (self.data[self.names_data[index_ref]]["mois"] == 1),"jf_dates"] =1
        self.data[self.names_data[index_ref]].loc[(self.data[self.names_data[index_ref]]["jour_mois"] == 1) & (self.data[self.names_data[index_ref]]["mois"] == 5),"jf_dates"] =1
        self.data[self.names_data[index_ref]].loc[(self.data[self.names_data[index_ref]]["jour_mois"] == 15) & (self.data[self.names_data[index_ref]]["mois"] == 8),"jf_dates"] =1
        self.data[self.names_data[index_ref]].loc[(self.data[self.names_data[index_ref]]["jour_mois"].isin([1,19])) & (self.data[self.names_data[index_ref]]["mois"] == 11),"jf_dates"] =1
        self.data[self.names_data[index_ref]].loc[(self.data[self.names_data[index_ref]]["jour_mois"].isin([25,8])) & (self.data[self.names_data[index_ref]]["mois"] == 12),"jf_dates"] =1
        
            ### jf paques, pentecote, ascension
        for i in range(self.start_year, self.end_year):
            j,m,a = self.datepaques(i) 
            self.data[self.names_data[index_ref]].loc[(self.data[self.names_data[index_ref]]["jour_mois"] == j ) & (self.data[self.names_data[index_ref]]["mois"] == m) & (self.data[self.names_data[index_ref]]["annee"] == a),"jf_dates"] =1
            date_paque = pd.datetime(a, m,j)
            
            #ascension
            self.data[self.names_data[index_ref]].loc[(self.data[self.names_data[index_ref]]["jour_mois"] == (date_paque + datetime.timedelta(days=39)).day ) & (self.data[self.names_data[index_ref]]["mois"] == (date_paque + datetime.timedelta(days=39)).month) & (self.data[self.names_data[index_ref]]["annee"] == a),"jf_dates"] =1
            #pentecote
            self.data[self.names_data[index_ref]].loc[(self.data[self.names_data[index_ref]]["jour_mois"] == (date_paque + datetime.timedelta(days=50)).day ) & (self.data[self.names_data[index_ref]]["mois"] == (date_paque + datetime.timedelta(days=50)).month) & (self.data[self.names_data[index_ref]]["annee"] == a),"jf_dates"] =1
        
        ListDf=[]
        for i in ["mois", "jour_semaine"]:
            ListDf.append(pd.get_dummies(self.data[self.names_data[index_ref]][i].astype(str),prefix=i))
        self.data[self.names_data[index_ref]] = pd.concat([self.data[self.names_data[index_ref]], pd.concat(ListDf, axis=1)], axis=1)
                      
        #### Creation de la variable j-1 | j | j+1 d'un jour hors travail
        self.data[self.names_data[index_ref]]["Proba_out_of_work"] = 0
              
        feriers = self.data[self.names_data[index_ref]]["Proba_out_of_work"].copy()
        feriers[self.data[self.names_data[index_ref]]["jour_semaine"].isin([5,6])] = 1
        local_lag_ferier = feriers.shift(-24)
        local_plus_lag_ferier = feriers.shift(24)
        
        del self.data[self.names_data[index_ref]]["Proba_out_of_work"]
        
        gc.collect()
        ##### 50% de chance de ne pas travailler pour un pont
        z = local_lag_ferier  + local_plus_lag_ferier + feriers 
        
        self.data[self.names_data[index_ref]]["Pont"] = 0
        self.data[self.names_data[index_ref]].loc[z==1,"Pont"]=1
        
        self.conso = self.data[self.names_data[index_ref]]
                
        #### is there nan
        for i in self.conso.columns:
            if np.any(pd.isnull(self.conso[i])) ==True:
                self.conso = self.conso[~pd.isnull(self.conso[i])]
        
        #### outliers suppression
        self.conso = self.conso[~((self.conso.Consommation  <25000)&(self.conso.Date <"2016-01-01"))] 
            
    def Temperature_modeling(self):
        self.conso["Temperature"] =  self.conso["Temperature"] - 4.750991 + self.conso["Heure"]*0.434547 -0.007320*self.conso["Heure"]**2 -0.000415*self.conso["Heure"]**3 + 0.354375*self.conso["mois"] + 0.500070*self.conso["Down"]-0.511187*self.conso["Rise"] -0.310402*self.conso["Last"] -0.010265*self.conso["jour_annee"]
        self.conso["Temperature"] =  savgol_filter(self.conso["Temperature"], 7, 3)

dt = DataPreparation(r"C:\Users\alexandre\Documents\Antonin project\Monaco Predictions\Modele long terme\Data\\")
#dt.conso.Temperature[0:200].plot()