# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 14:32:13 2021

@author: khares
"""

import pandas as pd


df_doc=pd.read_csv('Doctors.csv')

df_doc.head()

userinput = 'Chest X-Ray'

df_docfilter=df_doc[(df_doc['Status']=='Available') & (df_doc['Speciality']==userinput)]

df_docrec=df_docfilter[['Doctor Name','Contact Number']]

df_docrec.head()


df_hosp=pd.read_csv('Hospitals.csv')

df_hosp.head()

df_hospfilter=df_hosp[(df_hosp['Availability Percentage'] >= 60) & (df_hosp['institution_type']=='Hospital')]

df_hospfilter.head()

df_hosprec=df_hospfilter[['ID','Available Beds']]

df_hosprec.head()
