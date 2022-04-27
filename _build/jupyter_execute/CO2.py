#!/usr/bin/env python
# coding: utf-8

# # **CO2 emissions and climate change** 
# ### <br>Delving into the past, present and future responsibilities<br>  

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
import pycountry_convert as pc
import plotly.express as px
import plotly.offline as pyo
import plotly.express as px
import plotly.io as pio
import numpy as np
#pyo.init_notebook_mode()

pio.renderers.default = 'notebook'


# In[2]:


def country_to_continent(country_code):
    try:
        country_alpha2 = pc.country_alpha3_to_country_alpha2(country_code)
        country_continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
        country_continent_name = pc.convert_continent_code_to_continent_name(country_continent_code)
    except: 
        country_continent_name = country_code
        
    return country_continent_name


# In[3]:


df = pd.read_csv('data/owid-co2-data.csv',sep = ',',encoding = 'unicode_escape')
df = df.fillna(0)
temp = df[['iso_code', 'country', 'year', 'co2','energy_per_gdp', 'co2_per_capita','primary_energy_consumption','gdp','co2_per_gdp']] 


# In[4]:


df_sector = pd.read_csv('data/co-emissions-by-sector.csv',sep = ',',encoding = 'unicode_escape')
df_sector = df_sector.fillna(0)
df_sector['Continent'] = df_sector['Code'].apply(lambda x: country_to_continent(x))


# In[5]:


df_sector_18 = df_sector[(df_sector['Year'] == 2018) & ~(df_sector['Continent'].isin([0,'TLS','OWID_WRL']))]
df_sun = df_sector_18[~df_sector_18[['Buildings', 'Industry',
       'Land-use change and forestry', 'Other fuel combustion', 'Transport',
       'Manufacturing and construction', 'Fugitive emissions',
       'Electricity and heat']].eq(0).all(axis = 1)].drop(columns = ["Year","Code","Land-use change and forestry"])


# In[6]:


df_sun2 = df_sun.melt(id_vars=["Entity","Continent"], 
        var_name="Sector", 
        value_name="Value")
#temp = df_sun2[df_sun2['Entity'].isin(['Mongolia'])]
#temp
fig = px.sunburst(df_sun2, path=['Continent', 'Entity','Sector'], values='Value', color='Continent',width=1000, height=800)
fig.update_traces(textinfo="label+percent entry ")
fig.show()


# In[7]:


mask = (temp['year'] > 1990) & (temp['country'].isin(['Asia','Europe','Oceania','Africa',
                                                      'North America','South America','Antarctica']))

mask2 = (temp['year'] > 1990) & ~(temp['country'].isin(['Asia','Europe','Oceania','Africa',
                                                      'North America','South America','Antarctica','World']))

mask3 = (temp['year'] > 1990) & (temp['country'].isin(['China','United States','Canada','Qatar',
                                                      'India','Germany','United Kingdom','Saudi Arabia']))


# In[8]:


cont_co2 = temp[mask2]
cont_co22 = temp[mask]


# In[9]:


fig = px.choropleth(cont_co2, locations="iso_code", color="co2", hover_name="country", animation_frame="year",
                    range_color=[np.percentile(cont_co2['co2'],5),max(cont_co2['co2'])],
                    projection = 'natural earth',
                    title="Total CO2 emission per country from 1990 to 2020",labels={'co2':'million tonnes'})
fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 150
fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 10
fig.show()


# In[10]:


fig = px.choropleth(cont_co2, locations="iso_code", color="co2_per_capita", hover_name="country", animation_frame="year",
                    range_color=[np.percentile(cont_co2['co2_per_capita'],1),max(cont_co2['co2_per_capita'])],
                    projection='natural earth',
                    title="CO2 emission per capita per country from 1990 to 2020",labels={'co2_per_capita':'tonnes'})
fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 150
fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 10
fig.show()


# In[11]:


temp = cont_co2[cont_co2['country'].isin(['China','India','United States','Germany','Saudi Arabia'])]
fig = px.scatter(temp, x="primary_energy_consumption", y="co2", color='country', trendline="ols")
fig.show()


# In[12]:


df2 = px.data.gapminder()
fig = px.scatter(temp, x="primary_energy_consumption", y="co2", animation_frame="year", animation_group="country",
           size="co2_per_capita", color="country", hover_name="country", facet_col="country",labels={'country':'Country'},
           log_x=False, log_y= False, size_max=45, range_x=[1,max(temp['energy_per_gdp'])], range_y=[1,max(temp['co2_per_capita'])])

fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 150
fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 10
fig.show()


# In[13]:


df.describe()


# In[ ]:




