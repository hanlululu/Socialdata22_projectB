#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import plotly.express as px


# # Who is reposible for the increase in CO2 emissions?

# The rise in CO2 emissions is one of the greatest challenges humans face today and one of the primary reasons we experience global warming. Currently, most of the worlds countries recognize the rise in CO2 emissions as a major challenge, but it is hard to point to who is responsible for the current situation and perhaps more importantly, who should take responsibility. 

# ### Which countries emit the most CO2 today?

#  CO2 emissions have been on an exponential rise since the industrial revolution. Many countries have a share in the current level of CO2 emissions. Although the top three most CO2 emitting countries are China, the United States and India. The countries that have a large CO2 emission are either heavily industrialised countries such as China or western countries with a very energy consuming lifestyle such as the United States.

# In[2]:


df1 = pd.read_csv('annual-co-emissions-by-region.csv')
remove_list = ['Asia (excl. China & India)','Asia','EU-27','EU-28','Europe (excl. EU-27)','Europe (excl. EU-28)','Oceania','North America','North America (excl. USA)', 'World', 'High-income countries', 'European Union', 'Upper-middle-income countries', 'European Union (27)', 'European Union (28)', 'Africa', 'South America', 'Lower-middle-income countries','International transport','Europe', 'Low-income countries']
df1 = df1[~df1['Entity'].isin(remove_list)]
df1 = df1.sort_values('Annual CO2 emissions (zero filled)', ascending = False)
df1 = df1.rename(columns={'Annual CO2 emissions (zero filled)': "Annual CO2 emissions"})
fig = px.area(df1, x="Year", y="Annual CO2 emissions", color="Entity")
fig.update_layout(showlegend=False)
fig.show()


# The distribution of global CO2 emissions can also be investigated in the interactive worldmap below. Try dragging the handle in the timeline to see how the distribution of CO2 emission has changed over from 1990 to 2020.

# In[23]:


df = pd.read_csv('owid-co2-data.csv',sep = ',',encoding = 'unicode_escape')
df = df.fillna(0)
temp = df[['iso_code', 'country', 'year', 'co2','energy_per_gdp', 'co2_per_capita','primary_energy_consumption','gdp','co2_per_gdp']] 
mask = (temp['year'] > 1990) & (temp['country'].isin(['Asia','Europe','Oceania','Africa',
                                                      'North America','South America','Antarctica']))

mask2 = (temp['year'] > 1990) & ~(temp['country'].isin(['Asia','Europe','Oceania','Africa',
                                                      'North America','South America','Antarctica','World']))

cont_co2 = temp[mask2]
cont_co22 = temp[mask]
fig = px.choropleth(cont_co2, locations="iso_code", color="co2", hover_name="country", animation_frame="year",
                    range_color=[np.percentile(cont_co2['co2'],5),max(cont_co2['co2'])],
                    projection = 'natural earth',
                    title="Total CO2 emission per country from 1990 to 2020",labels={'co2':'million tonnes'})
fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 150
fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 10
fig.show()


# ### Who emitted the most CO2 in total?

# When trying to place the responsibility for the current level of CO2 emissions western countries often point to the eastern industrialised countries lack of action to reduce CO2 emissions. On the other hand, eastern countries argue that it is easier for western countries to reduce CO2 emissions as they are the product of a certain lifestyle rather than a matter of survival, since many western countries already have had their industrial revolution. When looking at the cumulative distribution of CO2 emission as percentage it is revealed historically the the UK has accounted for almost all CO2 emissions. 

# In[18]:


df2 = pd.read_csv('cumulative-co2-emissions-region.csv')
remove_list = ['Asia (excl. China & India)','Asia','EU-27','EU-28','Europe (excl. EU-27)','Europe (excl. EU-28)','Oceania','North America','North America (excl. USA)', 'World', 'High-income countries', 'European Union', 'Upper-middle-income countries', 'European Union (27)', 'European Union (28)', 'Africa', 'South America', 'Lower-middle-income countries','International transport','Europe', 'Low-income countries']
df2 = df2[~df2['Entity'].isin(remove_list)]
df2 = df2.sort_values('Cumulative CO2 emissions (zero filled)', ascending = False)
df2 = df2.rename(columns={'Cumulative CO2 emissions (zero filled)': "Cumulative CO2 emissions"})
fig = px.area(df2, x="Year", y="Cumulative CO2 emissions", color="Entity", groupnorm = 'percent')
fig.update_layout(showlegend=False)
fig.show()


# However, when not looking at the cumulative distribution as a percentage it becomes clear that countries such as China account for a much larger part than the UK as the total annual CO2 emission was much lower in the 19th and 20th century when the UK was mainly responsible for the worlds CO2 emissions.

# In[19]:


fig = px.area(df2, x="Year", y="Cumulative CO2 emissions", color="Entity")
fig.update_layout(showlegend=False)
fig.show()


# ### Looking at people instead of countries

# It can furthermore be argued that looking at the CO2 emissions per country is an unfair way of placing the responsibility for the worlds CO2 emissions since the largest CO2 emitter China is also the country with the largest population.
# 
# When looking at the 2020 distribution of CO2 emissions per capita China is no longer the largest CO2 emitter but rather countries such as Mongolia, Saudi Arabia, Australia, the US and Canada. Countries such as Mongolia and Saudi Arabia mainly have a large CO2 emission per capita because of their massive coal and oil productions.

# In[21]:


fig = px.choropleth(cont_co2, locations="iso_code", color="co2_per_capita", hover_name="country", animation_frame="year",
                    range_color=[np.percentile(cont_co2['co2_per_capita'],1),max(cont_co2['co2_per_capita'])],
                    projection='natural earth',
                    title="CO2 emission per capita per country from 1990 to 2020",labels={'co2_per_capita':'tonnes'})
fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 150
fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 10
fig.show()


# In conclusion, it is difficult to point exactly to who is responsible as global warming indeed is a global phenomenon caused by many different countries and factors. So all developed and industrialized countries really need to take action in order to reduce the CO2 emissions in time with even more ambitious political initiatives.

# Sources:
# https://climatepositions.com/mongolia-and-other-coal-producing-countries-the-thirteen-most-coal-dependent-countries/
