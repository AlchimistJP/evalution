import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA

# Utilisation de st.cache_data pour la mise en cache des données
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)
st.header("WULEME Komivi Jean-Paul")
st.header("Analyse des données commerciales  de vente en ligne")
# Importation du document et vérification du document
da = load_data("C:/Users/Huesped/Desktop/ExamenNOSQL/pythonProject1/atomic_data.csv")
#st.write(da.head())

# Ajouter une colonne 'Unit Cost' (si elle n'existe pas) - remplir cette colonne avec les coûts réels des unités
# Exemple: da['Unit Cost'] = [valeurs réelles des coûts unitaires pour chaque produit]

st.header("Le coût total")

# Calculer le coût total par transaction
da['Cost'] = da['Quantity'] * da['Unit Price']
column_sum = da['Cost'].sum()
st.write(f"Le Cout Total des produits vendu multiplier par le prix unitaire {column_sum}")
# Classement des produits par chiffre d'affaires
st.header("Classement des Produits par Chiffre d'Affaires")
product_revenue = da.groupby('Product Name')['Cost'].sum().sort_values(ascending=False)
st.write(product_revenue)
# Visualisation des produits par chiffre d'affaires
st.bar_chart(product_revenue)

# Afficher les 10 produits les plus vendus
st.header("Les 10 produits les plus vendus")
top_10_products = da.groupby('Product Name')['Quantity'].sum().sort_values(ascending=False).head(10)
st.write(top_10_products)
# Classement des produits par chiffre d'affaires
da['Product Category'] = pd.qcut(da['Cost'], q=[0, 0.7, 0.9, 1], labels=['C', 'B', 'A'])

# Moyen de paiement le plus utilisé
moyens_de_paiement = da['Payment Method'].value_counts()
st.write("Moyens de paiement les plus utilisés :\n", moyens_de_paiement)
st.write("Moyens de paiement le plus utilisés est le  :\n","Cash" )

# Pays avec les ventes les plus élevées
ventes_par_pays = da.groupby('Country')['Cost'].sum().sort_values(ascending=False)
st.write("Ventes par pays :\n", ventes_par_pays)
st.write("les ventes sont plus élevées au  :\n", "Portugal")
# Tendance des ventes en fonction du temps
da['Transaction Date'] = pd.to_datetime(da['Transaction Date'])
ventes_par_mois = da.resample('ME', on='Transaction Date')['Cost'].sum()

plt.plot(ventes_par_mois.index, ventes_par_mois.values)
plt.xlabel('Mois')
plt.ylabel('Chiffre d\'affaires')
plt.title('Tendance des ventes au fil du temps')
plt.xticks(rotation=45)  # Faire pivoter les étiquettes des mois
st.pyplot(plt)

# Prédire le chiffre d'affaires du mois de Mai 2024
model = ARIMA(ventes_par_mois, order=(5,1,0))
model_fit = model.fit()
forecast = model_fit.forecast(steps=1)
st.write("Prédiction du chiffre d'affaires pour Mai 2024:", forecast.iloc[0])
