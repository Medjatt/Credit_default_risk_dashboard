import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import pickle
import shap
from PIL import Image

# Désactiver l'avertissement pour l'utilisation de st.pyplot() sans arguments
st.set_option('deprecation.showPyplotGlobalUse', False)

# Charger le fichier application_train_subset.csv
data = pd.read_csv('application_subset.csv')

# Charger les données SHAP à partir du fichier shap_data.pkl
with open('shap_data_subset.pkl', 'rb') as f:
    shap_data_subset = pickle.load(f)
    shap_values = shap_data_subset['shap_values']
    feature_names = shap_data_subset['feature_names']

# Fonction pour calculer l'âge en années à partir de DAYS_BIRTH
def calculate_age(days_birth):
    return abs(days_birth) / 365

# Fonction pour calculer l'ancienneté d'emploi en années à partir de DAYS_EMPLOYED
def calculate_employment_years(days_employed):
    return abs(days_employed) / 365

# Charger l'image
image = Image.open('logo.png')

# Ajouter l'image à la barre latérale
st.sidebar.image(image, use_column_width=False)

# User interface with Streamlit
st.sidebar.title('Navigation')
page = st.sidebar.radio("Go to", ['Prediction', 'Features importances', 'Features visualization', 'Update Features'])

st.title('Credit Scoring Dashboard')

# Input area for SK_ID_CURR of the client
st.header('Input')
sk_id_curr = st.number_input("Enter client SK_ID_CURR:", min_value=0)  # Allowing only positive values

if page == 'Prediction':
    # Prediction button
    if st.button("Predict"):
        # Endpoint of the deployed Azure API
        endpoint = "https://modelscore.azurewebsites.net/predict/"

        # Make a GET request to the Azure API with the SK_ID_CURR
        response = requests.get(endpoint + str(sk_id_curr))

        if response.status_code == 200:
            # Extract prediction from the response
            prediction_data = response.json()
                
            # Display the prediction for the client with SK_ID_CURR
            st.write(f"Prediction for client SK_ID_CURR = {sk_id_curr}:")
            st.write(f"Probability of failure: {prediction_data['probability_of_failure']:.4f}")
            st.write(f"Prediction: {prediction_data['prediction']}")

            # Plot a gauge
            prob_failure = prediction_data['probability_of_failure']

            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                delta={'reference': 52.52,'increasing.color': "red", 'decreasing.color': "green"},
                value=prob_failure*100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Probability of failure"},
                gauge={'axis': {'range': [None, 100]},
                        'bar': {'color': "blue"},
                        'steps': [
                            {'range': [0, 52.52], 'color': "green"},
                            {'range': [52.52, 100], 'color': "red"}]}))

            st.plotly_chart(fig)

            # Additional information about the client
            st.header('Client Informations')
            client_info = data[data['SK_ID_CURR'] == sk_id_curr]
            if not client_info.empty:
                # Show principales informations about a client
                client_info = data[data['SK_ID_CURR'] == sk_id_curr]
                st.write(client_info)
                st.write(f"Age: {calculate_age(client_info['DAYS_BIRTH'].values[0]):.2f} years")
                st.write(f"Income: {client_info['AMT_INCOME_TOTAL'].values[0]}")
                st.write(f"Employment : {calculate_employment_years(client_info['DAYS_EMPLOYED'].values[0]):.2f} years")
            else:
                st.write("Client information not found.")

        else:
            st.write(f"Client ID = {sk_id_curr} not found in the database.")

elif page == 'Features importances':
    # Display SHAP contributions
    st.header('SHAP Contributions')
    if sk_id_curr in data['SK_ID_CURR'].values:
        client_index = data[data['SK_ID_CURR'] == sk_id_curr].index[0]
        shap_values_client = shap_values[client_index]

        # Waterfall Plot
        st.subheader('Summary Plot')
        # Create a list for short features names
        short_feature_names = [name[:20] for name in feature_names]
        # Plot the SHAP values
        plt.figure(figsize=(10,5))
        shap.summary_plot(shap_values, plot_type="bar", feature_names=short_feature_names, max_display=10)
        st.pyplot(plt)

        # Si le client est trouvé dans les données
        if client_index is not None:

            #waterfall plot
            st.subheader('Waterfall Plot')
            plt.figure(figsize=(10,5))
            shap.waterfall_plot(shap.Explanation(values=shap_values_client, base_values=0, feature_names=short_feature_names))
            st.pyplot(plt)
            # Afficher l'importance des features pour le client spécifique
            st.subheader('Force plot')
            # Calculer la valeur moyenne attendue
            expected_value = -0.4900967311988457
            # Utiliser le plot de force de SHAP avec l'option matplotlib=True
            plt.figure(figsize=(10,5))
            shap.force_plot(expected_value, shap_values_client, feature_names=short_feature_names, matplotlib=True)
            st.pyplot(plt)

elif page == 'Features visualization':
    # Liste des caractéristiques numériques et catégorielles
    numeric_features = ['EXT_SOURCE_3', 'EXT_SOURCE_2', 'AMT_GOODS_PRICE', 
                        'CREDIT_TERM', 'AMT_ANNUITY', 
                        'FLAG_DOCUMENT_3', 'ANNUITY_INCOME_PERCENT', 
                        'EXT_SOURCE_1', 'REGION_RATING_CLIENT_W_CITY']

    categorical_features = ['NAME_EDUCATION_TYPE', 'ORGANIZATION_TYPE', 
                            'EMERGENCYSTATE_MODE', 'HOUSETYPE_MODE', 
                            'FLAG_OWN_CAR', 'NAME_INCOME_TYPE']

    # Combinaison des deux listes en une seule pour la sélection
    features_list = numeric_features + categorical_features

    # Ajout d'une section pour la visualisation des caractéristiques du client
    st.header('Visualization of client characteristics')

    # Sélection de la caractéristique à visualiser
    feature = st.selectbox('Select a characteristic', features_list)

    # Détermination du type de caractéristique
    feature_type = 'Num' if feature in numeric_features else 'Cat'

    # Création d'un histogramme pour la caractéristique numérique sélectionnée
    if feature_type == 'Num':
        fig, ax = plt.subplots()
        ax.hist(data[feature], bins=30, edgecolor='black')
        ax.set_title(f'Distribution de {feature}')
        #sns.histplot(data[feature], bins=30, edgecolor='black', kde=True)
        #ax.set_title(f'Distribution de {feature}', fontsize=16)

        # Ajout d'une ligne verticale pour la valeur de la caractéristique du client sélectionné
        if sk_id_curr in data['SK_ID_CURR'].values:
            client_value = data[data['SK_ID_CURR'] == sk_id_curr][feature].values[0]
            if feature_type == 'Num':
                ax.axvline(client_value, color='red', linestyle='dashed', linewidth=2)
                ax.legend([f'Client : {round(client_value, 2)}'])

    else:  # Création d'un diagramme à barres pour la caractéristique catégorielle sélectionnée
        fig, ax = plt.subplots()
        data[feature].value_counts().plot(kind='bar', ax=ax)
        ax.set_title(f'Subplot of {feature}')
        #sns.countplot(data=data, x=feature, palette='pastel')
        #ax.set_title(f'Compte de {feature}', fontsize=16)

        # Ajout d'une barre rouge pour la valeur de la caractéristique catégorielle du client sélectionné
        if sk_id_curr in data['SK_ID_CURR'].values:
            client_value = data[data['SK_ID_CURR'] == sk_id_curr][feature].values[0]
            bars = ax.patches
            for bar in bars:
                if bar.get_height() == data[feature].value_counts()[client_value]:
                    bar.set_color('red')
                    break
            ax.legend([f'Client : {client_value}'], labelcolor='red')

    st.pyplot(fig)


    # Ajout d'une section pour l'analyse bivariée
    st.header('Bivariate analysis')

    # Sélection des deux caractéristiques à analyser
    feature1 = st.selectbox('Select the first characteristic', features_list)

    # Mise à jour de la liste des caractéristiques pour exclure la première caractéristique sélectionnée
    features_list_updated = [feature for feature in features_list if feature != feature1]

    # Sélection de la deuxième caractéristique à analyser
    feature2 = st.selectbox('Select the second characteristic', features_list_updated)

    # Détermination du type des caractéristiques
    feature1_type = 'Num' if feature1 in numeric_features else 'Cat'
    feature2_type = 'Num' if feature2 in numeric_features else 'Cat'

    # Création d'un diagramme de dispersion pour deux caractéristiques numériques
    if feature1_type == 'Num' and feature2_type == 'Num':
        fig, ax = plt.subplots()
        ax.scatter(data[feature1], data[feature2])
        ax.set_xlabel(feature1)
        ax.set_ylabel(feature2)
        ax.set_title(f'{feature1} et {feature2}')
        # Ajout d'un point rouge pour le client sélectionné
        if sk_id_curr in data['SK_ID_CURR'].values:
            client_value1 = data[data['SK_ID_CURR'] == sk_id_curr][feature1].values[0]
            client_value2 = data[data['SK_ID_CURR'] == sk_id_curr][feature2].values[0]
            ax.scatter(client_value1, client_value2, color='red')
            ax.legend([f'Client : ({round(client_value1, 2)}, {round(client_value2, 2)})'], labelcolor='red')
            
        st.pyplot(fig)
    # Création d'un tableau de contingence pour deux caractéristiques catégorielles
    elif feature1_type == 'Cat' and feature2_type == 'Cat':
        contingency_table = pd.crosstab(data[feature1], data[feature2])
        st.write(contingency_table)
    # Création d'un boxplot pour une caractéristique numérique et une caractéristique catégorielle
    else:
        if feature1_type == 'Cat':
            cat_feature, num_feature = feature1, feature2
        else:
            cat_feature, num_feature = feature2, feature1
        fig, ax = plt.subplots()
        data.boxplot(column=num_feature, by=cat_feature, ax=ax)
        ax.set_title(f'{num_feature} et {cat_feature}')
        ax.set_xlabel(cat_feature)
        ax.set_ylabel(num_feature)
        # Ajout d'une ligne horizontale pour le client sélectionné dans la catégorie correspondante
        if sk_id_curr in data['SK_ID_CURR'].values:
            client_cat_value = data[data['SK_ID_CURR'] == sk_id_curr][cat_feature].values[0]
            client_num_value = data[data['SK_ID_CURR'] == sk_id_curr][num_feature].values[0]
            for i, label in enumerate(ax.get_xticklabels()):
                if label.get_text() == client_cat_value:
                    ax.hlines(client_num_value, i+1-0.4, i+1+0.4, color='red', linestyle='solid', linewidth=2)
                    ax.legend([f'Client : {round(client_num_value, 2)}'], labelcolor='red')
                    break
        st.pyplot(fig)

elif page == 'Update Features':
    st.header('Update Features')
    
    # Input area for updating feature values
    ext_source_3 = st.number_input("Enter new EXT_SOURCE_3 value:", min_value=0.0)  # Allowing only positive values
    ext_source_2 = st.number_input("Enter new EXT_SOURCE_2 value:", min_value=0.0)  # Allowing only positive values
    amt_goods_price = st.number_input("Enter new AMT_GOODS_PRICE value:", min_value=0.0)  # Allowing only positive values

    if st.button("Update"):
        if sk_id_curr in data['SK_ID_CURR'].values:

            # Update the feature values in the dataset
            data.loc[data['SK_ID_CURR'] == sk_id_curr, 'EXT_SOURCE_3'] = ext_source_3
            data.loc[data['SK_ID_CURR'] == sk_id_curr, 'EXT_SOURCE_2'] = ext_source_2
            data.loc[data['SK_ID_CURR'] == sk_id_curr, 'AMT_GOODS_PRICE'] = amt_goods_price

            # Save the updated data to the CSV file
            data.to_csv('application_subset.csv', index=False)

            st.write("Features updated successfully!")
        
        else:
            st.write(f"Client ID = {sk_id_curr} not found in the database.")
