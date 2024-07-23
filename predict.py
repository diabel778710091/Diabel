import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Charger le modèle pré-entraîné
pipeline = joblib.load('logistic_model.pkl')  # Assurez-vous d'avoir sauvegardé votre pipeline avec joblib

# Définir le thème de la page
st.set_page_config(
    page_title="Prédiction du Churn",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ajouter une image d'entête (ajoutez votre propre image URL ou chemin local)
st.markdown(
    """
    <style>
    .header {
        background-color: #f0f2f6;
        padding: 10px;
        text-align: center;
        border-bottom: 2px solid #dfe1e5;
    }
    .header img {
        max-width: 100%;
        height: auto;
    }
    </style>
    <div class="header">
        <img src="https://www.linkedin.com/in/el-hadji-diabel-dieng-4abbba191/overlay/background-image/" alt="Header Image">
    </div>
    """,
    unsafe_allow_html=True
)

st.title('Prédiction du Churn')
st.subheader('Entrez les valeurs ci-dessous pour prédire le churn.')

# Créer une boîte de saisie élégante
st.markdown(
    """
    <style>
    .input-field {
        margin-bottom: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Définir les champs de saisie pour les fonctionnalités
region = st.text_input('Région', '', key='region', help='Entrez la région du client')
montant = st.number_input('Montant', min_value=0.0, value=0.0, key='montant', help='Montant dépensé par le client')
frequence_rech = st.number_input('Fréquence de Recherche', min_value=0.0, value=0.0, key='frequence_rech', help='Fréquence de recherche du client')
revenue = st.number_input('Revenu', min_value=0.0, value=0.0, key='revenue', help='Revenu généré par le client')
arpu_segment = st.number_input('ARPU Segment', min_value=0.0, value=0.0, key='arpu_segment', help='ARPU par segment')
frequence = st.number_input('Fréquence', min_value=0.0, value=0.0, key='frequence', help='Fréquence des achats du client')
data_volume = st.number_input('Volume de données', min_value=0.0, value=0.0, key='data_volume', help='Volume de données utilisé par le client')
on_net = st.number_input('On Net', min_value=0.0, value=0.0, key='on_net', help='Utilisation des services On Net')
orange = st.number_input('Orange', min_value=0.0, value=0.0, key='orange', help='Utilisation des services Orange')
tigo = st.number_input('Tigo', min_value=0.0, value=0.0, key='tigo', help='Utilisation des services Tigo')
zone1 = st.number_input('Zone 1', min_value=0.0, value=0.0, key='zone1', help='Utilisation de la Zone 1')
zone2 = st.number_input('Zone 2', min_value=0.0, value=0.0, key='zone2', help='Utilisation de la Zone 2')
regularity = st.number_input('Régularité', min_value=0.0, value=0.0, key='regularity', help='Régularité des achats')
top_pack = st.number_input('Top Pack', min_value=0.0, value=0.0, key='top_pack', help='Utilisation du Top Pack')
freq_top_pack = st.number_input('Fréquence Top Pack', min_value=0.0, value=0.0, key='freq_top_pack', help='Fréquence d’utilisation du Top Pack')

# Bouton de validation
if st.button('Faire une prédiction'):
    # Créer un DataFrame à partir des entrées de l'utilisateur
    user_input = pd.DataFrame({
        'REGION': [region],
        'MONTANT': [montant],
        'FREQUENCE_RECH': [frequence_rech],
        'REVENUE': [revenue],
        'ARPU_SEGMENT': [arpu_segment],
        'FREQUENCE': [frequence],
        'DATA_VOLUME': [np.log(data_volume + 1) if data_volume > 0 else 0],
        'ON_NET': [on_net],
        'ORANGE': [orange],
        'TIGO': [tigo],
        'ZONE1': [zone1],
        'ZONE2': [zone2],
        'REGULARITY': [regularity],
        'TOP_PACK': [top_pack],
        'FREQ_TOP_PACK': [freq_top_pack]
    })

    # Faire une prédiction
    prediction = pipeline.predict(user_input)

    # Afficher le résultat
    st.write('### Résultat de la Prédiction:')
    st.write('Prédiction :', 'Churn' if prediction[0] == 1 else 'No Churn')
