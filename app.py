import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
# Configuration de la page Streamlit
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Spaceship Titanic - ML",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
# Styles CSS personnalisés
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');
 
    html, body, [class*="css"] {
        font-family: 'Syne', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'Space Mono', monospace !important;
    }
    .main {
        background-color: #0d0f1a;
        color: #e0e6f0;
    }
    .stApp {
        background: linear-gradient(135deg, #0d0f1a 0%, #131629 60%, #0d1a2a 100%);
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1f3a, #0f1628);
        border: 1px solid #2a3560;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(59,130,246,0.1);
    }
    .section-header {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Space Mono', monospace;
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1628 0%, #131a2e 100%);
        border-right: 1px solid #2a3560;
    }
    .stRadio > label {
        color: #94a3b8 !important;
    }
    .result-box {
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid;
    }
    .transported-true {
        background: rgba(16, 185, 129, 0.1);
        border-color: #10b981;
        color: #34d399;
    }
    .transported-false {
        background: rgba(239, 68, 68, 0.1);
        border-color: #ef4444;
        color: #f87171;
    }
</style>
""", unsafe_allow_html=True)
 
# -----------------------------------------------------------------------------
# Fonctions de chargement des données et du modèle (à compléter par les étudiants)
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    
    train_df = pd.read_csv("train.csv")
    test_df  = pd.read_csv("test.csv")
    return train_df, test_df


# -----------------------------------------------------------------------------
# Menu de Navigation (Sidebar)
# -----------------------------------------------------------------------------
st.sidebar.markdown("## 🚀 Spaceship Titanic")
st.sidebar.markdown("**Orsini Pierre-Jean**")
st.sidebar.markdown("---")
 
menu = [
    "🏠 Accueil",
    "📊 Analyse Exploratoire (EDA)",
    "🤖 Test du Modèle",
    "📈 Performances du Modèle",
    "💡 Conclusions & Perspectives"
]
choice = st.sidebar.radio("Navigation :", menu)
 
st.sidebar.markdown("---")
 
if choice == "📊 Analyse Exploratoire (EDA)" and data_loaded:
    st.sidebar.subheader("Filtres")
    planet_filter = st.sidebar.multiselect(
        "Planète d'origine :",
        options=df_eda['HomePlanet'].dropna().unique(),
        default=df_eda['HomePlanet'].dropna().unique()
    )
    transported_filter = st.sidebar.multiselect(
        "Transporté :",
        options=[True, False],
        default=[True, False]
    )
    df_eda = df_eda[
        df_eda['HomePlanet'].isin(planet_filter) &
        df_eda['Transported'].isin(transported_filter)
    ]
 
st.sidebar.markdown("---")
st.sidebar.info("Projet ML – Kaggle Spaceship Titanic\nRandom Forest Classifier")
 
# Palette cohérente
PALETTE  = {"True": "#3b82f6", "False": "#f59e0b"}
BLUE     = "#3b82f6"
ORANGE   = "#f59e0b"
BG_PLOT  = "#131629"
TEXT_COL = "#e0e6f0"
 
def style_fig(fig, ax_or_axes=None):
    """Applique un style sombre cohérent."""
    fig.patch.set_facecolor(BG_PLOT)
    axes = ax_or_axes if ax_or_axes is not None else fig.get_axes()
    if not isinstance(axes, list):
        axes = [axes]
    for ax in axes:
        ax.set_facecolor(BG_PLOT)
        ax.tick_params(colors=TEXT_COL)
        ax.xaxis.label.set_color(TEXT_COL)
        ax.yaxis.label.set_color(TEXT_COL)
        ax.title.set_color(TEXT_COL)
        for spine in ax.spines.values():
            spine.set_edgecolor('#2a3560')
    return fig

# =============================================================================
# Section 1 : Accueil
# =============================================================================
if choice == "🏠 Accueil":
    st.markdown("<h1 class='section-header'>🚀 Spaceship Titanic</h1>", unsafe_allow_html=True)
    st.markdown("### Prédire qui sera téléporté dans une dimension parallèle")
 
    st.markdown("""
    L'année 2912 : le vaisseau spatial **Titanic** entre en collision avec une anomalie spatio-temporelle.
    Des passagers ont été **téléportés dans une dimension alternative**.  
    L'objectif est de prédire, à partir des données de bord, **quels passagers ont été transportés**.
    """)
    
    # Utilisation de st.expander pour cacher/afficher l'information détaillée
    with st.expander("📌 Vos Missions (Étudiants) : Cliquez pour développer", expanded=True):
        st.markdown("""
        1. **Données** : Placer le vrai dataset dans le dossier `data/` et l'importer dans la fonction `load_data()`.
        2. **EDA** : Remplacer les visualisations génériques par vos propres graphiques interactifs (Plotly, Seaborn, Altair).
        3. **Modélisation** : Entraîner un modèle dans un notebook (dossier `notebooks/`), le sauvegarder (ex: avec `joblib` ou `pickle`) dans le dossier `models/`, et le charger dans `load_model()`.
        4. **Prédiction** : Créer un formulaire interactif robuste pour tester le modèle sur de nouvelles données.
        5. **Conclusion** : Rédiger un bilan clair et argumenté des performances et pistes d'amélioration.
        """)

    st.subheader("Structure du projet recommandée :")
    st.code("""
    mon_projet/
    ├── app.py                  # Script principal Streamlit (ce fichier)
    ├── requirements.txt        # Fichier des dépendances
    ├── README.md               # Documentation de votre projet
    ├── data/
    │   ├── raw/                # Données brutes de Kaggle
    │   └── processed/          # Données nettoyées (optionnel)
    ├── models/                 # Modèles entraînés (.pkl, .joblib, etc.)
    ├── notebooks/              # Vos notebooks d'exploration et d'entraînement (Jupyter)
    └── src/
        └── utils.py          # Fonctions ou classes python externes au script principal
    """, language="markdown")

# =============================================================================
# Section 2 : Analyse Exploratoire (EDA)
# =============================================================================
elif choice == "📊 Analyse Exploratoire (EDA)":
    st.title("Analyse Exploratoire des Données (EDA) 📈")
    
    if data_loaded:
        # Utilisation de st.tabs pour organiser finement l'affichage
        tab1, tab2, tab3 = st.tabs(["Aperçu des données", "Statistiques Descriptives", "Visualisations"])
        
        with tab1:
            st.subheader("Extrait du jeu de données")
            st.dataframe(df.head(15), use_container_width=True)
            
            # Bouton de téléchargement
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Télécharger le dataset filtré au format CSV",
                data=csv,
                file_name='dataset_filtre.csv',
                mime='text/csv',
            )
            
        with tab2:
            st.subheader("Description des variables")
            st.write(df.describe())
            
            # Affichage de métriques clés "Dashboard style"
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Nombre de lignes", df.shape[0])
            col_m2.metric("Nombre de colonnes", df.shape[1])
            col_m3.metric("Valeurs manquantes totales", df.isna().sum().sum())

        with tab3:
            st.subheader("Visualisations Graphiques")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Distribution de Feature_1 (selon la Target)**")
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.histplot(data=df, x="Feature_1", hue="Target", kde=True, ax=ax, palette="Set1")
                st.pyplot(fig)
                
            with col2:
                st.markdown("**Matrice de Corrélation**")
                fig, ax = plt.subplots(figsize=(6, 4))
                corr = df.select_dtypes(include=[np.number]).corr()
                sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax, fmt=".2f", vmin=-1, vmax=1)
                st.pyplot(fig)
                
            st.info("💡 **Astuce Étudiant :** Explorer le package `plotly.express` pour des graphiques 100% interactifs (zooms, survol, etc.).")

# =============================================================================
# Section 3 : Test du Modèle
# =============================================================================
elif choice == "🤖 Test du Modèle":
    st.title("Tester le Modèle de Machine Learning 🚀")
    st.markdown("""
    Testez la prédiction du modèle en modifiant les paramètres via le formulaire ci-dessous.
    """)
    
    with st.form("prediction_form"):
        st.subheader("Définissez les paramètres d'entrée :")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            feat_1 = st.number_input("Feature 1 (Valeur continue)", value=0.0, step=0.1)
        with col2:
            feat_2 = st.slider("Feature 2 (Curseur)", min_value=0.0, max_value=100.0, value=25.0, step=1.0)
        with col3:
            feat_3 = st.selectbox("Feature 3 (Catégorie numérique)", options=[1, 2, 3, 4, 5])
            
        submit_button = st.form_submit_button(label="Exécuter le modèle")

    if submit_button:
        # Configuration des logs ou spin d'attente
        with st.spinner('Analyse par le modèle en cours...'):
            input_data = pd.DataFrame({
                "Feature_1": [feat_1], 
                "Feature_2": [feat_2],
                "Feature_3": [feat_3]
            })
            
            # Séparation de l'UI pour la réponse
            st.markdown("### Résultats")
            
            try:
                prediction = model.predict(input_data)
                proba = model.predict_proba(input_data)
                
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    if prediction[0] == 1:
                        st.success(f"La classe prédite est : **{prediction[0]}**")
                    else:
                        st.warning(f"La classe prédite est : **{prediction[0]}**")
                        
                with res_col2:
                    st.markdown("**Confiance du modèle (Probabilités)**")
                    proba_df = pd.DataFrame(proba, columns=["Classe 0", "Classe 1"]).T
                    st.bar_chart(proba_df)
                
            except Exception as e:
                st.error(f"Erreur lors de la prédiction. Vérifiez que les colonnes données au modèle correspondent exactement à celles attendues. Détail : {e}")

# =============================================================================
# Section 4 : Conclusions & Perspectives
# =============================================================================
elif choice == "💡 Conclusions & Perspectives":
    st.title("Conclusions et Pistes d'Amélioration 🎯")
    
    st.markdown("""
    ### 📝 Bilan du Projet
    Utilisez cette page pour résumer l'impact métier de votre modèle par rapport au problème Kaggle initial. 
    Parlez des compromis, de la précision vs le rappel de votre modèle (ex: importance des Faux Positifs).
    
    **Performances Finales du Modèle :**
    - **Accuracy :** 85% (exemple)
    - **F1-Score :** 82% (exemple)
    
    ### 🚧 Pistes d'Améliorations futures
    """)
    
    # Checklist des améliorations pour implémentation par les étudiants
    st.checkbox("✨ Ingénierie des caractéristiques avancées (Feature Engineering)")
    st.checkbox("🚀 Entraînement sur une infrastructure Cloud (AWS, GCP)")
    st.checkbox("🌍 Intégrer l'API FastAPI en backend plutôt que d'avoir le modèle directement dans Streamlit")
    st.checkbox("📈 Ajout d'explicabilité du modèle (SHAP values intégrées visuellement)")

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#4a5580;font-family:Space Mono,monospace;font-size:12px'>"
    "Spaceship Titanic – Orsini Pierre-Jean | Aflokkat | Random Forest Classifier</p>",
    unsafe_allow_html=True