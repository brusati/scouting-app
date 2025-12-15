import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

# =============================================================================
# 1. CONFIGURACIÓN DE LA PÁGINA
# =============================================================================
st.set_page_config(
    page_title="Racing Club Scouting",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# 2. CSS DE ALTO CONTRASTE (SOLUCIÓN DEFINITIVA)
# =============================================================================
st.markdown("""
    <style>
    /* 1. FORZAR FONDO CLARO EN LA APP PRINCIPAL */
    .stApp {
        background-color: #ffffff;
    }

    /* 2. BARRA LATERAL (SIDEBAR): FONDO OSCURO Y TEXTO BLANCO */
    section[data-testid="stSidebar"] {
        background-color: #0e2433 !important;
    }
    
    /* FUERZA BRUTA: Todo texto dentro de la sidebar debe ser BLANCO */
    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    
    /* Corrección específica para los Sliders en la sidebar */
    section[data-testid="stSidebar"] .stSlider p {
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] .stSlider div[data-testid="stTickBar"] > div {
        background-color: #ffffff !important;
    }
    
    /* 3. ZONA PRINCIPAL (MAIN): TEXTO OSCURO */
    /* Títulos principales */
    h1, h2, h3, h4, h5, h6 {
        color: #0e2433 !important;
    }
    /* Párrafos y textos generales */
    p, li, div {
        color: #212529;
    }
    
    /* 4. PESTAÑAS (TABS) */
    button[data-baseweb="tab"] div p {
        color: #0e2433 !important; /* Texto de la pestaña negro */
        font-weight: 600;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: #e3f2fd !important;
        border-bottom: 3px solid #6CABDD !important;
    }

    /* 5. TARJETAS DE MÉTRICAS */
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #dee2e6;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    /* Asegurar que el texto dentro de las tarjetas sea oscuro */
    .metric-card h3, .metric-card p, .metric-card span {
        color: #0e2433 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# =============================================================================
# 3. DEFINICIÓN DE COLUMNAS
# =============================================================================

COLS_TO_NORMALIZE = [
    'goals', 'yellowCards', 'redCards', 'groundDuelsWon', 'aerialDuelsWon',
    'successfulDribbles', 'tackles', 'assists', 'totalDuelsWon', 'wasFouled',
    'fouls', 'dispossessed', 'accurateFinalThirdPasses', 'bigChancesCreated',
    'accuratePasses', 'keyPasses', 'accurateCrosses', 'accurateLongBalls',
    'interceptions', 'clearances', 'dribbledPast', 'bigChancesMissed',
    'totalShots', 'shotsOnTarget', 'blockedShots', 'hitWoodwork', 'offsides',
    'expectedGoals', 'errorLeadToGoal', 'errorLeadToShot', 'passToAssist'
]

COLS_TO_DROP = [
    'player id', 'team id', 'appearances',
    'saves', 'savedShotsFromInsideTheBox', 'savedShotsFromOutsideTheBox',
    'goalsConcededInsideTheBox', 'goalsConcededOutsideTheBox',
    'highClaims', 'successfulRunsOut', 'punches', 'runsOut'
]

# =============================================================================
# 4. FUNCIONES DE CARGA Y PROCESAMIENTO
# =============================================================================

@st.cache_data
def load_and_clean_data(csv_path):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        st.error(f"No se encontró el archivo {csv_path}. Asegurate de subirlo.")
        return None

    # 1. Filtro de Arqueros
    df_field = df[~((df['offsides'] == 0) & (df['saves'] > 2))].copy()

    # 2. Normalización per 90
    for col in COLS_TO_NORMALIZE:
        if col in df_field.columns:
            df_field[f'{col}_per90'] = np.where(
                df_field['minutesPlayed'] > 0,
                (df_field[col] / df_field['minutesPlayed']) * 90,
                0
            )
            
    return df_field

@st.cache_data
def train_models(df, min_minutes, n_neighbors, n_clusters):
    # 1. Filtro de Minutos
    df_filtered = df[df['minutesPlayed'] >= min_minutes].copy()
    
    if df_filtered.empty:
        return None, None, None, None, None

    # 2. Selección de Features
    all_numeric = df_filtered.select_dtypes(include="number").columns
    features = [c for c in all_numeric if ('_per90' in c) or ('Percentage' in c)]
    features = [c for c in features if c not in COLS_TO_DROP and 'minutes' not in c]
    
    X = df_filtered[features].fillna(0)
    
    # 3. Escalado
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 4. K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df_filtered['Cluster'] = clusters.astype(str)
    
    # 5. PCA (Visualización 2D)
    pca = PCA(n_components=2)
    X_pca_coords = pca.fit_transform(X_scaled)
    df_filtered['PC1'] = X_pca_coords[:, 0]
    df_filtered['PC2'] = X_pca_coords[:, 1]
    
    # 6. KNN
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="mahalanobis")
    nn.fit(X_scaled)
    
    return df_filtered, X_scaled, nn, pca, features

# =============================================================================
# 5. INTERFAZ DE USUARIO
# =============================================================================

st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/5/56/Escudo_de_Racing_Club_%282014%29.svg", width=80)
st.sidebar.title("Filtros de Scouting")

DATA_FILE = "liga_argentina_player_stats_2025.csv"
df_raw = load_and_clean_data(DATA_FILE)

if df_raw is not None:
    # Controles
    min_minutes = st.sidebar.slider("Minutos Mínimos Jugados", 0, 1500, 600, step=50)
    n_neighbors = st.sidebar.slider("Cantidad de Similares", 3, 10, 5)
    
    # Entrenar modelos
    df_final, X_scaled, nn_model, pca_model, feature_names = train_models(df_raw, min_minutes, n_neighbors, 5)
    
    if df_final is None:
        st.error("No hay jugadores que cumplan con el criterio de minutos.")
        st.stop()

    st.title("🎓 Racing Club - Scouting Tool")

    # =============================================================================
    # 6. PESTAÑAS PRINCIPALES
    # =============================================================================
    
    tab1, tab2, tab3 = st.tabs(["🔍 Buscador de Jugadores", "🗺️ Mapa de Mercado (PCA)", "📊 Análisis de Clusters"])

    # --- TAB 1: BUSCADOR ---
    with tab1:
        st.subheader("Herramienta de Reemplazo y Comparación")
        
        col_search, col_space = st.columns([2, 1])
        with col_search:
            player_list = df_final['player'].sort_values().unique()
            target_player = st.selectbox("Seleccioná un jugador objetivo:", player_list)
        
        if target_player:
            # Lógica KNN
            idx = df_final[df_final["player"] == target_player].index[0]
            item_pos = df_final.index.get_loc(idx)
            vec = X_scaled[item_pos].reshape(1, -1)
            dists, indices = nn_model.kneighbors(vec)
            
            similares = df_final.iloc[indices[0]].copy()
            similares['Similitud (%)'] = (1 - dists[0]) * 100
            rec = similares.iloc[1] 
            
            st.markdown(f"""
            <div class='metric-card'>
                <h3 style='margin:0;'>Recomendación Principal: <span style='color: #6CABDD'>{rec['player']}</span></h3>
                <p style='margin:0; font-size: 1.1em;'>
                    <b>Equipo:</b> {rec['team']} | 
                    <b>Similitud:</b> {rec['Similitud (%)']:.1f}% | 
                    <b>Cluster:</b> {rec['Cluster']}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # --- RADAR CHART MEJORADO (CONTRASTE NEGRO) ---
            st.subheader("Comparativa Visual (Radar)")
            radar_cols = [
                'goals_per90', 'assists_per90', 'keyPasses_per90', 
                'successfulDribbles_per90', 'tackles_per90', 'interceptions_per90',
                'totalDuelsWonPercentage', 'accuratePassesPercentage'
            ]
            radar_labels = ['Goles', 'Asist.', 'Pases Clave', 'Regates', 'Entradas', 'Intercep.', '% Duel', '% Pases']
            
            def get_percentile(val, col):
                return (df_final[col] < val).mean()

            vals_p1 = [get_percentile(df_final.loc[idx, c], c) for c in radar_cols]
            vals_p2 = [get_percentile(rec[c], c) for c in radar_cols]
            
            fig_radar = go.Figure()
            # Jugador 1 (Azul Oscuro Racing)
            fig_radar.add_trace(go.Scatterpolar(
                r=vals_p1, theta=radar_labels, fill='toself', name=target_player, 
                line=dict(color='#1d3557')
            ))
            # Jugador 2 (Celeste Racing)
            fig_radar.add_trace(go.Scatterpolar(
                r=vals_p2, theta=radar_labels, fill='toself', name=rec['player'], 
                line=dict(color='#6CABDD')
            ))

            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True, 
                        range=[0, 1],
                        tickfont=dict(color="black", size=10, family="Arial") # TEXTO EJE NEGRO
                    ),
                    angularaxis=dict(
                        tickfont=dict(color="black", size=14, family="Arial Black") # ETIQUETAS GRANDES Y NEGRAS
                    ),
                    bgcolor="white" # FONDO DEL POLAR BLANCO
                ),
                showlegend=True,
                legend=dict(
                    font=dict(color="black", size=12),
                    bgcolor="rgba(255,255,255,0.8)", # Fondo leyenda blanco semitransparente
                    bordercolor="Black",
                    borderwidth=1
                ),
                paper_bgcolor='white', # FONDO PAPEL BLANCO
                plot_bgcolor='white',
                margin=dict(l=60, r=60, t=40, b=40)
            )
            st.plotly_chart(fig_radar, use_container_width=True)

            st.subheader("Lista de Similares")
            # Forzamos que la tabla también tenga texto legible
            st.dataframe(similares[['player', 'team', 'Similitud (%)', 'minutesPlayed', 'Cluster', 'goals_per90', 'assists_per90']].style.background_gradient(subset=['Similitud (%)'], cmap="Blues").format({"Similitud (%)": "{:.1f}", "goals_per90": "{:.2f}", "assists_per90": "{:.2f}"}))

    # --- TAB 2: MAPA PCA ---
    with tab2:
        st.subheader("Mapa de Talento (PCA)")
        st.info("Cada punto es un jugador. Los colores representan 'Roles' (Cluster) o 'Equipos'.")
        
        opciones_color = {
            "Cluster (Rol Táctico)": "Cluster",
            "Equipo": "team"
        }
        
        seleccion_usuario = st.radio("Colorear por:", list(opciones_color.keys()), horizontal=True)
        columna_real = opciones_color[seleccion_usuario] 
        
        fig_pca = px.scatter(
            df_final, x="PC1", y="PC2",
            color=columna_real,  
            hover_name="player",
            hover_data=["team", "minutesPlayed", "goals_per90"],
            color_discrete_sequence=px.colors.qualitative.G10,
            height=600
        )
        fig_pca.update_traces(marker=dict(size=8, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
        
        # Configuración de contraste para PCA
        fig_pca.update_layout(
            paper_bgcolor='white',
            plot_bgcolor='white',
            xaxis=dict(
                title=dict(text="Componente Principal 1", font=dict(color="black")), 
                tickfont=dict(color="black"),
                showgrid=True, gridcolor='#eeeeee'
            ),
            yaxis=dict(
                title=dict(text="Componente Principal 2", font=dict(color="black")), 
                tickfont=dict(color="black"),
                showgrid=True, gridcolor='#eeeeee'
            ),
            legend=dict(font=dict(color="black"), title=dict(font=dict(color="black")))
        )
        
        st.plotly_chart(fig_pca, use_container_width=True)

    # --- TAB 3: CLUSTERS ---
    with tab3:
        st.subheader("Interpretación de Roles")
        
        features_plot = [
            'goals_per90', 'assists_per90', 'tackles_per90', 
            'interceptions_per90', 'successfulDribbles_per90', 
            'accuratePassesPercentage', 'aerialDuelsWon_per90'
        ]
        
        cluster_means = df_final.groupby('Cluster')[features_plot].mean().reset_index()
        scaler_minmax = MinMaxScaler()
        heatmap_data = scaler_minmax.fit_transform(cluster_means[features_plot])
        
        fig_heat = px.imshow(
            heatmap_data,
            x=features_plot,
            y=cluster_means['Cluster'],
            labels=dict(x="Métrica", y="Cluster", color="Intensidad"),
            color_continuous_scale="RdBu_r"
        )
        
        fig_heat.update_layout(
            paper_bgcolor='white',
            plot_bgcolor='white',
            xaxis=dict(tickfont=dict(color="black")),
            yaxis=dict(tickfont=dict(color="black")),
            font=dict(color="black") # Texto general negro
        )
        
        st.plotly_chart(fig_heat, use_container_width=True)
        
        st.markdown("##### Jugador Representativo por Cluster")
        cols = st.columns(5)
        for i, col in enumerate(cols):
            cluster_id = str(i)
            if cluster_id in df_final['Cluster'].values:
                top_p = df_final[df_final['Cluster'] == cluster_id].sort_values('minutesPlayed', ascending=False).iloc[0]
                col.metric(f"Cluster {cluster_id}", top_p['player'], top_p['team'])

else:
    st.warning("Esperando carga de datos...")