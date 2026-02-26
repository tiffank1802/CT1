import numpy as np
import streamlit as st
import plotly.graph_objects as go
import CAL 

st.set_page_config(page_title="Maillage FEM interactif", layout="wide")

st.title("Visualisation interactive du maillage FEM")

# --- Paramètres / options ---
st.sidebar.header("Options")

# Chemins des fichiers
node_file = st.sidebar.text_input("Fichier NODE", "NODE.txt")
elem_file = st.sidebar.text_input("Fichier ELEMENT", "ELEMENT.txt")

amplification = st.sidebar.slider("Facteur d'amplification des déplacements", 0.0, 50.0, 10.0, 1.0)
afficher_deplacement = st.sidebar.checkbox("Afficher maillage déformé (si U disponible)", value=False)
afficher_numeros_noeuds = st.sidebar.checkbox("Afficher numéros de nœuds", value=True)

st.sidebar.write("Clique sur le bouton ci-dessous pour recharger les données.")
reload = st.sidebar.button("Recharger les fichiers")

# --- Chargement des données ---
@st.cache_data
def charger_maillage(node_file, elem_file):
    nodeData = np.loadtxt(node_file)
    elementData = np.loadtxt(elem_file)

    

# À ADAPTER selon ton format :
    

# Si nodes sont en colonnes 7 et 8 (comme dans ton Octave) :
    nodes = nodeData[:, 6:8]          

# (n_noeuds, 2)
    elements = elementData[:, 5:8]    

# (n_elem, 3)
    elements = elements.astype(int)

    return nodes, elements

nodes, elements = charger_maillage(node_file, elem_file)

st.write(f"Nombre de nœuds : {nodes.shape[0]}")
st.write(f"Nombre d'éléments : {elements.shape[0]}")

# --- Optionnel : chargement des déplacements U (si tu les as dans un fichier) ---
U = CAL.U
if afficher_deplacement:
    

# Exemple : si tu as sauvegardé U dans un fichier texte "U.txt"
    

# avec 2*ne lignes (ux1, uy1, ux2, uy2, ...)
    try:
        U = np.loadtxt("U.txt")  

# à adapter
        if U.ndim > 1:
            U = U.flatten()
    except OSError:
        st.warning("Fichier U.txt introuvable ou illisible. Le maillage déformé ne sera pas affiché.")
        U = None

# --- Calcul des nœuds déformés si U est disponible ---
if U is not None and afficher_deplacement:
    ne = nodes.shape[0]
    if U.shape[0] != 2*ne:
        st.error(f"Taille de U incohérente : attendu {2*ne}, trouvé {U.shape[0]}")
        nodes_def = nodes.copy()
        disp_norm = np.zeros(ne)
    else:
        Depl = np.zeros_like(nodes)
        Depl[:, 0] = U[0::2]   

# ux
        Depl[:, 1] = U[1::2]   

# uy
        nodes_def = nodes + amplification * Depl
        disp_norm = np.linalg.norm(Depl, axis=1)
else:
    nodes_def = nodes.copy()
    disp_norm = np.zeros(nodes.shape[0])

# --- Construction de la figure Plotly ---
fig = go.Figure()

# Maillage (non déformé) en 3D (z=0)
fig.add_trace(go.Mesh3d(
    x=nodes[:, 0],
    y=nodes[:, 1],
    z=np.zeros(nodes.shape[0]),
    i=elements[:, 0] - 1,   

# -1 si indices 1-based dans le fichier
    j=elements[:, 1] - 1,
    k=elements[:, 2] - 1,
    color='lightgray',
    opacity=0.3,
    name="Maillage initial",
    showscale=False
))

# Maillage déformé (si U dispo)
if U is not None and afficher_deplacement:
    fig.add_trace(go.Mesh3d(
        x=nodes_def[:, 0],
        y=nodes_def[:, 1],
        z=np.zeros(nodes_def.shape[0]),
        i=elements[:, 0] - 1,
        j=elements[:, 1] - 1,
        k=elements[:, 2] - 1,
        intensity=disp_norm,          

# couleur selon norme du déplacement
        colorscale='Jet',
        opacity=0.8,
        name="Maillage déformé",
        showscale=True,
        colorbar_title="||U||"
    ))

# Points des nœuds + numéros
if afficher_numeros_noeuds:
    fig.add_trace(go.Scatter3d(
        x=nodes_def[:, 0],
        y=nodes_def[:, 1],
        z=np.zeros(nodes_def.shape[0]),
        mode='markers+text',
        # text=[str(i+1) for i in range(nodes_def.shape[0])],
        textposition='top center',
        marker=dict(size=3, color='red'),
        # hovertemplate='Noeud %{text}<br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>',
        name="Nœuds"
    ))
else:
    fig.add_trace(go.Scatter3d(
        x=nodes_def[:, 0],
        y=nodes_def[:, 1],
        z=np.zeros(nodes_def.shape[0]),
        mode='markers',
        marker=dict(size=3, color='red'),
        hovertemplate='X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>',
        name="Nœuds"
    ))

fig.update_layout(
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='',
        aspectmode='data'
    ),
    margin=dict(l=0, r=0, t=40, b=0),
    height=700,
    title="Maillage FEM (Plotly + Streamlit)"
)

st.plotly_chart(fig, width='stretch')

st.info("Modifie tes fichiers NODE.txt / ELEMENT.txt (et U.txt si utilisé), puis clique sur 'Recharger les fichiers' dans la barre latérale pour mettre à jour le graphe.")
