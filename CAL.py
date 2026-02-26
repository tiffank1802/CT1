import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from assemble import assemblage,nodes,elements

#Paramètres généraux de la simulation

E=210_000
NU=.3
C=E/((1+NU)*(1-2*NU))*np.array([
    [(1-NU), NU, 0],
    [NU,(1-NU), 0],
    [0, 0,(1-2*NU)/2]]
    )
K=assemblage()
F=np.zeros_like(K[0,:])
print(F.shape)

    
Kreack=K.copy()

Freac=F.copy()
P=np.max(np.max(K))*10
for i in range(nodes.shape[0]):
    if nodes[i,1]==12.5 and nodes[i,2]==13.75:
        ind0=(i-1)*2+0
        ind1=(i-1)*2+1
        K[ind0,ind0]+=P
        K[ind1,ind1]+=P
        F[ind0]+=1000
    if nodes[i,1]==12.5 and nodes[i,2]==-13.75:
        ind0=(i-1)*2+0
        ind1=(i-1)*2+1
        K[ind0,ind0]+=P
        K[ind1,ind1]+=P
        F[ind0]-=1000
        

      
U=np.linalg.solve(K,F)
Depl = np.zeros_like(nodes)
Depl[:, 0] = U[0::2]    
np.savetxt("U.txt",U)
# en x
Depl[:, 1] = U[1::2]    

# en y

# amplification = 10
# nodes_def = nodes + amplification * Depl  

# # nœuds déplacés amplifiés

# plt.figure()

# # Choix des couleurs : ici on colorie selon la norme du déplacement (valeur scalaire par nœud)
# disp_norm = np.linalg.norm(Depl, axis=1)

# plt.tripcolor(nodes_def[:, 0], nodes_def[:, 1], elements - 1, disp_norm, edgecolors='k', cmap='jet')

# plt.colorbar(label='Norme du déplacement')
# plt.title('Maillage déformé coloré selon déplacement')
# plt.xlabel('X (mm)')
# plt.ylabel('Y (mm)')
# plt.axis('equal')
# plt.grid(True)
# plt.show()
# # plt.savefig("CT/results/maillage_def.png")






# # Supposons que tu as déjà :
# # nodes : (n_noeuds, 2), coordonnées XY, indices 0-based
# # elements : (n_elem, 3), indices de nœuds, indices 0-based

# # Si tes éléments sont 1-based, convertis :
# # elements = elements - 1

# # Liste des faces pour Plotly
# I = elements[:, 0]
# J = elements[:, 1]
# K = elements[:, 2]

# fig = go.Figure()

# # Maillage triangulaire
# fig.add_trace(go.Mesh3d(
#     x=nodes[:, 0], y=nodes[:, 1], z=np.zeros(nodes.shape[0]),
#     i=I, j=J, k=K,
#     color='lightblue',
#     opacity=0.4,
#     flatshading=True,
#     name="Maillage",
#     showscale=False,
#     hoverinfo='skip' 

# # Pour éviter duplication de survol
# ))

# # Points des nœuds + leur numéro au survol
# fig.add_trace(go.Scatter3d(
#     x=nodes[:,0]+Depl[:,0], y=nodes[:,1]+Depl[:,1], z=np.zeros(nodes.shape[0]),
#     mode='markers+text',
#     # text=[str(i+1) for i in range(nodes.shape[0])], 

# # Numéro affiché
#     textposition='top center',
#     marker=dict(size=4, color='red'),
#     # hovertemplate=
#     #     'Noeud %{text} <br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>',
#     showlegend=False,
# ))

# fig.update_layout(
#     scene=dict(
#         xaxis_title='X',
#         yaxis_title='Y',
#         zaxis_title='',
#         aspectmode='data'
#     ),
#     title="Maillage FEM interactif (collaborez, zoomez, survolez!)",
#     margin=dict(l=0, r=0, t=40, b=0),
#     height=600,
# )

# fig.show()
