from modularities_notebooks.modularity import clustered_connections, print_weight_matrix, compute_density_matrix, plot_connection_matrices
from sklearn.preprocessing import StandardScaler
import torch
import matplotlib.pyplot as plt

def get_Ws(N,nb_classes,epsilon=1, modularity =1,base_w = 1,w = 1,plot=True):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f'device: {device}')
    else:
        device = torch.device("cpu")
        print(f'device: {device}')

    n_assemblies = nb_classes
    ss = StandardScaler()
    mask, cluster_ids, _ = clustered_connections(n_neurons=N, n_clusters=n_assemblies, density=1./n_assemblies, modularity=modularity)
    mask = (mask* (base_w*w)) + base_w

    W, cluster_ids, t = clustered_connections(n_neurons=N, n_clusters=n_assemblies, density=epsilon, modularity=0.)
    if w > 0.:
        W *= base_w
        W *= mask

    if epsilon ==1:
        W = W-1
    W2 = (W != 0).astype(int)
    if plot:
        plt.figure(figsize=(6,6))
        plt.title(f'Weight Mask. N={N}; modularity={modularity}')
        plt.imshow(W,cmap='viridis')
        plt.colorbar()
        plt.show()
        plt.figure(figsize=(6,6))
        plt.title('Weight Mask: training')
        plt.imshow(W2)
        plt.colorbar()
        plt.show()
    return torch.tensor(W,device=device,dtype=torch.float32), torch.tensor(W2,device=device,dtype=torch.float32)