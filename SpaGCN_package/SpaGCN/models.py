import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.nn.parameter import Parameter
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import scanpy as sc
from tqdm import tqdm
from . layers import GraphConvolution


class simple_GC_DEC(nn.Module):
    def __init__(self, nfeat, nhid, alpha=0.2):
        super().__init__()
        self.gc = GraphConvolution(nfeat, nhid)
        self.nhid = nhid
        # self.mu determined by the init method
        self.alpha = alpha
        self.trajectory = None
        self.n_clusters = None
        self.mu = None

    def forward(self, x, adj):
        x = self.gc(x, adj)
        q = 1.0 / ((1.0 + torch.sum((x.unsqueeze(1) - self.mu)**2, dim=2) / self.alpha) + 1e-8)
        q = q**(self.alpha + 1.0) / 2.0 + 1e-8
        q = q / torch.sum(q, dim=1, keepdim=True)
        return x, q

    def loss_function(self, p, q):
        def kld(target, pred):
            return torch.mean(torch.sum(target * torch.log(target / (pred + 1e-6)), dim=1))
        loss = kld(p, q)
        return loss

    def target_distribution(self, q):
        p = q**2 / torch.sum(q, dim=0)
        p = p / torch.sum(p, dim=1, keepdim=True)
        return p

    def init_clusters(self, init="louvain", adata=None, n=10, res=0.4):
        if isinstance(init, np.ndarray):
            print("Using given initial cluster assignments (init argument)")
            y_pred = init
            self.n_clusters = len(np.unique(y_pred))
        elif init == "kmeans":
            print("Initializing cluster centers with kmeans, n_clusters known")
            self.n_clusters = n
            kmeans = KMeans(self.n_clusters, n_init=20)
            y_pred = kmeans.fit_predict(adata)
        elif init == "louvain":
            print("Initializing cluster centers with louvain, resolution = ", res)
            sc.pp.neighbors(adata, n_neighbors=n)
            sc.tl.louvain(adata, resolution=res)
            y_pred = adata.obs['louvain'].astype(int).to_numpy()
            self.n_clusters = len(np.unique(y_pred))
        else:
            raise ValueError("Invalid value for init")
        return y_pred

    def fit(self, X, adj, lr=0.001, max_epochs=5000, update_interval=3, trajectory_interval=50,
            weight_decay=5e-4, opt="adam", init="louvain", res=0.4, n_clusters=10, n_neighbors=10,
            init_spa=True, tol=1e-3):
        if opt == "sgd":
            optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        elif opt == "admin":
            optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        features = self.gc(torch.FloatTensor(X), torch.FloatTensor(adj))
        # ----------------------------------------------------------------
        y_pred = self.init_clusters(init=init, adata=(features.detach.numpy() if init_spa else X),
                                    n=(n_clusters if init == "kmeans" else n_neighbors), res=res)
        # ----------------------------------------------------------------
        y_pred_last = y_pred
        self.mu = Parameter(torch.Tensor(self.n_clusters, self.nhid))
        X = torch.FloatTensor(X)
        adj = torch.FloatTensor(adj)
        self.trajectory = [y_pred]
        features = pd.DataFrame(features.detach().numpy(), index=np.arange(0, features.shape[0]))
        Group = pd.Series(y_pred, index=np.arange(0, features.shape[0]), name="Group")
        Mergefeature = pd.concat([features, Group], axis=1)
        cluster_centers = np.asarray(Mergefeature.groupby("Group").mean())

        self.mu.data.copy_(torch.Tensor(cluster_centers))
        self.train()
        pbar = tqdm(range(max_epochs), desc="Training")
        for epoch in pbar:
            if epoch % update_interval == 0:
                _, q = self.forward(X, adj)
                p = self.target_distribution(q).data
            optimizer.zero_grad()
            _, q = self(X, adj)

            loss = self.loss_function(p, q)

            loss.backward()

            optimizer.step()

            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

            if epoch % trajectory_interval == 0:
                self.trajectory.append(torch.argmax(q, dim=1).data.cpu().numpy())

            # Check stop criterion
            y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / X.shape[0]
            y_pred_last = y_pred
            if epoch > 0 and (epoch - 1) % update_interval == 0 and delta_label < tol:
                print('delta_label ', delta_label, '< tol ', tol)
                print("Reach tolerance threshold. Stopping training.")
                print("Total epoch:", epoch)
                break
        pbar.close()

    def predict(self, X, adj):
        z, q = self(torch.FloatTensor(X), torch.FloatTensor(adj))
        return z, q


class GC_DEC(simple_GC_DEC):
    def __init__(self, nfeat, nhid1, nhid2, dropout=0.5, alpha=0.2):
        super().__init__(nfeat, nhid1, alpha)
        self.nhid = nhid2  # So that in fit() we use nhid2 to create mu
        self.gc2 = GraphConvolution(nhid1, nhid2)
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.gc(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        q = 1.0 / ((1.0 + torch.sum((x.unsqueeze(1) - self.mu)**2, dim=2) / self.alpha) + 1e-6)
        q = q**(self.alpha + 1.0) / 2.0
        q = q / torch.sum(q, dim=1, keepdim=True)
        return x, q
