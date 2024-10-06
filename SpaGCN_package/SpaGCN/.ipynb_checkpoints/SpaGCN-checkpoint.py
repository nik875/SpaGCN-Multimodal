import numpy as np
from scipy.sparse import issparse
from anndata import AnnData
import torch
from sklearn.decomposition import PCA
from . models import simple_GC_DEC


class multiSpaGCN:
    def __init__(self, num_pcs=50):
        self.model = None
        self.embed = None
        self.adj_exp = None
        self.adata_all = None
        self.num_pcs = num_pcs
        self.params = {
            'lr': 0.005,
            'max_epochs': 2000,
            'weight_decay': 0,
            'opt': "admin",
            'init_spa': True,
            'init': "louvain",
            'n_neighbors': 10,
            'n_clusters': None,
            'res': 0.4,
            'tol': 1e-3
        }

    def train(self, adata_list, adj_list, l_list, **kwargs):
        # Update parameters with any provided kwargs
        if 'num_pcs' in kwargs:
            self.num_pcs = kwargs.pop('num_pcs')
        self.params.update(kwargs)

        # Prepare data
        num_spots = sum(adata.shape[0] for adata in adata_list)
        adj_exp_all = np.zeros((num_spots, num_spots))
        start = 0
        for adata, adj, l in zip(adata_list, adj_list, l_list):
            size = adata.shape[0]
            adj_exp = np.exp(-1 * (adj**2) / (2 * (l**2)))
            adj_exp_all[start:start + size, start:start + size] = adj_exp
            start += size

        # Concatenate AnnData objects
        batch_cat = [str(i) for i in range(len(l_list))]
        self.adata_all = AnnData.concatenate(
            *adata_list,
            join='inner',
            batch_key="dataset_batch",
            batch_categories=batch_cat)

        # Perform PCA
        pca = PCA(n_components=self.num_pcs)
        X = self.adata_all.X.toarray() if issparse(self.adata_all.X) else self.adata_all.X
        self.embed = pca.fit_transform(X)

        # Train model
        self.model = simple_GC_DEC(self.embed.shape[1], self.embed.shape[1])
        self.model.fit(self.embed, adj_exp_all, **self.params)
        self.adj_exp = adj_exp_all

    def predict(self):
        _, q = self.model.predict(self.embed, self.adj_exp)
        y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
        # Max probability plot
        prob = q.detach().numpy()
        return y_pred, prob


class SpaGCN(multiSpaGCN):
    """
    Specific case of multiSpaGCN
    """

    def __init__(self, l=None):
        super().__init__()
        self.l = l

    def set_l(self, l):
        self.l = l

    def train(self, adata, adj, *args, **kwargs):  # pylint: disable=arguments-differ
        super().train([adata], [adj], [self.l], *args, **kwargs)

    @staticmethod
    def test():
        print('hi from spagcn')
