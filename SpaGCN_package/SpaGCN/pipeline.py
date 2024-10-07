import copy
import matplotlib.pyplot as plt
import scanpy as sc
import anndata as ad
import numpy as np
import torch
from torch_geometric.data import Data
import scipy.sparse as sp
from . SpaGCN import SpaGCN
from . util import prefilter_genes, prefilter_specialgenes, search_l, refine
from . calculate_adj import calculate_adj_matrix, extract_regions
from . PyGCL_GBT import auto_train_and_embed


class Pipeline:
    """
    SpaGCN + outer product from MISO + PyGCL representation learning
    """

    def __init__(self, adata_list, image, spot_x, spot_y, **kwargs):
        self.adata_list = adata_list
        self.adata_reprs = None
        self.image = image
        self.spot_x = spot_x
        self.spot_y = spot_y
        self.spagcn = SpaGCN(**kwargs)

    def _encode_data(self, adata, adj, repr_size, **kwargs):
        # Convert adjacency matrix to COO format
        adj_coo = sp.coo_matrix(adj)

        # Create edge_index
        edge_index = torch.tensor(np.array([adj_coo.row, adj_coo.col]), dtype=torch.long)

        # Create edge_attr (weights)
        edge_attr = torch.tensor(adj_coo.data, dtype=torch.float)

        # Get node features from adata
        # Assuming adata.X contains the feature matrix
        x = torch.tensor(adata.X.toarray() if sp.issparse(adata.X) else adata.X, dtype=torch.float)

        # Create the PyTorch Geometric Data object
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        return auto_train_and_embed(graph, repr_size, **kwargs)

    def _outer_product(self, adata_reprs):
        with_prod = copy.copy(adata_reprs)
        for a, i in enumerate(adata_reprs):
            for b in range(a + 1, len(adata_reprs)):
                j = adata_reprs[b]
                assert i.X.shape[0] == j.X.shape[0]
                n_rows, n_cols = i.X.shape[0], i.X.shape[1] * j.X.shape[1]
                prod = np.empty((n_rows, n_cols))
                for k in range(n_rows):
                    prod[k] = np.outer(i.X[k], j.X[k]).flatten()
                # Create a new AnnData object with the result
                result_adata = ad.AnnData(prod)
                # Set var_names as combinations of original var_names
                result_adata.var_names = [f"{v1}_{v2}" for v1 in i.var_names for v2 in j.var_names]
                result_adata.obs_names = i.obs_names  # Keep original obs_names
                with_prod.append(result_adata)
        return with_prod

    def _elementwise_product(self, adata_reprs):
        with_prod = copy.copy(adata_reprs)
        for a, i in enumerate(adata_reprs):
            for b in range(a + 1, len(adata_reprs)):
                j = adata_reprs[b]
                assert i.X.shape == j.X.shape
                prod = i.X * j.X
                # Create a new AnnData object with the result
                result_adata = ad.AnnData(prod)
                # Set var_names as combinations of original var_names
                result_adata.var_names = [f"{v1}_{v2}" for v1, v2 in zip(i.var_names, j.var_names)]
                result_adata.obs_names = i.obs_names  # Keep original obs_names
                with_prod.append(result_adata)
        return with_prod

    def fit_transform(self, repr_size=32, beta=50, p=.5, shape='hexagon', savepath=None,
                      encode_args=None, product='outer', num_pcs=0, **kwargs):
        """
        p: percentage of total expression contributed by neighborhoods.
        num_pcs: number of principal components. set to 0 to disable pca
        """
        # Calculate adjacency matrix based only on spatial data
        adj = calculate_adj_matrix(x=self.spot_x, y=self.spot_y, histology=False)

        # Set the l
        self.spagcn.l = search_l(p, adj, start=0.01, end=1000, tol=0.01, max_run=100)

        # Encode all adata
        # TODO: KNOWN ISSUE WITH IMAGE DIMENSIONS
        img_regions = extract_regions(self.image, self.spot_x, self.spot_y, beta)
        adata_reprs = [self._encode_data(i, adj, repr_size, **encode_args) for i in
                       [img_regions, *self.adata_list]]
        if product == 'outer':
            # Calculate outer products
            adata_reprs = self._outer_product(adata_reprs)
        elif product == 'elementwise':
            adata_reprs = self._elementwise_product(adata_reprs)

        # Merge the anndata objects
        self.adata_reprs = ad.concat([adata_reprs], axis=1, join='outer', merge='same')

        # Train model
        params = {  # Defaults for spagcn
            'adata': self.adata_reprs,
            'adj': adj,
            'num_pcs': num_pcs,
            'init_spa': True,
            'init': "louvain",
            'tol': 5e-3,
            'lr': 0.05,
            'max_epochs': 200,
        }
        self.spagcn.train(**(params | kwargs))

        y_pred, prob = self.spagcn.predict()
        self.adata_reprs.obs["pred"] = y_pred
        self.adata_reprs.obs["pred"] = self.adata_reprs.obs["pred"].astype("category")
        self.adata_reprs.obs["prob"] = prob[np.arange(len(prob)), y_pred]

        adj_2d = calculate_adj_matrix(x=self.spot_x, y=self.spot_y, histology=False)
        refined_pred = refine(
            sample_id=self.adata_reprs.obs.index.tolist(),
            pred=self.adata_reprs.obs["pred"].tolist(),
            dis=adj_2d,
            shape=shape)
        self.adata_reprs.obs["refined_pred"] = refined_pred
        self.adata_reprs.obs["refined_pred"] = self.adata_reprs.obs["refined_pred"].astype(
            "category")

        if savepath is not None:  # Save results
            self.adata_reprs.write_h5ad(savepath)

        return self.adata_reprs, self.spagcn

    def plot_spatial_domains(self, x_col: str, y_col: str, plot_col="refined_pred", savepath=None):
        # Plot spatial domains
        ax = sc.pl.scatter(
            self.adata_reprs,
            alpha=1,
            x=x_col,
            y=y_col,
            color=plot_col,
            title=plot_col,
            show=False,
            size=100000 / self.adata_reprs.shape[0])
        ax.set_aspect('equal', 'box')
        ax.axes.invert_yaxis()
        if savepath is not None:
            plt.savefig(savepath, dpi=600)
        else:
            plt.show()
        plt.close()
