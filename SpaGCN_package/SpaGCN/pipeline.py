import matplotlib.pyplot as plt
import scanpy as sc
import anndata as ad
from . SpaGCN import SpaGCN
from . util import prefilter_genes, prefilter_specialgenes, search_l, refine
from . calculate_adj import calculate_adj_matrix


class Pipeline:
    """
    Eventually will incorporate histology autoencoder.
    """

    def __init__(self, gene_adata, image, spot_x, spot_y, adata_list=None, **kwargs):
        self.gene_adata = gene_adata
        self.adata_list = adata_list
        self.adata_all = None
        self.image = image
        self.spot_x = spot_x
        self.spot_y = spot_y
        self.spagcn = SpaGCN(**kwargs)

    def fit_transform(self, alpha=1, beta=50, min_cells=3, p=.5,
                      shape='hexagon', savepath=None, **kwargs):
        """
        p: percentage of total expression contributed by neighborhoods.
        """
        # Calculate adjacency matrix based on histology image
        adj = calculate_adj_matrix(
            x=self.spot_x,
            y=self.spot_y,
            x_pixel=self.spot_x,
            y_pixel=self.spot_y,
            image=self.image,
            histology=True,
            alpha=alpha,
            beta=beta
        )

        # Preprocess the gene data
        self.gene_adata.var_names_make_unique()
        prefilter_genes(self.gene_adata, min_cells=min_cells)
        prefilter_specialgenes(self.gene_adata)
        sc.pp.normalize_per_cell(self.gene_adata)
        sc.pp.log1p(self.gene_adata)

        # Set the l
        self.spagcn.l = search_l(p, adj, start=0.01, end=1000, tol=0.01, max_run=100)

        # Merge the anndata objects
        self.adata_all = ad.concat([self.gene_adata, *self.adata_list], axis=1, join='outer',
                                   merge='same')

        default_params = {  # Defaults for spagcn
            'init_spa': True,
            'init': "louvain",
            'tol': 5e-3,
            'lr': 0.05,
            'max_epochs': 200
        }

        # Train model
        self.spagcn.train(adata=self.adata_all, adj=adj, **default_params.update(kwargs))

        y_pred, prob = self.spagcn.predict()
        self.adata_all.obs["pred"] = y_pred
        self.adata_all.obs["pred"] = self.adata_all.obs["pred"].astype("category")
        self.adata_all.obs["prob"] = prob

        adj_2d = calculate_adj_matrix(x=self.spot_x, y=self.spot_y, histology=False)
        refined_pred = refine(
            sample_id=self.adata_all.obs.index.tolist(),
            pred=self.adata_all.obs["pred"].tolist(),
            dis=adj_2d,
            shape=shape)
        self.adata_all.obs["refined_pred"] = refined_pred
        self.adata_all.obs["refined_pred"] = self.adata_all.obs["refined_pred"].astype('category')

        if savepath is not None:  # Save results
            self.adata_all.write_h5ad(savepath)

        return self.adata_all, self.spagcn

    def plot_spatial_domains(self, x_col: str, y_col: str, plot_col="refined_pred", savepath=None):
        # Set colors used
        plot_color = [
            "#F56867",
            "#FEB915",
            "#C798EE",
            "#59BE86",
            "#7495D3",
            "#D1D1D1",
            "#6D1A9C",
            "#15821E",
            "#3A84E6",
            "#997273",
            "#787878",
            "#DB4C6C",
            "#9E7A7A",
            "#554236",
            "#AF5F3C",
            "#93796C",
            "#F9BD3F",
            "#DAB370",
            "#877F6C",
            "#268785"]
        # Plot spatial domains
        num_celltype = len(self.adata_all.obs[plot_col].unique())
        assert num_celltype <= len(plot_color)
        self.adata_all.uns[plot_col + "_colors"] = list(plot_color[:num_celltype])
        ax = sc.pl.scatter(
            self.adata_all,
            alpha=1,
            x=x_col,
            y=y_col,
            color=plot_col,
            title=plot_col,
            color_map=plot_color,
            show=False,
            size=100000 / self.adata_all.shape[0])
        ax.set_aspect('equal', 'box')
        ax.axes.invert_yaxis()
        if savepath is not None:
            plt.savefig(savepath, dpi=600)
        else:
            plt.show()
        plt.close()
