# -*- coding: utf-8 -*-
"""main.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1iqHqqEZo5818JH-oU91Oxp7kMsjxP4-V
"""

import pandas as pd
import h5py
import zarr
import seaborn as sns
import matplotlib.pyplot as plt
import scanpy as sc
import json
import numpy as np

def load_data():

    transcripts_df = pd.read_parquet('data/transcripts.parquet')
    transcripts_zarr = zarr.open('data/transcripts.zarr.zip', mode='r')
    cell_features = h5py.File('data/cell_feature_matrix.h5', 'r')
    cell_boundaries_df = pd.read_parquet('data/cell_boundaries.parquet')

    with open('data/gene_panel.json', 'r') as f:
        gene_panel = json.load(f)

    adata = sc.read_10x_h5('data/4plex_40k_fixed_tumor_ovary_skin_breast_lung_multiplex_OvarianCancer_BC1_count_sample_filtered_feature_bc_matrix.h5')
    print("Data loading complete.")
    return transcripts_df, transcripts_zarr, cell_features, cell_boundaries_df, gene_panel, adata

def preprocess_data(adata):

    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    adata = adata[adata.obs.pct_counts_mt < 5, :]
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata = adata[:, adata.var.highly_variable]
    adata.write('processed_data/preprocessed_data.h5ad')
    print("Preprocessing complete.")
    return adata

def perform_analysis(adata):

    sc.tl.pca(adata, svd_solver='arpack')
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, resolution=0.5)
    print("Analysis complete.")
    return adata

def visualize_results(adata, transcripts_df):

    sc.pl.pca_variance_ratio(adata, log=True)
    sc.pl.umap(adata, color=['gene_name', 'leiden'])

    genes_to_visualize = ['CFB', 'EPCAM', 'LAMP1', 'PLEK', 'ITGAX', 'CD163', 'EGFL7', 'SPARC']
    for gene in genes_to_visualize:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=transcripts_df['x'], y=transcripts_df['y'], hue=transcripts_df[gene])
        plt.title(f"Spatial Distribution of {gene}")
        plt.show()

    sc.pl.rank_genes_groups(adata, n_genes=20, sharey=False)
    print("Visualization complete.")

def main():

    transcripts_df, transcripts_zarr, cell_features, cell_boundaries_df, gene_panel, adata = load_data()
    adata = preprocess_data(adata)
    adata = perform_analysis(adata)
    visualize_results(adata, transcripts_df)
    adata.write('processed_data/final_processed_data.h5ad')

if __name__ == "__main__":
    main()