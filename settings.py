

fpath = "./data/"

# settings for data acquisition
simbad_top_k_objects = 15
simbad_refcodes_per_object = 25


# settings for main_reco
pca_G_ncomponents = 15

# this value needs tweaking more often, because the NLP gives back high dimensional data
# (with each new block of papers added)
pca_A_ncomponents = 150

mlp_iter = 1500
funksvd_iter = 1500
funksvd_latent_features = 25