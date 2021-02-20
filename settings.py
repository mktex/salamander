

fpath = "./data/"

# settings for data acquisition
simbad_top_k_objects = 10
simbad_refcodes_per_object = 20


# settings for main_reco
pca_G_ncomponents = 10

# this value needs tweaking more often, because the NLP gives back high dimensional data
# (with each new block of papers added)
pca_A_ncomponents = 150

mlp_iter = 800
funksvd_iter = 800
funksvd_latent_features = 20