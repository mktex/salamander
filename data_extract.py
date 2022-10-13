from functools import reduce
import pickle
from pprint import pprint

import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from heimat.nlp.textverarbeitung_nltk import TXTVerarbeitung
from heimat.transform import kateg

import cache_manager as cmng
import data_cleaning

_pipe = None

# Pandas DataFrame of articles in numeric format representation
A = None
keyword_features = None
nlp_features = None

# Pandas DataFrame of object features (spectrogram data, redshifts, type of object)
U = None

# Object Mapping dictionary from astronomical objects to articles
MAP = None

papers_total, objects_articles_dict, objects_df = cmng.load_work()
objects_df = objects_df.reset_index(drop=True)

"""
Example usage:
    import data_cache as dc
    dc.run()
    A, U, MAP, txtclf = dc.load_matrices()
"""


def reload_objects_papers():
    """
        relaod objects to work on
    """
    global papers_total, objects_articles_dict, objects_df
    papers_total, objects_articles_dict, objects_df = cmng.load_work()
    objects_df = objects_df.reset_index(drop=True)


def get_record_map_keywords(keyword_record):
    global keyword_features
    return [int(t in keyword_record) for t in keyword_features]


def nlp_prep(X_train, y_train):
    """
        Performs a chain calculation to transform input text rows (X_train) into
        one Matrix representation after:
        - url replacements
        - text normalization through lowering, but keeping all-caps words such as 'US' as they are)
        - tokenization
        - normalization of tokens with a lemmatizer
        - removal of stop words
        - counts of remaining tokens and transformation to tf-idf representation
        - finally dimensionality reduction using PCA
    :param X_train: Input dataset, rows of text
    :param y_train: target variable; can be multidimensional
    :return: X_transformed_train (Matrix representation) and target variables (to make sure sorting ist kept the same)
    """
    global _pipe
    pca = PCA(n_components=20)
    txt_train = TXTVerarbeitung(X_train, y_train)
    pipe = Pipeline([
        ('txt_v_nltk', txt_train),
        ('pca_output', pca)
    ])
    X_transformed_train = pipe.fit_transform(X_train)
    _pipe = pipe
    return X_transformed_train, y_train


def prep_matrix_A():
    """
        Preparing matrix A of articles. Provides access to NLP processed data
    """
    global papers_total, A, keyword_features, nlp_features

    if True:
        papers_df = pd.DataFrame(papers_total)
        papers_df.columns = ['refcode', "title", "description", "link", "keywords"]
        print(papers_df.sample(10))
        papers_df["keywords"] = [data_cleaning.clean_tags(t) for t in papers_df["keywords"].tolist()]
        keyword_features = list(set(list(reduce(lambda a, b: a + b, papers_df["keywords"].tolist()))))
        keyword_features.sort()

    # Construction of matrix A of articles
    A = []
    for k in range(0, papers_df.shape[0]):
        kws = papers_df.iloc[k].keywords
        paper_id = papers_df.iloc[k].refcode
        A.append(
            [paper_id] + get_record_map_keywords(kws)
        )
    A = pd.DataFrame(A)
    A.columns = ["paper_id"] + keyword_features

    # Append NLP processed text from title and description
    X_train = [t1 + " " + t2 for t1, t2 in papers_df[["title", "description"]].values.tolist()]
    X_transformed, y_train = nlp_prep(X_train, np.zeros(len(X_train)))
    X_transformed = pd.DataFrame(X_transformed)
    X_transformed.columns = ["NLP{}".format(k) for k in range(0, X_transformed.shape[1])]
    nlp_features = X_transformed.columns

    # put all together
    A = pd.concat([A, X_transformed], axis=1)

    print("[x] Example out of matrix of articles (A):")
    print(A.sample(10).iloc[5])
    print("[x] Article matrix A shape:", A.shape)


def safe_decoding(t):
    res = t
    try:
        res = t.decode("utf-8")
    except:
        pass
    return res


def prep_matrix_U():
    global objects_df, U

    objects_df["CLASS"] = [b'NA' if t == b'' else t for t in objects_df["CLASS"].tolist()]
    objects_df["SUBCLASS"] = [b'NA' if t == b'' else t for t in objects_df["SUBCLASS"].tolist()]
    objects_df["CLASS"] = [safe_decoding(t) for t in objects_df["CLASS"].tolist()]
    objects_df["SUBCLASS"] = [safe_decoding(t) for t in objects_df["SUBCLASS"].tolist()]

    U = kateg.kateg2dummy(objects_df[["CLASS", "SUBCLASS"]])
    U = pd.concat(
        [
            objects_df[['OBJID', 'SN_MEDIAN_ALL', 'Z', 'SPEC_U', 'SPEC_G',
                        'SPEC_R', 'SPEC_I', 'SPEC_Z', 'Ly_alpha', 'N_V 1240', 'C_IV 1549',
                        'He_II 1640', 'C_III] 1908', 'Mg_II 2799', '[O_II] 3725', '[O_II] 3727',
                        '[Ne_III] 3868', 'H_epsilon', '[Ne_III] 3970', 'H_delta', 'H_gamma',
                        '[O_III] 4363', 'He_II 4685', 'H_beta', '[O_III] 4959', '[O_III] 5007',
                        'He_II 5411', '[O_I] 5577', '[N_II] 5755', 'He_I 5876', '[O_I] 6300',
                        '[S_III] 6312', '[O_I] 6363', '[N_II] 6548', 'H_alpha', '[N_II] 6583',
                        '[S_II] 6716', '[S_II] 6730', '[Ar_III] 7135']],
            U
        ],
        axis=1
    )
    U["OBJID"] = [safe_decoding(t) for t in objects_df["OBJID"].tolist()]
    print("Preparing U matrix ended successfully:")
    print(U.sample(10))


def prep_matrix_MAP():
    global MAP
    MAP = {}
    for xkey in objects_articles_dict.keys():
        MAP[safe_decoding(xkey)] = objects_articles_dict[xkey]


def save_data():
    global A, U, MAP, keyword_features, nlp_features, _pipe
    xdict = {
        "A": A,
        "U": U,
        "MAP": MAP,
        "keyword_features": keyword_features,
        "nlp_features": nlp_features,
        "_pipe": _pipe
    }
    with open("./data/data_cache.pckl", "wb") as f:
        pickle.dump(xdict, f)
    print("\nSuccessfully cached data to: ./data/data_cache.pckl")


def load_matrices():
    global A, U, MAP, keyword_features, nlp_features, _pipe

    # data_cache.pckl gets verwritten each time; run() and main_reco.data_extraction() methods
    with open("./data/data_cache.pckl", "rb") as f:
        xdict = pickle.load(f)

    A = xdict["A"]
    U = xdict["U"]
    MAP = xdict["MAP"]
    keyword_features = xdict["keyword_features"]
    nlp_features = xdict["nlp_features"]
    _pipe = xdict["_pipe"]
    return A, U, MAP, (keyword_features, nlp_features, _pipe)


def run():
    global A, U, MAP, keyword_features, nlp_features, _pipe
    reload_objects_papers()
    prep_matrix_A()
    prep_matrix_U()
    prep_matrix_MAP()
    save_data()


def get_paper(paper_id):
    global papers_total
    df = pd.DataFrame(papers_total)
    df.columns = ["refcode", "title", "description", "link", "keywords"]
    res = df[df.refcode == paper_id]
    if res.shape[0] != 0:
        pprint(res.iloc[0].to_dict())
        return res.iloc[0]
    return None


def dist_coord(c1, c2):
    """
        c1 = SkyCoord('05h55m10.30536s', '+07d24m25.4304s', frame='icrs')
        c2 = SkyCoord('05h55m10.9s', '+07d26m08s', frame='icrs')
    """
    sep = c1.separation(c2)
    return sep.arcsecond


def hmsdms_to_deg(ra='05h56m0s', dec='+07d25m0s'):
    """
        Determine Coordinates in degree from format {RA in HMS, DEC in DMS} to {RA and DEC in degrees}
        This can then be copy/pasted in http://simbad.u-strasbg.fr/simbad/sim-fcoo
        Example usage:
         _ = hmsdms_to_deg(ra='05h55m10.30536s', dec='+07d24m25.4304s')
         Gets Coordinates for Betelgeuse in degrees
    :param ra:
    :param dec:
    :return: (ra, dec) coordinates in degrees as well as the SkyCoord object
    """
    xcoord = SkyCoord(ra, dec, frame='icrs')
    # print("{} {}".format(xcoord.ra.deg, xcoord.dec.deg))
    return (xcoord.ra.deg, xcoord.dec.deg), xcoord
