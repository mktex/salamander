import pickle
from pprint import pprint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import sys
import time

from sklearn.decomposition import PCA
from sklearn import cluster as sklearn_clustering
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import data_extract as dext
from heimat import reco
import settings
import cache_manager as cmng
import portals_urls
import data_cleaning

pca_u, G = None, None
pca_a, A_star = None, None

A, U, MAP, txtclf = None, None, None, None
M = None
CLF_MLP = None
CLF_DBSCAN = None
D, PAPERS_LIST = None, None
UVT = None

papers_total, objects_articles_dict, objects_df = cmng.load_work()

pca_G_ncomponents = settings.pca_G_ncomponents
pca_A_ncomponents = settings.pca_A_ncomponents
mlp_iter = settings.mlp_iter
funksvd_iter = settings.funksvd_iter
funksvd_latent_features = settings.funksvd_latent_features

pd.set_option("max_rows", 50)

np.random.seed()


def dist_em(xs1, xs2):
    euklid = np.linalg.norm(xs1 - xs2)
    manhattan = sum(abs(e - s) for s, e in zip(xs1, xs2))
    return euklid, manhattan


def show_articles_by_group(group=0):
    """
        Shows paper_id corresponding to objects in some particular group
    :param group:
    :return:
    """
    global U
    r = U[U.group == group]
    articles = []
    for paper_id in r['OBJID'].values:
        articles.extend(MAP[paper_id])
    for paper_id in list(set(articles)):
        print("--------------------------------------------------")
        dext.get_paper(paper_id)


def show_first3_components(matrix, title="", start_at_index=0):
    """
    :param matrix: G or A_star matrices
    :param title:
    :param start_at_index: Depending on whether matrix is G or A_star, start_at_index differs (1, respectively 0)
    :return:
    """
    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection='3d')
    if matrix.shape[0] > 2000:
        indexes = np.random.randint(matrix.shape[0], size=2000)
    else:
        indexes = list(range(matrix.shape[0]))
    i, j, k = [start_at_index + t for t in range(0, 3)]
    ax.scatter3D(matrix[indexes, i], matrix[indexes, j], matrix[indexes, k], s=8, cmap='Greens', edgecolors='k')
    if title:
        plt.title(title)
    plt.show()
    plt.close()
    time.sleep(1)


def gen_matrix_G(ncomp=25):
    """
    matrix G of principal components for the object representation
        - generates the PCA form of matrix U
        - adds the OBJID value on the first column
    :param ncomp:
    :return:
    """
    global pca_u, G, U
    print("\n[x] PCA for matrix G:")
    pca_u = PCA(n_components=ncomp)
    U_matrix = U[list(filter(lambda x: x not in ["OBJID", "group"], U.columns))]
    G = pca_u.fit_transform(U_matrix.fillna(U_matrix.mean()).values)
    G = np.append(U['OBJID'].values.reshape(U.shape[0], 1), G, axis=1)
    print("[x] Explained variance ratio:")
    print(pca_u.explained_variance_ratio_)
    print("[x] Singular values:")
    print(pca_u.singular_values_)
    print("[x] Sum of variance:")
    print(np.sum(pca_u.explained_variance_ratio_))
    show_first3_components(G, title="First 3 principal components for G", start_at_index=1)


def gen_matrix_A_star(ncomp=25):
    """
    matrix A* of principal components for the article representation
        - generates the PCA form of matrix U
        - adds the OBJID value on the first column
    :param ncomp:
    :return:
    """
    global pca_a, A_star
    print("\n[x] PCA for matrix A:")
    pca_a = PCA(n_components=ncomp)
    A_star = pca_a.fit_transform(A.fillna(A.mean()).values[:, 1:])
    A_star = np.append(A['paper_id'].values.reshape(A_star.shape[0], 1), A_star, axis=1)
    print("[x] Explained variance ratio:")
    print(pca_a.explained_variance_ratio_)
    print("[x] Singular values:")
    print(pca_a.singular_values_)
    print("[x] Sum of variance:")
    print(np.sum(pca_a.explained_variance_ratio_))
    show_first3_components(A_star, title="First 3 principal components for A_star", start_at_index=1)


def get_indexes_articles_in_df(objid):
    """
        MAP contains the mapping between astronomical object ids and the paper ids
        returns the indexes in matrix A of object with objid
    :param objid:
    :return:
    """
    global A, MAP
    res = []
    for paper_id in MAP[objid]:
        record = A[A.paper_id == paper_id].index.values.tolist()
        if len(record) != 0:
            res.append(record[0])
        else:
            # ignoring for the moment if a paper id couldn't be found
            # (probably there was an exception at download phase)
            pass
    return res


def gen_matrix_M(balance_factor=3):
    """
        - construct matrix M by combining values from G and A_star
        - since a brute force would require too much time and would lead to overly unbalanced training set
        decided to build up by factor of 3 (balance_factor):
            - a portion of data is "as is", thus object data in G corresponds to data in A_star (by MAP)
            - a portion of data (3 times bigger) is "simulated" and contains objects to articles that are not associated
        - target value is set to 1 if association is given, otherwise 0
    :param balance_factor:
    :return:
    """
    global G, U, A_star, A
    M = []
    y = []
    print("Building matrix M, this will take a while .. ")
    for i in range(0, G.shape[0]):
        if i != 0 and i % int(0.1 * G.shape[0]) == 0:
            print("%.2f" % (100 * i / G.shape[0]) + "% of objects")
        r1 = G[i, 1:].tolist()
        object_id = U.values[i, 0]
        indexes_associations = get_indexes_articles_in_df(object_id)
        indexes_non_associations = list(filter(lambda k: k not in indexes_associations, range(A.shape[0])))
        indexes_non_associations = pd.Series(indexes_non_associations).sample(
            len(indexes_associations) * balance_factor).tolist()
        for j in indexes_associations + indexes_non_associations:
            r2 = A_star[j, 1:].tolist()
            M.append(r1 + r2)
            y.append(1 if j in indexes_associations else 0)
    M = np.array(M)
    return M, y


def gen_matrix_Mi(i):
    """
        Generates matrix Mi, that is the portion of Matrix M given an astronomical object id OBJID found at index i in G
        This is done by taking the record from G of object and combine it with all records from A_star,
            so that the calculation of probability P(Association | Gi, A_star) gets calculated for all A_star papers
    :param i:
    :return:
    """
    global U, G, A, A_star
    Mi = []
    yi = []
    r1 = G[i, 1:].tolist()
    for j in range(0, A_star.shape[0]):
        object_id = U.values[i, 0].encode("utf-8")
        articles_found_related = dext.objects_articles_dict[object_id]
        r2 = A_star[j, 1:].tolist()
        article_id = A.values[j, 0]
        target_value = int(article_id in articles_found_related)
        Mi.append(
            r1 + r2
        )
        yi.append(target_value)
    Mi = np.array(Mi)
    return Mi, yi


def get_confusion_matrix_stats(cm, i):
    """
        Given a Confusion Matrix cm, calculates precision, recall and F1 scores
    :param cm: confusion matrix
    :param i: position of the variable, for with the caculation be done
    :return: three statistics: precision, recall and the F1-Score
    """
    tp = cm[i, i]
    fp = np.sum(cm[i, :]) - tp
    fn = np.sum(cm[:, i]) - tp
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1_score


def check_mlp(x, y):
    global CLF_MLP
    print("+++++++++++++++++++++++++++++++++++++")
    labels_zuordnung_mlp = CLF_MLP.classes_
    beispiel_mlp_x = x
    beispiel_mlp_y = y
    y_true = np.array(beispiel_mlp_y)
    y_pred = np.array([labels_zuordnung_mlp[np.argmax(t)] for t in CLF_MLP.predict_proba(beispiel_mlp_x)])
    accuracy = (y_pred == y_true).mean()
    cm = confusion_matrix(y_true, y_pred, labels=labels_zuordnung_mlp)
    if True:
        print("Labels:", labels_zuordnung_mlp)
        print("Confusion Matrix:")
        print(cm)
        for i in range(0, len(cm)):
            precision, recall, f1_score = get_confusion_matrix_stats(cm, i)
            print("Label {} - precision {}, recall {}, f1_score {}: ".format(
                i, np.round(precision, 2), np.round(recall, 2), np.round(f1_score, 2)
            ))
        print("precision:", accuracy)
    print("+++++++++++++++++++++++++++++++++++++")


def show_object_details(object_id, article_indexes, pred_df=None, topk=10):
    """
        Shows associated papers for an object id according to predicted article_indexes
        # U expands categorical variables, so it has a dimension larger than dext.objects_df
    :param object_id:
    :param article_indexes:
    :param pred_df:
    :param topk:
    :return:
    """
    global A
    print("""
                            \nObject with ID: {}
    """.format(object_id))
    if pred_df is not None:
        print("[x] Predicted articles in pred_df:")
        print(pred_df)
    objid = object_id.encode("utf-8")

    url = "http://skyserver.sdss.org/dr16/en/tools/explore/Summary.aspx?id={}".format(
        object_id
    )

    print("[x] You can check the SkyServer Explore page at: ")
    print(url, "\n")
    print("[x] Compact form from original object pandas dataframe (objects_df as in data_extract.py):")
    print(dext.objects_df[dext.objects_df.OBJID == objid].transpose())
    print("\n[x] Showing maximum Top-{}:".format(topk))

    for k in range(0, min(len(article_indexes), topk)):
        print("*************************************************************************************")
        if pred_df is not None:
            print(pred_df.iloc[k])
        j = article_indexes[k]
        dext.get_paper(paper_id=A.paper_id.iloc[j])
        input(".....")


def apply_mlp(object_id=None):
    """
        uses trained MLP classifier to calculate probability P(Bij | ui, aj) for one object_id ui and all aj
        - uses construction of matrix Mi to achieve that, that is the portion of general matrix M for the object
    :param object_id:
    :return:
    """
    global U, G, CLF_MLP
    if object_id is None:
        i = pd.Series(range(0, G.shape[0])).sample(10).iloc[5]  # index of object id in matrices G, U
        object_id = U.OBJID.iloc[i]
    else:
        i = U[U.OBJID == object_id].index.values.tolist()[-1]
    print("\n[x] Object ID:", object_id)
    Mi, yi = gen_matrix_Mi(i)
    Mi = pd.DataFrame(Mi)
    print("[x] The portion of M matrix, corresponding to | ui | aj |, with j in [0, A_star.shape[0]]: ")
    print(Mi)
    preds = [np.round(t[1], 2) for t in CLF_MLP.predict_proba(Mi.values)]
    # print("\n[x] Predictions:")
    # print(preds)
    pred_df = pd.DataFrame(
        {
            "article_index": Mi.index.values.tolist(),
            "mlp_proba": preds,
            "associated": yi
        }
    )
    pred_df = pred_df.sort_values(by="mlp_proba", ascending=False)
    pred_df = pred_df[pred_df.mlp_proba > 0.5]
    pred_df = pred_df.reset_index(drop=True)
    print("\n[x] Summarised with a threshold for probabilty of 50%, that is P(Bij | ui, aj) > 0.5:")
    print(pred_df)
    articles_indexes = pred_df.article_index.values.tolist()
    print("")
    return object_id, articles_indexes, pred_df


def data_extraction():
    """
        with module dext original data is accessible: papers_total, objects_articles_dict, objects_df
    :return:
    """
    print("[x] Extracting data and creating matrices A, U and dictionary map MAP .. ")
    dext.run()
    A, U, MAP, txtclf = dext.load_matrices()
    return A, U, MAP, txtclf


####################### Constructing Matrix M and MLP model #######################

def construct_G_Astar_M_matrices():
    """
        uses above methods to construct training data M by combining G and A_star matrices
    :return:
    """
    global G, A_star, M, pca_A_ncomponents, pca_G_ncomponents
    print("[x] Generating PCA projections of:"
          "\n- matrices U (matrix G of astronomical objects)"
          "\n- and A (matrix A_star of related papers)")
    gen_matrix_G(ncomp=pca_G_ncomponents)
    # TODO: increase automatically pca_A_ncomponents if the explained variance drops to less than, for instance, 0.85
    gen_matrix_A_star(ncomp=pca_A_ncomponents)
    print("\n[x] Generating matrix M out of two parts "
          "| ui | aj | target {1 if related, 0 otherwise} ")
    M, y = gen_matrix_M()
    M = pd.DataFrame(M)
    target_col = M.shape[1]
    M[target_col] = y
    p = 100 * M[target_col].sum() / M.shape[0]
    print("The percentage of articles that are related directly (found at NED) at about: {}%".format(
        np.round(p, 2)
    ))
    print("[x] Done. Head(10):")
    print(M.head(10))
    print("")
    time.sleep(5)


def do_model_mlp():
    """
        perform modeling using MLP on constructed matrix M
    :return:
    """
    global M, CLF_MLP, labels_mlp
    print("\n[x] Performing MLP modeling with balancing by choosing combinations objects (matrix G) "
          "to articles (matrix A_star)"
          "\n(target == 1) and three times those not related")
    X = M.copy()
    indx = X.index.values.tolist()
    np.random.shuffle(indx)
    X = X.loc[indx]
    X, Y = X.values[:, :-1], X.values[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    mlp = MLPClassifier(max_iter=mlp_iter, verbose=True, early_stopping=False)
    CLF_MLP = mlp.fit(X_train, y_train)
    labels_mlp = CLF_MLP.classes_
    print("[x] Validation of MLP classifier results on test data:")
    check_mlp(x=X_test, y=y_test)
    input("enter to continue ...")
    print("")


def apply_clf_mlp(object_id="M  79", show_details=False):
    """
        applies trained CLF_MLP model on one object
    :param object_id:
    :return:
    """
    # example prediction using the MLP classifier to calculate probabability of association to any paper
    print("\n[x] Applying model MLP to object: {}".format(object_id))
    object_id, articles_indexes, pred_df = apply_mlp(object_id=object_id)  # * sig Ori
    if show_details:
        print("\n[x] Example prediction:")
        show_object_details(object_id, articles_indexes, pred_df)


####################### Constructing Matrix D, Clustering and FunkSVD models #######################
def perform_object_optimal_clustering():
    """
        performs a search for clustering with DBSCAN to be able to construct a reduced form of a "user-item" matrix
    :return:
    """
    global CLF_DBSCAN, U, G
    print("\n[x] Choosing an optimal parameter for DBSCAN object cluster classifier .. ")
    list_dist = []
    for i in range(0, G.shape[1] - 1):
        for j in range(i + 1, G.shape[1]):
            euclidean, _ = dist_em(G[i, 1:], G[j, 1:])
            list_dist.append(euclidean)

    number_of_clusters = []
    data_noise_list = []
    distribution_data_list = []
    param = np.linspace(0.01, pd.Series(list_dist).quantile(0.5), 100)
    eps_list = []
    for eps in param:
        clf = sklearn_clustering.DBSCAN(eps=eps, metric="euclidean")
        clf.fit(G[:, 1:])
        U["group"] = clf.labels_
        res = U.groupby("group").count()['OBJID']
        # don't consider any results under some lower threshold N clusters (no point in using later D)
        if res.shape[0] <= 5:
            continue
        eps_list.append(eps)
        distribution_data = res.loc[list(filter(lambda x: x != -1, res.index.values))]
        distribution_data = (distribution_data / distribution_data.sum()).mean()
        number_of_clusters.append(len(set(clf.labels_)))
        if -1 in res.index.values:
            data_noise = res.loc[-1]
            data_noise_list.append(data_noise)
        else:
            data_noise_list.append(0)
        distribution_data_list.append(distribution_data)

    param_choose = pd.DataFrame(
        {
            "nclusters": number_of_clusters,
            "eps": eps_list,
            "noise": data_noise_list,
            "distribution": distribution_data_list
        }
    )
    param_choose['score'] = [t1 + t1 * t3 - np.log10(t2) for t1, t2, t3 in
                             param_choose[["nclusters", "noise", "distribution"]].values]
    param_choose = param_choose.sort_values(by="score", ascending=False)
    param_choose = param_choose.reset_index(drop=True)
    param_choose_backup = param_choose.copy()

    param_choose = param_choose[param_choose.nclusters >= 3]
    param_choose = param_choose[param_choose.nclusters <= 15]
    q90 = param_choose.distribution.quantile(0.9)
    q10 = param_choose.distribution.quantile(0.1)
    q80 = param_choose.noise.quantile(0.8)
    param_choose = param_choose[param_choose.distribution >= q10]
    param_choose = param_choose[param_choose.distribution <= q90]
    param_choose = param_choose[param_choose.noise <= q80]

    if param_choose.shape[0] != 0:
        eps_choice = param_choose.eps.iloc[0]
    else:
        print("[x] Oops, restrictions on eps parameter where too strong! Relaxing only to nclusters in [2, 10]")
        param_choose = param_choose_backup.copy()
        param_choose = param_choose[param_choose.nclusters >= 2]
        param_choose = param_choose[param_choose.nclusters <= 10]
        eps_choice = param_choose.eps.iloc[0]

    print(param_choose)

    # visualization of choice for parameter epsilon
    plt.scatter(x=eps_list, y=number_of_clusters, s=5)
    plt.xlabel("optimal eps parameter: {}".format(eps_choice))
    plt.ylabel("expected number of clusters")
    plt.axvline(x=eps_choice, color='k', linestyle='--')
    plt.title("Choice for optimal eps parameter for DBSCAN")
    plt.show()

    print("[x] (Re-)building the classifier for clusters of objects with parameter eps={}".format(eps_choice))
    CLF_DBSCAN = sklearn_clustering.DBSCAN(eps=eps_choice, metric="euclidean")
    CLF_DBSCAN.fit(G[:, 1:])
    print("[x] Number of clusters:", len(set(CLF_DBSCAN.labels_)))
    U["group"] = CLF_DBSCAN.labels_
    print("Distribution of objects into groups:")
    print(U.groupby("group").count()['OBJID'])
    print("")
    time.sleep(5)


def construct_D_matrix():
    """
        Based on groupping with DBSCAN reduces data to centers of clusters and constructs D matrix such that:
            - Dkj is 1 if any object in cluster k has an association to article j and None otherwise
            - the value None is left for the method FunkSVD to be filled in
    :return:
    """
    global D, PAPERS_LIST, MAP, U
    print("[x] Constructing 'user-item' matrix D out of the clustered data .. ")
    PAPERS_LIST = list(set(A.paper_id.values))
    PAPERS_LIST.sort(reverse=True)
    D = []
    list_object_groups = list(set(CLF_DBSCAN.labels_))
    for cluster in list_object_groups:
        objects_in_cluster = U[U.group == cluster].OBJID.values.tolist()
        list_associated_articles = list(set(list(reduce(lambda a, b: a + b,
                                                        [MAP[objid] for objid in objects_in_cluster]))))
        D.append(
            [cluster]
            + list(map(lambda id: 1.0 if id in list_associated_articles else None, PAPERS_LIST))
        )

    D = pd.DataFrame(np.array(D), columns=(["cluster"] + PAPERS_LIST))
    print("First 10 rows out of {} (clusters):".format(D.shape[0]))
    D = D.fillna(value=np.nan)
    print(D.head(10))
    time.sleep(5)


def generate_UVT():
    """
        performs FunkSVD method on D matrix, it allows finding of two matrices U, VT such that:
        U @ VT ~ D
        - this allows using SVD to find the latent features
    :return:
    """
    global UVT, funksvd_latent_features
    print("\n[x] Generating U, VT matrices through FunkSVD for the final SVD model .. ")
    if funksvd_latent_features > D.shape[0]:
        funksvd_latent_features = D.shape[0]
    U_clusters, V_articles, sse_means = reco.mat.FunkSVD(D.values[:, 1:],
                                                         latent_features=funksvd_latent_features,
                                                         learning_rate=0.0001,
                                                         iters=funksvd_iter)

    D_approx = U_clusters @ V_articles
    u, s, vt = reco.mat.svd_dekomposition(D_approx)
    k = funksvd_latent_features
    s_new, u_new, vt_new = np.diag(s[:k]), u[:, :k], vt[:k, :]
    res = u_new @ s_new @ vt_new
    res = pd.DataFrame(res)
    print("[x] 'User-Item' matrix successfully build and ready for predictions:")
    print(res)
    UVT = res
    time.sleep(5)


def apply_clf_svd(object_id):
    """
        uses the newly obtained matrix UVT to make predictions on associations between object_id and all available papers
    :param object_id:
    :return:
    """
    global UVT
    print("[x] Applying SVD classifier .. ")
    # object_indexes = U[U.OBJID == object_id].index.values.tolist()
    object_cluster = U[U.OBJID == object_id].group.values.tolist()
    k = 0
    # corresponding to PAPERS_LIST ids
    preds = UVT.iloc[object_cluster[k]].values
    pred_dict = {}
    for i in range(len(preds)):
        pred_dict[PAPERS_LIST[i]] = preds[i]
    print("[x] Done.")
    return pred_dict


def apply_mlp_svd_voting(object_id="M  79", topk=10, return_df=False, ignore_articles=None):  # "* sig Ori"
    """
        combines both results from above (MLP and SVD) to deliver a final scoring on most probably intersting
        papers related to a given object
    :param object_id:
    :param topk: how many articles to recommend
    :param return_df: whether to return dataframe
    :param ignore_articles: list of article ids to ignore
    :return:
    """
    _, articles_indexes, pred_df = apply_mlp(object_id=object_id)
    articles_indexes_to_paper_ids = [A.paper_id.iloc[j] for j in articles_indexes]
    pred_dict = apply_clf_svd(object_id)
    pred_df['article_id'] = articles_indexes_to_paper_ids
    pred_df['cf_score'] = [pred_dict[paper_id] for paper_id in articles_indexes_to_paper_ids]
    pred_df['recommend'] = [(a + b) / 2.0 for a, b in pred_df[["mlp_proba", "cf_score"]].values]
    pred_df = pred_df.sort_values("recommend", ascending=False)
    pred_df = pred_df.reset_index(drop=True)

    if ignore_articles is not None:
        pred_df = pred_df[list(map(lambda id: id not in ignore_articles, pred_df["article_id"].values.tolist()))]

    if return_df:
        return pred_df.iloc[:topk]
    else:
        print(pred_df)
        articles_indexes = pred_df.article_index.values.tolist()
        show_object_details(object_id, articles_indexes, pred_df, topk=topk)


def check_voting_model():
    """
        Randomly choosing objects and run the model on them
        Checks typical precision, recall and f1_score for the expected associations
        precision is mostly pessimistic, since the new connections to papers,
            that are actually legitimate are not labeled as such.
    :return:
    """
    global A, MAP
    print("+++++++++++++++++++++++++++++++++++++")
    # reco.load_model(model_filepath)

    # A = reco.A; MAP = reco.MAP; U = reco.U; apply_mlp_svd_voting = reco.apply_mlp_svd_voting
    statistic = []
    for object_id in U.OBJID.sample(100):
        # object_id = "M  79"
        pred_df = apply_mlp_svd_voting(object_id=object_id, topk=20, return_df=True)
        expected = MAP[object_id]
        predicted = list(map(lambda k: A.paper_id.iloc[k], pred_df.article_index.values))
        actual = list(filter(lambda x: x in expected, predicted))
        tp = len(actual)
        fp = len(predicted) - tp  # np.sum(cm[i, :]) - tp
        fn = len(expected) - tp  # np.sum(cm[:, i]) - tp
        precision = (tp / (tp + fp)) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        f1_score = (2 * (precision * recall) / (precision + recall)) if ((precision + recall) != 0) else 0.0
        statistic.append([precision, recall, f1_score])
        print("[x] Current statistic:")
        statistics_df = pd.DataFrame(statistic, columns=["precision", "recall", "f1_score"])
        print(statistics_df.describe())
        print("[x] Progress: %.2f" % (100 * np.round(statistics_df.shape[0] / 100, 2)) + "%")
        print("\n\n")


def clean_matrix_U_invalid_object_id(_U):
    """
        in pipeline object_ids that couldn't be processed and end up having no OBJID ('')
        should be removed
    :param _U:
    :return: invalid records from _U where OBJID missing
    """
    return _U[_U.OBJID != ""]


def drop_dups(xdf):
    xdf = xdf.drop_duplicates()
    xdf = xdf.reset_index(drop=True)
    return xdf


def update_mlp_svd_model(object_id="* sig Ori", model_filepath="./data/salamander_model.pckl"):
    """
        - performs a complete pipeline to update the model based on current data
        - currently added data by download either from SDSS or SIMBAD will be included in model
    :param object_id:
    :param model_filepath:
    :return:
    """
    global A, U, MAP, txtclf

    data_cleaning.clean_object_names()
    data_cleaning.handle_objects_nan_values()

    A, U, MAP, txtclf = data_extraction()
    U = clean_matrix_U_invalid_object_id(U)
    U = drop_dups(U)
    A = drop_dups(A)
    construct_G_Astar_M_matrices()
    do_model_mlp()
    apply_clf_mlp(object_id=object_id, show_details=True)
    perform_object_optimal_clustering()
    construct_D_matrix()
    generate_UVT()
    apply_clf_svd(object_id=object_id)
    apply_mlp_svd_voting(object_id=object_id)
    save_model(model_filepath=model_filepath)

    xy = input("\n\n[x] check now voting model on data? (takes a while, checks 100 objects) [y/n]: ")
    if xy == "y" or xy == "":
        check_voting_model()


def save_model(model_filepath):
    global A, U, MAP, pca_u, G, pca_a, A_star, CLF_MLP, CLF_DBSCAN, D, PAPERS_LIST, UVT
    salamander_model = {
        "A": A,
        "U": U,
        "MAP": MAP,
        "G": G, "pca_u": pca_u,
        "A_star": A_star, "pca_a": pca_a,
        "CLF_MLP": CLF_MLP,
        "CLF_DBSCAN": CLF_DBSCAN,
        "D": D,
        "PAPERS_LIST": PAPERS_LIST,
        "UVT": UVT
    }
    with open(model_filepath, 'wb') as f:
        pickle.dump(salamander_model, f)
    print("Model successfully saved under {}".format(model_filepath))


def load_model(model_filepath):
    """
        Load model example:
            import main_reco as mr
            mr.load_model(mr.get_model_filepath_by_name('salamander'))
    """
    global A, U, MAP, pca_u, G, pca_a, A_star, CLF_MLP, CLF_DBSCAN, D, PAPERS_LIST, UVT
    global papers_total, objects_articles_dict, objects_df
    import os

    if not os.path.isfile(model_filepath):
        print("[x] Model doesn't exist at specified path: {}".format(model_filepath))
        print("[x] Need to first rebuild it.")
        return

    with open(model_filepath, 'rb') as f:
        salamander_model = pickle.load(f)

    A = salamander_model["A"]
    U = salamander_model["U"]
    MAP = salamander_model["MAP"]
    G = salamander_model["G"]
    pca_u = salamander_model["pca_u"]
    A_star = salamander_model["A_star"]
    pca_a = salamander_model["pca_a"]
    CLF_MLP = salamander_model["CLF_MLP"]
    CLF_DBSCAN = salamander_model["CLF_DBSCAN"]
    D = salamander_model["D"]
    PAPERS_LIST = salamander_model["PAPERS_LIST"]
    UVT = salamander_model["UVT"]

    print("Model loaded from {}".format(model_filepath))


def print_object_dict_as_df(object_dict_form):
    """
        instead of using pprint, uses pandas dataframe to better display content of object dictionary form
    :param object_dict_form:
    :return:
    """
    from copy import deepcopy
    pd.set_option("max_rows", None)
    xdict = deepcopy(object_dict_form)
    for xkey in xdict.keys():
        xdict[xkey] = [xdict[xkey]]
    print(pd.DataFrame(xdict).transpose())
    pd.set_option("max_rows", 50)


def update_matrices_for_new_object_id(object_U_dict_form):
    """
        In case a new object was just downloaded, it is possible to get recommendations
        without updating the model, but only for the articles that are already in A.
        This needs following steps:
            - adds object_id to matrix U, including the prediction of cluster (CLF_DBSCAN)
            - calculates the PCA form of the object and append it to G
    :param object_U_dict_form: dictionary for the object obtained by reading data from local cache with data_extraction()
    :return:
    """
    global U, pca_u, G

    # create record to append to U
    record = {}
    for col in U.columns:
        if col != "group":
            record[col] = [object_U_dict_form[col]]
    record = pd.DataFrame(record)
    record['group'] = [None]
    record = record[U.columns]

    # update U with the new object
    U = pd.concat([U, record])
    U = U.reset_index(drop=True)

    """
    At this point there is only one way to make a quess about the possible cluster a new object could be in,
        by measuring the distance to the vectors G from Matrix M and guessing the possible cluster of closest record 
        - currently G matrix to execute PCA need to average based on previous data (as done in gen_matrix_G)
    """
    print("[x] G matrix-form for object:")
    record_averaged = U.fillna(U.mean()).values[:, 1:-1][-1:, :]  # group feature is located on last position
    record_G_form = pca_u.transform(record_averaged)
    record_G_form = np.append(np.array([object_U_dict_form['OBJID']], dtype='object').reshape(1, -1),
                              record_G_form, axis=1)
    print(record_G_form)

    distances = [dist_em(g_record[1:], record_G_form[:, 1:])[0] for g_record in G]
    position_closest_distance = np.argsort(distances)[0]
    cluster_at_pos = U.group.iloc[position_closest_distance]
    print("[x] Based on closest distance to current data, most probable cluster would be:", cluster_at_pos)

    # update group in matrix U
    U.at[U.shape[0] - 1, "group"] = cluster_at_pos

    # update data in G
    G = np.append(G, record_G_form, axis=0)


def apply(object_id, model_filepath, topk=10, return_value=False, ignore_articles=None):
    """
        If the object is already in U, that is in the model, then the prediction get's done directly without
        need for constructing the G vector
    :param object_id:
    :param model_filepath:
    :return:
    """
    global U, A, MAP, G, A_star, CLF_MLP, CLF_DBSCAN, D, PAPERS_LIST, UVT

    _object_id = object_id

    print("================================ Recommending papers for {} ================================".format(
        object_id
    ))
    print("")

    load_model(model_filepath)
    object_id = search_object_name(object_id, U)

    if object_id is None:
        print("[x] Object ID is not in modeled U matrix")
        print("[x] Checking if the object is in cache .. ")
        time.sleep(3)
        # by accessing data_extraction() => matrices A, U and MAP are new, since all new objects are included.
        # => the only matrix that is important is U with the record for the newly downloaded object.
        _, U_new, _, _ = data_extraction()
        object_id = search_object_name(_object_id, U_new)  # has it been already downloaded but model not updated?
        if object_id is None:
            print("[x] Data for the object must be at least downloaded."
                  "\nUse either download path from SDSS or SIMBAD before continuing here.")
            print_help()
            return
        object_U_dict_form = U_new[U_new.OBJID == object_id].iloc[0].to_dict()
        print("\n[x] Object found: ")
        print_object_dict_as_df(object_U_dict_form)
        update_matrices_for_new_object_id(object_U_dict_form)

    if not return_value:
        apply_mlp_svd_voting(object_id=object_id, topk=topk, ignore_articles=ignore_articles)
    else:
        return apply_mlp_svd_voting(object_id=object_id, topk=topk, return_df=True, ignore_articles=ignore_articles)


def reformat_object_id(xstr):
    return ' '.join(list(filter(lambda x: x.strip() != "", xstr.split(" "))))


def describe_data(model_filepath="./data/salamander_model.pckl"):
    """
        - Prints out information about downloaded and processed data:
            - how many objects, papers
            - what are the most important papers
        - plots figures to show what type of objects are available
    :return:
    """
    global U, A, MAP, G, A_star, CLF_MLP, CLF_DBSCAN, D, PAPERS_LIST, UVT
    load_model(model_filepath)


def search_object_name(xname, input_df):
    """
        SIMBAD gives names in specific format, so "M   1" and not "M1"
        This function looks for the name without the spaces and returns the correct SIMBAD or SDSS expected format
    :param xname:
    :param U: matrix U as parameter so that it is possible to search either in U from saved model or in U_new
             (see method apply())
    :return:
    """
    res = list(filter(lambda x: reformat_object_id(xname) in reformat_object_id(x), input_df['OBJID'].values.tolist()))
    if len(res) != 0:
        return res[0]
    else:
        return None


def get_latest_articles_by_reference(objid, show=True, latestk=10):
    global A, MAP
    records = A[list(map(lambda x: x in MAP[objid], A.paper_id.values))].sort_values(by="paper_id").iloc[-latestk:]
    paper_ids = records.paper_id.values
    papers_data = list(filter(lambda x: x[0] in paper_ids, papers_total))
    if show:
        for elem in papers_data:
            print(elem)
    return pd.DataFrame(papers_data, columns=["paper_id", "title", "description", "url", "keywords"])


def append_latest_articles_to_output(dfresult, objid, latestk=10):
    papers_data = get_latest_articles_by_reference(objid, show=False, latestk=latestk)
    return pd.concat([
        dfresult,
        pd.DataFrame({
            "article_index": list(map(lambda x: A[A.paper_id == x].index[0], papers_data["paper_id"].values)),
            "mlp_proba": [0.0] * latestk,
            "associated": [1.0] * latestk,
            "article_id": papers_data["paper_id"].values,
            "cf_score": [0.0] * latestk,
            "recommend": [0.0] * latestk
        })
    ])


def get_coordinates(object_id):
    global objects_df
    res = objects_df[
        list(map(lambda x: reformat_object_id(x.strip().lower().decode("utf-8")) ==
                           reformat_object_id(object_id.lower()),
                 objects_df.OBJID.values))]
    if res.shape[0] == 0:
        print("Nix gefunden .. ")
        print("object_id:", object_id)
    return res[["PLUG_RA", "PLUG_DEC"]]


def get_model_filepath_by_name(name):
    return './data/{}_model.pckl'.format(name)

def print_help():
    print("\n[x] Example usage:"

          "\n Update current model with (newly) downloaded data:       "
          "\n\t   python main_reco.py update 'salamander'                   "

          "\n\n Apply model for a downloaded object                    "
          "\n\t   python main_reco.py papers 'salamander' '* sig Ori' 15    "
          "\n"
          "\n\t Check wether an object already exists in cache: "
          "       > main_reco.get_coordinates(object_id='M 22')"
          "\n")

    print("""[x] Alternatives:
                # Download data from SIMBAD, much less data available (most probably no spectra)
                # - for instance CDS Portal can be used to get the coordinates
                python read_simbad.py  05 34 31.940 +22 00 52.20

                # Download data from SDSS (https://dr12.sdss.org/advancedSearch); 
                #   as long as the object is in the database, then complete record is available
                # - use DR12 Advanced Search to get the download.txt for the region of the sky with the object
                # - replace download.txt under ./data/ 
                python read_sdss.py                
            """)


"""    
    This script is the main component to be used for:
        - rebuild the proposed voting model
        - check recommendation for astronomical objects (that were included in the model)
        - check recommendations for a set of astronomical objects 
        
    Object types, nomenclature, Abkürzungen:
        http://simbad.u-strasbg.fr/simbad/sim-display?data=otypes
        
    When adding new objects, might need to update clean_tags() method
    import data_cleaning
    keywords = data_cleaning.clean_tags(keywordlist=None, debug=True)
    # replacements are hardcoded
    records, papers_df = data_cleaning.get_records_with_tags('tuc')
"""
if __name__ == "__main__":

    print("\n")

    try:
        if sys.argv[1] not in ['update', 'papers', 'process']:
            print_help()

        # python main_reco.py update 'salamander'
        elif sys.argv[1] == 'update':
            model_path = get_model_filepath_by_name(sys.argv[2])
            update_mlp_svd_model(model_filepath=model_path)

        # python main_reco.py papers 'salamander' '* sig Ori' 15
        elif sys.argv[1] == 'papers':
            model_path = get_model_filepath_by_name(sys.argv[2])
            topk = 10 if len(sys.argv) == 4 else int(sys.argv[4])
            apply(object_id=sys.argv[3], model_filepath=model_path, topk=topk)

        # python main_reco.py process 'salamander' 15 'NGC 2566' 'NGC 2207' 'NGC 2974' 'NGC 2559' 'NGC 2292' 'NGC 2613' 'NGC 3115'
        elif sys.argv[1] == 'process':
            model_path = get_model_filepath_by_name(sys.argv[2])
            topk = int(sys.argv[3])
            object_id_list = sys.argv[4:]
            output_table = []
            paper_ids = []
            for object_id in object_id_list:
                res = get_coordinates(object_id)
                portals_urls.RA = res.PLUG_RA.iloc[0]
                portals_urls.DEC = res.PLUG_DEC.iloc[0]
                url_img, url_cas, url_simbad, url_cds, url_ned = portals_urls.show_urls()
                urls = "\n\n{}" \
                       "\n\n{}" \
                       "\n\n{}" \
                       "\n\n{}" \
                       "\n\n{}".format(
                    url_img, url_cas, url_simbad, url_cds, url_ned
                )
                pred_df = apply(object_id=object_id, model_filepath=model_path, topk=topk,
                                return_value=True, ignore_articles=paper_ids)
                pred_df = append_latest_articles_to_output(pred_df, object_id, latestk=topk)
                for i in range(0, pred_df.shape[0]):
                    article_index = pred_df.article_index.iloc[i]
                    paper_id = A.paper_id.iloc[article_index]
                    associated = pred_df.associated.iloc[i]
                    scores = str([pred_df.mlp_proba.iloc[i], pred_df.cf_score.iloc[i]])
                    if paper_id not in paper_ids:
                        res = dext.get_paper(paper_id=paper_id)
                        output_table.append([object_id, associated, scores,
                                             urls,
                                             res['title'] + "\n\n" + res['link'], res['description']])
                        paper_ids.append(paper_id)
            pd.DataFrame(output_table, columns=["identifier", "association", "scores",
                                                "url", "title", "description"]) \
              .to_csv("./data/output.csv", index=False)
            print("[x] List saved under ./data/output.csv")

        elif sys.argv[1] == 'help' or sys.argv[1] == 'info':
            print_help()

    except:
        import traceback

        traceback.print_exc()
        print_help()
