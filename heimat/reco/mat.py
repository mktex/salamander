import pandas as pd
import numpy as np
import progressbar
import matplotlib.pyplot as plt


def pivot_i(xdfinput):
    """
    Ergibt eine Matrize mit Datensätze für jeden id_n und Spalten für jeden id_m
    Ähnliche Funktionalität wie "pivot"
    Daten in Format: | id_n | id_m | r |
        id_n, id_m können die Nutzer IDs und IDs der sortierten Objekten / Einheiten
        Wert kann entweder binomial oder entsprechend Ranking-System
    :param:
    :return: m
    """
    xdf = xdfinput.copy()
    feature_namen = xdf.columns[0]
    xdf.columns = ['id_m', 'id_n', 'r']
    m = None
    liste_ids = list(set(xdf['id_m']))
    bar = progressbar.ProgressBar(maxval=len(liste_ids),
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    cnt = 0
    for id in liste_ids:
        cnt += 1
        bar.update(cnt)
        records = xdf[xdf['id_m'] == id]
        if records["id_n"].drop_duplicates().shape[0] != records.shape[0]:
            print("[x] WARNUNG: mehrfache Werte für selbe Kombination (id_m: {})".format(id))
        xadd_df = records[["id_n", "r"]].sort_values(by="id_n").transpose().loc[['r']]
        xcols_namen = list(xadd_df.columns)
        xadd_df[feature_namen] = [id] * xadd_df.shape[0]
        xadd_df = xadd_df[[feature_namen] + xcols_namen]
        if m is None:
            m = xadd_df
        else:
            m = pd.concat([m, xadd_df])
    return m


def pivot_ii(xdf, xcol_m, xcol_n, xcol_r):
    """
        Ähnlich pivot_i aber ohne den Vorteil der letzter Wert in der Reihe zu kriegen
        Implementierung aus Pandas deutlich performanter
    :param xdf:
    :param xcol_m:
    :param xcol_n:
    :param xcol_r:
    :return:
    """
    return xdf.groupby([xcol_m, xcol_n])[xcol_r].max().unstack()


def FunkSVD(ratings_mat, latent_features=4, learning_rate=0.0001, iters=100):
    """
    This function performs matrix factorization using a basic form of FunkSVD with no regularization
    Beispiel: ratings_mat[:5,:]
    np.matrix([[10., 10., 10., 10.],
            [10.,  4.,  9., 10.],
            [ 8.,  9., 10.,  5.],
            [ 9.,  8., 10., 10.],
            [10.,  5.,  9.,  9.]])

    _ = mat.FunkSVD(ratings_mat, latent_features=4, learning_rate=0.001, iters=1000)
    user_mat, movie_mat, sse_means = _

    :param ratings_mat: - (numpy array) a matrix with users as rows, movies as columns, and ratings as values
    :param latent_features: - (int) the number of latent features used
    :param learning_rate: - (float) the learning rate
    :param iters: - (int) the number of iterations

    :return
        user_mat  - (numpy array) a user by latent feature matrix
        movie_mat - (numpy array) a latent feature by movie matrix
    """

    n_users = ratings_mat.shape[0]
    n_movies = ratings_mat.shape[1]
    user_mat = np.random.rand(n_users, latent_features)
    movie_mat = np.random.rand(latent_features, n_movies)

    print("Optimization Statistics")
    print("Iterations | Mean Squared Error ")

    sse_means = []
    for i in range(0, iters):
        sse_accum = 0
        sse_means_iter = []
        # For each user-movie pair
        for uid in range(0, n_users):
            for mid in range(0, n_movies):
                # if the rating exists
                if not np.isnan(ratings_mat[uid, mid]):
                    # compute the error as the actual minus the dot product of the user and movie latent features
                    y = ratings_mat[uid, mid]
                    u_i = user_mat[uid, :]
                    v_i = movie_mat[:, mid]
                    d_sse_u_i = -2 * (y - u_i @ v_i) * v_i
                    d_sse_v_i = -2 * (y - u_i @ v_i) * u_i
                    u_i_new = u_i - learning_rate * d_sse_u_i
                    v_i_new = v_i - learning_rate * d_sse_v_i
                    # Keep track of the total sum of squared errors for the matrix
                    sse_accum += (y - u_i @ v_i) ** 2
                    sse_means_iter.append(sse_accum)
                    # update the values in each matrix in the direction of the gradient
                    user_mat[uid, :] = u_i_new
                    movie_mat[:, mid] = v_i_new
        sse_means.append(pd.Series(sse_means_iter).mean())
        if i % int(0.1 * iters) == 0:
            print("{}\t|{}".format(i, sse_means[-1]))

    pd.Series(sse_means[(-int(iters / 2)):]).plot()
    plt.title("SSE means in FunkSVD")
    plt.xlabel("iteration")
    plt.ylabel("SSE")
    plt.show()

    return user_mat, movie_mat, sse_means


def svd_dekomposition(A):
    u, s, vt = np.linalg.svd(A)
    return u, s, vt


def check_latent_features_visualize(u, s, vt):
    """
        user_item_matrix dataframe in format: user_id | .... item_id ....
    """
    global user_item_matrix
    num_latent_feats = np.arange(10, 700 + 10, 20)
    sum_errs = []

    for k in num_latent_feats:
        # restructure with k latent features
        s_new, u_new, vt_new = np.diag(s[:k]), u[:, :k], vt[:k, :]

        # take dot product
        user_item_est = np.around(np.dot(np.dot(u_new, s_new), vt_new))

        # compute error for each prediction to actual value
        diffs = np.subtract(user_item_matrix.values[:, 1:], user_item_est)

        # total errors and keep track of them
        err = np.sum(np.sum(np.abs(diffs)))
        sum_errs.append(err)

    plt.plot(num_latent_feats, 1 - np.array(sum_errs) / user_item_matrix.shape[0])
    plt.xlabel('Number of Latent Features')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Number of Latent Features')


"""
Beispiel Pred:
idx = np.where(index_user_liste == id)[0][0]
preds = np.dot(user_mat[idx, :], movie_mat)
indices = preds.argsort()[-10:][::-1] # ergibt Top 10 Movies

"""
