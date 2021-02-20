from sklearn.neighbors import LocalOutlierFactor
import pandas as pd

def ausreisser_lof(xs):
    """
    :param xs: Pandas Series
    :return: die Outliers mit entsprechendem Index
    """
    # Ergibt ein Pandas Series mit entsprechenden LOF Markierung
    clf = LocalOutlierFactor(n_neighbors=2)
    xindex = xs.index
    xdf = pd.DataFrame({'werte': xs})
    res = pd.DataFrame({
        "LOF": clf.fit_predict(xdf)
    })
    res.index = xindex
    return res[res.LOF == -1]
