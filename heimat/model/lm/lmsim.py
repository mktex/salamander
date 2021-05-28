import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

import statsmodels.api as smapi

from heimat.statistik import stat_op


def mit_lm(xdf_input, xselekt_cols, xtarget, outputs=True, do_norm=True, shuffle_cols=False):
    """
        Erkläre die Beziehung zwischen Zielvariable und ausgewählten numerische Werte anhand eines LM
        Beispiel:
            xselekt_cols=['CareerSatisfaction', 'HoursPerWeek', 'JobSatisfaction', 'StackOverflowSatisfaction']
            mit_lm(xdfInput, xselekt_cols, xtarget = "Salary")

    :param xdf_input: Input
    :param xselekt_cols: Auswahl Features
    :param xtarget: Zielvariable
    :param outputs: Visualisierungen?
    :param do_norm: Normalisierung?
    :param shuffle_cols:
    :return: vis_x (DataFrame), r2_perf, rmse_perf, df_koeffizienten, grundwert
    """
    dfres = xdf_input.copy()
    dfres = dfres.reset_index(drop=True)
    if shuffle_cols:
        np.random.shuffle(xselekt_cols)
    X = dfres[xselekt_cols]
    y = dfres[xtarget]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30)  # random_state=42
    lm_model = LinearRegression(normalize=do_norm)
    lm_model.fit(X_train, y_train)
    xpred = lm_model.predict(X)
    vis_x = X.copy()
    vis_x["pred"] = xpred
    vis_x["IST"] = y
    vis_x["delta"] = [a - b for a, b in vis_x[["IST", "pred"]].values]

    r2_perf = r2_score(y, xpred)
    rmse_perf = np.sqrt(mean_squared_error(y, xpred))
    df_koeffizienten = pd.DataFrame({
        "feature": xselekt_cols,
        "gewicht": lm_model.coef_
    })
    grundwert = lm_model.intercept_
    df_koeffizienten = df_koeffizienten.sort_values(by="gewicht")
    df_koeffizienten = df_koeffizienten.reset_index(drop=True)
    if outputs:
        print("Grundwert:")
        print(grundwert)
        print(df_koeffizienten, "\n")
        print("delta = IST - pred")
        print(vis_x.describe())
        print("R^2 {}".format(r2_perf))
        print("RMSE {}".format(rmse_perf))
        _ = plt.subplots(figsize=(10, 8))
        vis_x.delta.hist(grid=False, bins=50)
        plt.title("delta = IST - Voraussage")
        plt.tight_layout()
        plt.show()
        vis_x.plot.scatter("IST", "pred", s=3)
        plt.xlabel(xtarget)
        plt.ylabel("pred({})".format(xtarget))
        plt.title("{} vs pred({})".format(xtarget, xtarget))
        plt.tight_layout()
        plt.show()

    return vis_x, r2_perf, rmse_perf, df_koeffizienten, grundwert


def kateg_felder_schwellenwert_auswahl_mit_lm(xdfcat_dummy, target, starte_ab=0.8, do_plot=True):
    """ Ergibt eine Liste von ausgewählten Features, die ein LM Modell optimieren
        target: eine Spalte wird in xdf_cat mit pd.merge() eingebunden
        BeispieL:
            print("Zielvariable:", target_col, "\n")
            target = dfnum[[target_col]]
            xLdf, dict_coefs, empfK = dabfrage.kateg_felder_schwellenwert_auswahl_mit_lm(xdfcat_dummy,
                                                                                         target,
                                                                                         starte_ab = 0.7,
                                                                                         do_plot=True)
            print("Parsimonie + R^2: ", empfK)
        Zugriff dann mit:
            xLdf.iloc[k].xselekt_cols

    :param xdfcat_dummy:
    :param target:
    :param starte_ab:
    :param do_plot:
    :return:
    """
    # Für die kategorialen Variablen: np.where((X.sum() > cutoff) == True)[0] ergibt die Spalten-Indexen
    X = xdfcat_dummy.copy()
    xtarget = target.columns[0]
    xliste = []
    dict_coefs = {}
    while starte_ab > 0.0:
        cutoff = starte_ab * X.shape[0]
        # muss '==' statt 'is' bleiben, sonst klappt die Selektion nicht
        # noinspection PyPep8
        xselekt_cols = [X.columns[t] for t in np.where((X.sum() > cutoff) == True)[0]]
        if len(xselekt_cols) == 0:
            starte_ab -= 0.01
            continue
        dflm = pd.merge(X[xselekt_cols], target, left_index=True, right_index=True)
        vis_x, r2_perf, rmse_perf, df_koeffizienten, grundwert = mit_lm(dflm, xselekt_cols, xtarget, outputs=False)
        xliste.append([cutoff, r2_perf, len(xselekt_cols), xselekt_cols, grundwert])
        dict_coefs[cutoff] = df_koeffizienten
        starte_ab -= 0.01
    xlist_df = pd.DataFrame(xliste)
    xlist_df.columns = ["cutoff", "r2_perf", "#cols", "xselekt_cols", "grundwert"]
    xlist_df = xlist_df[xlist_df.r2_perf > 0]
    xlist_df = xlist_df[xlist_df.grundwert > 0]
    xlist_df = xlist_df.sort_values(by="r2_perf", ascending=False)
    xlist_df = xlist_df.reset_index(drop=True)
    print(xlist_df)
    if do_plot:
        xlist_df.plot.scatter("r2_perf", "cutoff", s=4)
        plt.tight_layout()
        plt.ylabel("Schwellenwert")
        plt.xlabel("R^2")
        plt.show()
    u = xlist_df['r2_perf'] * [10.0 / t for t in xlist_df['#cols'].values]
    # wenn u berechnet wurde, dann existiert auch u.max() => u[u == u.max()].shape[0] != 0
    # noinspection PyUnresolvedReferences
    return xlist_df, dict_coefs, (u[u == u.max()]).index[0]


def simulation_lm(xdf_input, xselekt_cols, xtarget, nsim=300):
    """ Gegeben eine Auswahl an Features,
        führt den LM mehrmals aus und findet die Durchschnittliche Werte für die Parameters des Modells

    :param xdf_input:
    :param xselekt_cols:
    :param xtarget:
    :param nsim:
    :return:
    """
    if True:
        xliste = []
        for simk in range(nsim):
            _ = mit_lm(xdf_input, xselekt_cols, xtarget, outputs=False, do_norm=False, shuffle_cols=True)
            vis_x, r2_perf, rmse_perf, koeffizienten, grundwert = _
            xliste.append([r2_perf, rmse_perf, koeffizienten, grundwert])
        xdict_cols = {}
        grundwerte = []
        for _, _, xr, grundwert in xliste:
            grundwerte.append(grundwert)
            for xcol, gewicht in xr.values:
                if xcol not in list(xdict_cols.keys()):
                    xdict_cols[xcol] = [gewicht]
                else:
                    xdict_cols[xcol].append(gewicht)
        xdf_cols_stats = pd.DataFrame(xdict_cols)
        xdf_cols_stats["Grundwert"] = grundwerte
        print(xdf_cols_stats.describe())

    plt.figure(figsize=(20, 10))
    xdf_cols_stats.hist(bins=25, grid=False)
    plt.autoscale(enable=True)
    plt.tight_layout()
    plt.show()

    print("Durchschnitte: \n", xdf_cols_stats.mean())
    parameter_means = xdf_cols_stats.mean().to_dict()

    return xdf_cols_stats, parameter_means


def do_lm_statsmodels(xdf_input, xselekt_cols, target_col):
    """

    :param xdf_input:
    :param xselekt_cols:
    :return:
    """
    X = xdf_input[xselekt_cols]
    y = xdf_input[target_col]

    X2 = smapi.add_constant(X)
    est = smapi.OLS(y, X2)
    est2 = est.fit()
    print(est2.summary())

    significant_table = pd.DataFrame({
        'feature': est2.pvalues.index.values.tolist(),
        'pValue': est2.pvalues.values.tolist(),
        'significant': [t <= 0.1 for t in est2.pvalues.tolist()]
    })
    significant_table = significant_table[significant_table.feature != "const"]
    print("\n[x] Signifikanztabelle:")
    print(significant_table[significant_table.significant == True])

    return est2, significant_table


def do_lm_berechnung(xrecord, xtarget, parameter_means, voraussage=False):
    """

    :param xrecord:
    :param xtarget:
    :param parameter_means:
    :param voraussage:
    :return:
    """
    xformel = []
    for xkey in xrecord.keys():
        if xkey != xtarget:
            xformel.append(
                [xkey, parameter_means[xkey], xrecord[xkey]]
            )
    xformel = pd.DataFrame(xformel)
    xformel.columns = ["feature", "gewicht", "wert"]
    xformel["gewicht * wert"] = [t * v for t, v in xformel[["gewicht", "wert"]].values]
    xformel = xformel.sort_values(by="gewicht")
    xformel = xformel.reset_index(drop=True)
    x = xformel["gewicht * wert"].sum()
    b = parameter_means["Grundwert"]
    if not voraussage:
        return xformel, x, b
    else:
        return x + b


def zeige_lm_berechnung(xdf_input, xselekt_cols, xtarget, parameter_means):
    """

    :param xdf_input:
    :param xselekt_cols:
    :param xtarget:
    :param parameter_means:
    :return:
    """
    xrecord = xdf_input[xselekt_cols + [xtarget]].sample(10).iloc[5].to_dict()
    xformel, x, b = do_lm_berechnung(xrecord, xtarget, parameter_means)
    print(xformel)
    print("\n")
    print("\tSUMME       : ", x)
    print("\tGrundwert   : ", b)
    print("\t          =>  ", x + b)
    print("\tTatsächlich : ", xrecord[xtarget])


def abzug_features_optimaler_lm_parameter_set(empfK, xLdf, num_cols, target_col, id_col):
    """
        empfK, xLdf kommen vom Aufruf kateg_felder_schwellenwert_auswahl_mit_lm
        Ergibt eine Liste mit Features:
            - Dummy Features selektiert
            - Numerische Features
    :return:
    """
    from functools import reduce
    cols_filter = list(filter(lambda x: x is not None, [target_col, id_col]))
    # Optimale Auswahl der Lösung rund um empK Index
    # Verwende 1-4 Resultate rund um empfK Index und mache daraus eine Liste bester Features
    xmin = (empfK - 3) if (empfK - 3) > 0 else 0
    xmax = (empfK + 3) if (empfK + 3) < xLdf.shape[0] else (xLdf.shape[0] - 1)
    best_dummys = list(set(reduce(lambda a, b: a + b, xLdf.iloc[:(empfK + 3)].xselekt_cols.tolist())))
    xselekt_cols = best_dummys + num_cols
    xselekt_cols = list(filter(lambda x: x not in cols_filter, xselekt_cols))
    print("Auswahl:", xselekt_cols)
    return xselekt_cols


def performance(xerwartet, xpred):
    """

    :return:
    """
    getvalpred = lambda predval: 1.0 if predval >= 0.5 else 0.0
    cm = np.zeros((2, 2))
    for i in [0, 1]:
        for j in [0, 1]:
            cm[i, j] = len(list(filter(lambda x: getvalpred(x[0]) == i and x[1] == j,
                                       zip(xpred, xerwartet))))

    precision0, recall0, f1_score0 = stat_op.get_confusion_matrix_stats(cm, 0)
    precision1, recall1, f1_score1 = stat_op.get_confusion_matrix_stats(cm, 1)
    accuracy = (np.array(list(map(lambda x: getvalpred(x), xpred))) == np.array(xerwartet)).mean()
    print("""
    Label "<=50K":
        Precision {}, Recall {}, F1-Score {}

    Label ">50K":
        Precision {}, Recall {}, F1-Score {}

    Accuracy {}
    """.format(precision0, recall0, f1_score0,
               precision1, recall1, f1_score1, accuracy))
