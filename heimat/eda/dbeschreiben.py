from contextlib import contextmanager
from functools import reduce

import numpy as np
import pandas as pd


@contextmanager
def halt_pandas_warnings():
    with pd.option_context("mode.chained_assignment", None):
        yield


# Anzahl oder Prozente der Nullen in einer Spalte
def nullen(df, xcol): return df[df[xcol].isnull()].shape[0]


def prozent_nullen(df, xcol): return df[df[xcol].isnull()].shape[0] / df.shape[0]


def nullen_in_df(xdf): return pd.DataFrame([(xcol, nullen(xdf, xcol)) for xcol in xdf.columns])


def prozent_nullen_in_df(xdf): return pd.DataFrame([(xcol, prozent_nullen(xdf, xcol)) for xcol in xdf.columns])


# Ergibt die Durchschnittswerte einer kontinuierlicher Variable pro Wert aus einer Kategorialer Variable
def durchschnittswerte_pro_kateg(df, xkateg, xwert): return df[[xkateg, xwert]]\
                                                                .groupby(xkateg).mean().sort_values(by=xwert)


def kateg_werte_liste(xdf_input, xcol, sep=None):
    """ Ergibt die Liste der Werte in kategorialer Variable
        Bleibt der sep None, dann sind die Werte nichts anderes als ein set(xL)
    """
    xliste = list(map(lambda x: str(x), xdf_input[xcol].values.tolist()))
    xliste = list(filter(lambda x: x is not np.nan, xliste))
    if sep is not None:
        xliste = list(map(lambda x: list(set(x.split(sep))), xliste))
        xliste = list(reduce(lambda a, b: a + b, xliste))
    xliste = [str(t).strip() for t in xliste]
    return xliste


def frequenz_werte(xdf_input, xcol="CousinEducation", prozente=False, sep=None):
    """ Anzahl der Datensätze mit xcol == {Wert} ODER Wert in xcol.split(sep)
    """
    df = xdf_input.copy()
    xl = kateg_werte_liste(df, xcol, sep=sep)
    xs = pd.DataFrame({xcol: xl})
    xs['id'] = xs.index.values.tolist()
    xres = xs.groupby(xcol, as_index=True).count()
    xres = xres.sort_values(by="id", ascending=False)
    if prozente:
        xres['id'] = [t / xdf_input.shape[0] for t in xres['id'].values.tolist()]
    return xres


def num_cat(xdf_input):
    """ Spalte den Datenbestand vertikal in kategoriale und numerische Teile
        Beispiel:
            xdf_num, xdf_cat = dbeschreiben.num_cat(dfres)
    """
    xdf_cat = xdf_input.select_dtypes(include=['object']).copy()
    xdf_num = xdf_input[list(filter(lambda xc: xc not in xdf_cat.columns, xdf_input.columns))]
    print("\nKategoriale Felder: {}".format(xdf_cat.columns))
    print("\nNumerische Felder: {}".format(xdf_num.columns))
    return xdf_num, xdf_cat


def zeige_kateg_features(xdata, target_col, id_col):
    # Zeige die Situation der kategorialen Variablen, welche Werte steckt drin?
    cols_filter = list(filter(lambda x: x is not None, [target_col, id_col]))
    num_cols = list(filter(lambda x: x not in cols_filter, xdata.describe().columns))
    cat_cols = list(filter(lambda x: x not in num_cols and x not in cols_filter, xdata.columns))
    for xcol in cat_cols:
        print("\n==========================")
        print("COL: {}".format(xcol))
        xr = frequenz_werte(xdata, xcol=xcol, prozente=False, sep=";")
        print(xr.index.values.tolist())


def zeige_korrelationen(xdf_input, target_col, threshold_corr = 0.01):
    # Korrelationen stärker als 0.02 (oder niedriger als -0.02)
    xtemp_corr = xdf_input.corr()[target_col]
    xcorr2 = xtemp_corr[list(map(lambda x: np.abs(x) > threshold_corr, xtemp_corr.values))]
    xcorr2df = pd.DataFrame({
        'feature': xcorr2.index.values,
        'IncomeCorr': xcorr2.values
    })
    # dfcorr = pd.merge(xcorr1df, xcorr2df, left_index=False, right_index=False)
    dfcorr = xcorr2df
    dfcorr = dfcorr.sort_values(by=["IncomeCorr"], ascending=False)
    return dfcorr