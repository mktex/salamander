import numpy as np

def melting(xdf_input):
    df_rural, _ = get_dataframes_i()
    df_rural.features_ausfiltern("Country Name", "Indicator Name")
    df1 = pd.melt(df_rural.data, id_vars=['Country Code', 'Indicator Code'],
                  value_vars=df_rural.data.columns[2:])


def pivot(self):
    df_rural, df_electricity = get_dataframes_i()
    df_res = CSV(DATA_DIR + "output.csv")
    df_res.data = df_rural + df_electricity
    xres = df_res.data.pivot(index='Country Name', columns='Indicator Name', values='2016')
    xres['Country Name'] = xres.index.values
    xres = xres.reset_index(drop=True)
    xres.columns.name = None


def reduktion_multikolinearitaet(xdfcat_dummy):
    """
        Reduziert DataFrame mit Dummy Variablen entsprechend Multikolinearität
        xdfcat_dummy kommt zB aus einem kateg2dummy Aufruf:
            xdfcat_dummy = kateg.kateg2dummy(dfcat, sep=None)
    :param xdfcat_dummy:
    :return:
    """
    xcorrdf = xdfcat_dummy.corr()
    remove_cols = []
    for xcol in xcorrdf.columns:
        if xcol in remove_cols:
            continue
        list_high_corrs = list(map(lambda x: np.abs(x) > 0.7, xcorrdf[xcol].values))
        _cols = xcorrdf[list_high_corrs].index.values.tolist()
        _cols = list(filter(lambda x: x != xcol, _cols))
        if len(_cols) != 0:
            print("\n{}:".format(xcol), _cols)
            remove_cols.extend(_cols)

    print("\Features mit hoher Korrelation zu anderen Features (non-target) werden entfernt:")
    print(remove_cols)
    xdfcat_dummy = xdfcat_dummy[list(filter(lambda x: x not in remove_cols, xdfcat_dummy.columns))]
    return xdfcat_dummy


def split_dataframe_num_cat(xdata, target_col, id_col):
    """
        Gegeben DataFrame xdata und target_col und id_col, ergibt
    :return:
    """
    # separat numerische und kategoriale Variablen
    cols_filter = list(filter(lambda x: x is not None, [target_col, id_col]))
    num_cols = list(filter(lambda x: x not in cols_filter, xdata.describe().columns))
    cat_cols = list(filter(lambda x: x not in num_cols and x not in cols_filter, xdata.columns))

    print("[x] Numerisch:")
    print(num_cols, "\n")

    print("[x] Kategorial:")
    print(cat_cols, "\n")
    print(target_col)

    dfnum = xdata[num_cols + [target_col]]
    dfcat = xdata[cat_cols]

    return dfnum, dfcat, num_cols, cat_cols