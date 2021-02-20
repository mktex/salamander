
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

