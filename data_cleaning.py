import time
import numpy as np
import pandas as pd

import cache_manager as cmng
import read_simbad
import astrocalc


def show_two_dicts(d1, d2):
    """
        given two dictionaries with same keys, show using pandas dataframe the changes between the two
    :param d1:
    :param d2:
    :return:
    """
    dcombined = []
    for key in d1.keys():
        dcombined.append([key, d1[key], d2[key]])
    dcombined = pd.DataFrame(dcombined, columns=["feature", "old", "new"])
    print(dcombined)


def fix_missing_simbad_spectral_data():
    """
        For those objects downloaded from simbad, mising data for spectra values as well as redshift
        can be obtained from the detail page of the object.
    :return:
    """

    papers_total, objects_articles_dict, objects_df = cmng.load_work()
    objects_df_new = objects_df.copy()

    _objects_df = objects_df[list(map(lambda x: not np.isnan(x), objects_df['Z'].values))]
    objects_df_missing_data = objects_df[list(map(lambda x: np.isnan(x), objects_df['Z'].values))]
    list_objects = objects_df_missing_data.OBJID.values.tolist()
    fields_update = ["Z", "SPEC_U", "SPEC_G", "SPEC_R", "SPEC_I", "SPEC_Z"]

    object_detail_dictionaries = {}
    for object_id in list_objects:
        object_detail_dictionaries[object_id] = read_simbad.get_simbad_detail_identifier(object_id)
        time.sleep(1)

    # apply updates to dataframe
    for object_id in list_objects:
        print("========================================================================================")
        record = objects_df_missing_data[objects_df_missing_data.OBJID == object_id]
        index = record.index.values[-1]
        object_dict = record[fields_update].iloc[-1].to_dict()
        detail_data = object_detail_dictionaries[object_id]
        object_dict_updated = astrocalc.fillup_object_dict_by_detail_page(object_dict, detail_data)
        for field in fields_update:
            objects_df_new.at[index, field] = object_dict_updated[field]
        print("[x] Object: {}".format(object_id))
        show_two_dicts(objects_df[objects_df.OBJID == object_id][fields_update].iloc[0],
                       objects_df_new[objects_df_new.OBJID == object_id][fields_update].iloc[0])

    cmng.save_work(papers_total, objects_articles_dict, objects_df_new)


def clean_tags(keywordlist, debug=False):
    """
        Given list of keywords, clean it to be able to have corresponding features
        return cleaned list of keywords
    :param keywordlist:
    :param debug:
    :return:
    """
    replacements_dict = {
        "surveys: galaxies: fundamental parameters: galaxies: statistics": "surveys",
        "submillimetre": "submillimeter",
        "stars: magnetic fields": "stars: magnetic field",
        "x rays": "x ray",
        "astronomical data bases miscellaneous": "astronomical databases miscellaneous",
        "catalogues": "catalog",
        "catalogs": "catalog",
        "hertzsprung russell and c m diagrams": "hertzsprung",
        "hertzsprung russell and colour magnitude diagrams": "hertzsprung",
        "white dwarfs": "white dwarf",
        'statistical': 'statistics',
        'statistics computation': 'statistics',
        'statistics machine learning': 'statistics',
        'supernovae general': "supernovae"
    }
    def replpunctuation(xstr):
        import string
        for c in string.punctuation:
            xstr = xstr.replace(c, " ")
        return " ".join(list(filter(lambda y: y != "", xstr.split(" "))))
    author_names = list(filter(lambda x: "," in x, keywordlist))
    keywords = list(filter(lambda x: "," not in x, keywordlist))
    keywords = list(map(lambda x: x.strip(), keywords))
    keywords = list(map(lambda x: x.lower(), keywords))
    keywords = list(map(lambda x: x.lower(), keywords))
    keywords = list(map(lambda x: replpunctuation(x), keywords))
    keywords = list(filter(lambda x: not x.isdigit(), keywords))
    keywords = list(filter(lambda x: not 'ngc ' in x, keywords))
    keywords = list(filter(lambda x: not 'sn ' in x, keywords))
    keywords = list(filter(lambda x: not 'ugc ' in x, keywords))
    keywords = list(map(lambda x:
                           x.split(' individual ')[0].strip() if ' individual ' in x else x,
                keywords))
    keywords = list(filter(lambda x: not ('and' in x and 'more' in x), keywords)) # eg "and 119 more"
    keywords = list(map(lambda x:
                        " ".join(list(filter(lambda y: y != "", x.replace("-", " ").split(" ")))),
                keywords))
    for key, val in replacements_dict.items():
        keywords = list(map(lambda x: x.replace(key, val), keywords))
    final_removals = [
        'm 101 group',
        'm 51',
        'm 63',
        'm10 amp ngc 6254',
        'm22',
        'm42',
        'm54',
        'm80 amp ngc 6093',
        '85 10',
        '85a15',
        '85a30',
        '95 10 eg',
        '97 10 nf',
        '97 10 vm',
        '97 80 af',
        '97 80 di',
        'a2199',
        'sn2012cs',
        'sn2012p',
        'sn2013bb'
    ]
    keywords = list(filter(lambda x: not x in final_removals, keywords))
    if debug:
        print(keywordlist)
        print("\t Authors :", author_names)
        print("\t Keywords:", keywords)
        print("")
    return keywords
