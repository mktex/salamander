import time
import numpy as np
import pandas as pd
from importlib import reload

import cache_manager as cmng
import read_simbad
import astrocalc
import data_clean_objects


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


def get_records_with_tags(tag):
    """
    :param tag: some tag to search for, e.g. 'velocity'
    :return: the papers_df DataFrame object
    """
    papers_total, objects_articles_dict, objects_df = cmng.load_work()
    papers_df = pd.DataFrame(papers_total, columns=['refcode', "title", "description", "link", "keywords"])
    is_in_list = lambda keyword, keywordlist: len(list(filter(lambda key:
                                                              keyword.lower() in key.lower(), keywordlist))) != 0
    records = papers_df[list(map(lambda keywordlist: is_in_list(tag, clean_tags(keywordlist)), papers_df["keywords"].values))]
    if records.shape[0] > 10:
        print(records.sample(10))
    else:
        print(records)
    if records.shape[0] != 0:
        print(records.iloc[0].keywords)
    return records, papers_df


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


def clean_tags(keywordlist=None, debug=False):
    """
        Given list of keywords, clean it to be able to have corresponding features
        return cleaned list of keywords

        debug example:
            import data_cleaning
            keywords = data_cleaning.clean_tags(keywordlist=None, debug=True)
            # then back and forth in this method hardcoded values; it only needs to lead to relatively lower dimensions

    :param keywordlist:
    :param debug:
    :return:
    """
    if keywordlist is None:
        from functools import reduce
        papers_total, objects_articles_dict, objects_df = cmng.load_work()
        papers_df = pd.DataFrame(papers_total, columns=['refcode', "title", "description", "link", "keywords"])
        keywordlist = list(set(reduce(lambda a, b: a + b, papers_df["keywords"].tolist())))

    reload(data_clean_objects)

    replacements_dict = data_clean_objects.replacements_dict
    replacements_by_startswith_dict = data_clean_objects.replacements_by_startswith_dict
    ignore_tag_list = data_clean_objects.ignore_tag_list

    def replpunctuation(xstr):
        import string
        for c in string.punctuation:
            xstr = xstr.replace(c, " ")
        return " ".join(list(filter(lambda y: y != "", xstr.split(" "))))

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

    # remove numbers
    xfuncstr = lambda xstr: " ".join(
        list(filter(lambda y: y != "",
            list(map(lambda x: x.strip(), xstr.split(" ")))
        ))
    )
    for c in "0123456789":
        keywords = list(map(lambda x: xfuncstr(x.replace(c, "")), keywords))

    # remove anything that is less than 3 chars
    keywords = list(set(filter(lambda x: len(x) > 3, keywords)))

    # replacements by dictionary
    for key, val in replacements_dict.items():
        keywords = list(map(lambda x: x.replace(key, val), keywords))

    for key, val in replacements_by_startswith_dict.items():
        keywords = list(map(lambda x: val if key == x[:len(key)] else x, keywords))

    keywords = list(set(filter(lambda x: x not in ignore_tag_list, keywords)))

    keywords = list(set(keywords))
    keywords.sort()

    if debug:
        from pprint import pprint
        pprint(keywords)
        print(f"[x] current lenght of keyword list: {len(keywords)}")
        # print("\t Authors :", author_names)
        # print("\t Keywords:", keywords)
        print("")

    return keywords


def clean_object_names():
    """
        Update names in objects_articles_dict and objects_df then save back data to cache
        - no multiple whitespaces
        - remove objects with no name, ie b''
    :return:
    """
    import cache_manager as cmng
    import pandas as pd
    from pprint import pprint
    papers_total, objects_articles_dict, objects_df = cmng.load_work()
    t = list(objects_articles_dict.keys())
    t = list(filter(lambda x: '123' != x.decode('utf-8')[:3], t))
    t.sort()
    print("Cleanning object names..")
    pprint(t)
    for objid in t:
        objid_new = " ".join(
            list(filter(lambda y: y != "",
                list(map(lambda x: x.strip(),
                         objid.decode("utf-8").split(" ")))
            ))
        ).encode("utf-8")
        if objid != objid_new:
            cmng.update_object_name(objid, objid_new)


def check_nan_values_spectra_features(objects_df):
    spec_features = ["SPEC_U", "SPEC_G", "SPEC_R", "SPEC_I", "SPEC_Z"]
    objects_df_nans = objects_df[["OBJID", "CLASS", "SUBCLASS", "PLUG_RA", "PLUG_DEC"] + spec_features].copy()
    nan_values = []
    for ik in range(objects_df_nans.shape[0]):
        record = objects_df_nans.iloc[ik]
        n = np.sum([t is None or np.isnan(t) for t in record[spec_features].values.tolist()]) / len(spec_features)
        nan_values.append(n)
    objects_df_nans["pnan"] = nan_values
    return objects_df_nans


def get_distances_to_objects(objects_df, objid, objects_removal):
    """
        Returns a dataframe with distances to objid ignoring those objects that have been planned for removal
    """
    from scipy.spatial.distance import cdist
    func_distance = lambda xs1, xs2: cdist(np.array(xs1).reshape(-1, 1), np.array(xs2.reshape(-1, 1)), 'euclidean')
    obj_idx = objects_df[objects_df.OBJID == objid].index.values[0]
    obj_pos = objects_df[["PLUG_RA", "PLUG_DEC"]].iloc[obj_idx].values.tolist()
    G = objects_df[["PLUG_RA", "PLUG_DEC"]].values
    distances = []
    for j in range(0, G.shape[0]):
        if j != obj_idx:
            xdist = func_distance(obj_pos, G[j]).flatten()[0]
            distances.append([j, xdist])
    xdf = pd.DataFrame(distances, columns=["id", "d"])
    xdf["will_be_removed"] = [objects_df['OBJID'].iloc[t] in objects_removal for t in xdf['id'].values.tolist()]
    xdf = xdf[xdf["will_be_removed"] == False]
    xdf = xdf.sort_values(by="d", ascending=True)
    xdf = xdf.reset_index(drop=True)
    return xdf, obj_idx, obj_pos


def handle_objects_nan_values():
    """
        Objects that only have a name could be eliminated and related papers given to closest object
    :return:
    """
    papers_total, objects_articles_dict, objects_df = cmng.load_work()

    objects_df_nans = check_nan_values_spectra_features(objects_df)
    objects_removal = objects_df_nans[objects_df_nans.pnan == 1.0].copy()['OBJID'].values.tolist()

    removal_list = []
    for objid in objects_removal:
        xdf, obj_idx, obj_pos = get_distances_to_objects(objects_df, objid, objects_removal)
        print("========================================================================================")
        print("OBJID: {}".format(objid))
        print("Neighbouring Object:")
        print(objects_df.loc[[obj_idx, xdf["id"].iloc[0]]].transpose())
        objid_receiving_papers = objects_df.loc[xdf["id"].iloc[0]].OBJID
        removal_list.append([objid, objid_receiving_papers])

    removal_df = pd.DataFrame(removal_list, columns=["objid_old", "objid_new"])

    # not allowing to loose Ms, NGCs, HDs
    fchoose = lambda s1, s2: s1 if (b'M' == s1[:1] or b'HD' == s1[:2] or b'NGC' == s1[:3]) else s2
    removal_df['rename_to'] = [fchoose(*t) for t in removal_df[["objid_old", "objid_new"]].values]

    for ik in range(removal_df.shape[0]):
        print("=======================================================================================")
        from_objid = removal_df["objid_old"].iloc[ik]
        to_objid = removal_df["objid_new"].iloc[ik]
        rename_to = removal_df["rename_to"].iloc[ik]
        cmng.transfer_papers(from_objid, to_objid)
        cmng.remove(from_objid)
        if rename_to != to_objid:
            cmng.update_object_name(to_objid, rename_to)

    print("Finished cleaning data of Objects with too many null-values.")
    print("You can check results with: \n"
          "\tpapers_total, objects_articles_dict, objects_df = cmng.load_work()")
