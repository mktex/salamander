import pickle
import os

def save_work(papers_total, objects_articles_dict, objects_df):
    """
        Used to save to local cache processed files
    :param papers_total:
    :param objects_articles_dict:
    :param objects_df:
    :return:
    """
    with open("./data/papers_total.pckl", "wb") as f:
        pickle.dump(papers_total, f)
    with open("./data/objects_articles_dict.pckl", "wb") as f:
        pickle.dump(objects_articles_dict, f)
    with open("./data/objects_df.pckl", "wb") as f:
        pickle.dump(objects_df, f)
    print("[x] finished saving work to cache files:")
    print("\t./data/objects_articles_dict.pckl")
    print("\t./data/objects_df.pckl")
    print("\t./data/papers_total.pckl")


def load_work():
    """
        Loads the three files from the local cache to continue processing
    :return: three objects papers_total, objects_articles_dict, objects_df
    """
    def do_load_pickle_file_if_exists(fpath, default_val=None):
        if os.path.isfile(fpath):
            with open(fpath, "rb") as f:
                data = pickle.load(f)
        else:
            data = default_val
        return data

    _papers_total = do_load_pickle_file_if_exists("./data/papers_total.pckl", [])
    _objects_articles_dict = do_load_pickle_file_if_exists("./data/objects_articles_dict.pckl", {})
    _objects_df = do_load_pickle_file_if_exists("./data/objects_df.pckl", None)
    print("[x] Previous work loaded")
    return _papers_total, _objects_articles_dict, _objects_df

def keep_last_object_record(_objects_df):
    # don't drop duplicates, just keep the last version of the object data:
    objdf = _objects_df.copy()
    objdf = objdf.reset_index(drop=True)
    ik = objdf.shape[0] - 1
    objidlist = []
    indexlist = []
    while ik >= 0:
        if objdf.at[ik, 'OBJID'] not in objidlist:
            objidlist.append(objdf.at[ik, 'OBJID'])
            indexlist.append(ik)
        else:
            # print(objects_df.at[ik, 'OBJID'])
            pass
        ik -= 1
    objdf = objdf.loc[indexlist]
    objdf = objdf.reset_index(drop=True)
    return objdf