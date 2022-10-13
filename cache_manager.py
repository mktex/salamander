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


def update_object_name(old_value, new_value):
    """
        Changes name of the object in both cache objects: objects_articles_dict and objects_df

        Example:
            cmng.update_object_name(old_value=b'[SMN83] RCW  16 1', new_value=b'[SMN83] RCW 16 1')
    """
    from copy import deepcopy
    _papers_total, _objects_articles_dict, _objects_df = load_work()
    record = _objects_df[_objects_df.OBJID == old_value]
    if record.shape[0] != 0:
        idx = record.index.values[0]
        _objects_df.at[idx, 'OBJID'] = new_value
        _objects_articles_dict[new_value] = deepcopy(_objects_articles_dict[old_value])
        del _objects_articles_dict[old_value]
        print("[x] OBJID |{}| -> |{}|".format(old_value, new_value))
        save_work(_papers_total, _objects_articles_dict, _objects_df)
    else:
        print("[x] OBJID |{}| not found in cache".format(old_value))


def transfer_papers(from_objid, to_objid):
    """
        Transfers paper from one object to the other; this way from_objid can be deleted, without loss of information
    """
    _papers_total, _objects_articles_dict, _objects_df = load_work()
    from_record = _objects_df[_objects_df.OBJID == from_objid]
    to_record = _objects_df[_objects_df.OBJID == to_objid]
    if from_record.shape[0] != 0 and to_record.shape[0] != 0:
        transfer_papers = _objects_articles_dict[from_objid]
        result_list = list(set(_objects_articles_dict[to_objid] + transfer_papers))
        result_list.sort(reverse=True)
        _objects_articles_dict[to_objid] = result_list
        print("[x] Transfering papers from |{}| to |{}|:".format(from_objid, to_objid))
        print(transfer_papers)
        save_work(_papers_total, _objects_articles_dict, _objects_df)
    else:
        print("[x] One of two objects not in cache ({}, {})".format(from_objid, to_objid))


def remove(objid):
    """
        Removes objectid from cache
    """
    _papers_total, _objects_articles_dict, _objects_df = load_work()
    record = _objects_df[_objects_df.OBJID == objid]
    if record.shape[0] != 0:
        _objects_df = _objects_df[_objects_df.OBJID != objid].copy()
        del _objects_articles_dict[objid]
        save_work(_papers_total, _objects_articles_dict, _objects_df)
    else:
        print("[x] OBJID |{}| not found in cache".format(objid))
