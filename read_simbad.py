from bs4 import BeautifulSoup
import requests
import time
import traceback
import sys
from pprint import pprint

import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord

import portals_urls
import cache_manager as cmng
import ned_table
import astrocalc
import settings

simbad_top_k_objects = settings.simbad_top_k_objects
simbad_refcodes_per_object = settings.simbad_refcodes_per_object


def safe_float(xstring):
    try:
        res = float(xstring)
        return res
    except:
        pass
    return None


def fill_zeros(coords):
    """
        fill with zeros, if coord is missing seconds
    :param coords:
    :return:
    """
    if len(coords) == 2:
        if "." in coords[1]:
            coords = [coords[0], "00", coords[1]]
        else:
            coords = [coords[0], coords[1], "00"]
    return coords


def get_simbad_detail_identifier(object_id):
    """
        Access page
            http://simbad.u-strasbg.fr/simbad/sim-id?Ident=NGC+2293&NbIdent=1&Radius=2&Radius.unit=arcmin&submit=submit+id
        With selected cookie, corresponding options on page
            http://simbad.u-strasbg.fr/simbad/sim-fout
    :param simbad_url:
    :return:
    """
    simbad_ident_url = portals_urls.simbad_ident(object_id)
    with requests.Session() as s:
        time.sleep(1.0)
        _content = s.get(simbad_ident_url).text

    _content = [txt.strip() for txt in _content.split("\n")]
    print('\n'.join(_content))

    def fix_string(xstr):
        return " ".join(list(filter(lambda x: x != "", xstr.replace(":", " ").split(" "))))

    def safe_float(xstring):
        try:
            res = float(xstring)
            return res
        except:
            pass
        return None

    dict_object = {}
    for key in ["Redshift",
                "Flux U", "Flux B", "Flux V", "Flux G", "Flux R", "Flux I", "Flux Z", "Flux J", "Flux H", "Flux K",
                "Flux u", "Flux b", "Flux v", "Flux g", "Flux r", "Flux i", "Flux z"
                ]:
        line = list(filter(lambda xline: key in xline, _content))
        if len(line) != 0:
            line = line[0]
            line_fixed = fix_string(line)
            dict_object[key] = safe_float(line_fixed.replace(key, "").strip().split(" ")[0])
        else:
            dict_object[key] = None

    return dict_object


def read_simbad_table_object_info(rows):
    """
        example page result:
            http://simbad.u-strasbg.fr/simbad/sim-coo?Coord=88.792939+7.407064&CooFrame=FK5&CooEpoch=2000&CooEqui=2000&CooDefinedFrames=ICRS-J2000&Radius=2&Radius.unit=arcmin&submit=submit+query&CoordList=
    :param rows:
    :return:
    """
    list_object_data = []
    for row in rows[:10]:
        object_dict = {}
        column_data = row.find_all("td")
        # object_dict['url'] = column_data[1].find('a')['href']
        object_dict['OBJID'] = column_data[1].text.strip().encode("utf-8")
        object_dict['CLASS'] = column_data[3].text.strip().encode("utf-8")
        object_dict['SUBCLASS'] = column_data[11].text.strip().encode("utf-8")

        ra_str = fill_zeros(column_data[4].text.strip().split(" "))
        dec_str = fill_zeros(column_data[5].text.strip().split(" "))
        print("[x] ra:", ra_str)
        print("[x] dec:", dec_str)
        coord = SkyCoord("{}h{}m{}s".format(*ra_str),
                         "{}d{}m{}s".format(*dec_str), frame='icrs')
        object_dict['PLUG_RA'] = coord.ra.deg
        object_dict['PLUG_DEC'] = coord.dec.deg

        # reading U, B, V, R, I and setting as SPEC_U, SPEC_G, SPEC_R, SPEC_I, SPEC_Z
        # TODO: these are not the same, but can use as aproximation for the moment
        # https://en.wikipedia.org/wiki/Photometric_system
        # need to use a special method to actually normalize the values, such as
        # implemented in https://dawn.nbi.ku.dk/events/Master__Thesis_Suk_Joo_Ko.pdf
        U, B, V, R, I = [safe_float(column_data[k].text.strip()) for k in range(6, 11)]
        object_dict['SPEC_U'] = U
        object_dict['SPEC_B'] = B
        object_dict['SPEC_V'] = V
        object_dict['SPEC_G'] = None
        object_dict['SPEC_R'] = R
        object_dict['SPEC_I'] = I
        object_dict['SPEC_Z'] = None

        refcodes_url = column_data[12].find("a")['href']
        print("\n[x] Object data:")
        pprint(object_dict)
        list_object_data.append([object_dict, refcodes_url])
    return list_object_data


def read_simbad_table_biblio(rows):
    """
        extracts bibliographic information from each row
    :param rows:
    :return:
    """
    articles = []
    for row in rows:
        column_data = row.find_all("td")
        refcode = column_data[0].text.strip()
        score = column_data[1].text.strip()
        score = int(score) if score != "" else None
        relevance_code = column_data[2].text.strip()
        title = column_data[8].text.strip()
        article = [refcode, score, relevance_code, title]
        articles.append(article)
    art_obj_df = pd.DataFrame(articles, columns=["refcode", "score", "relevance_code", "title"])
    print("\n[x] Found {} articles for this object".format(art_obj_df.shape[0]))
    print(art_obj_df)
    art_obj_df = art_obj_df.sort_values(by="score", ascending=False)
    art_obj_df = art_obj_df.reset_index(drop=True)
    return art_obj_df


def read_simbad_webpage_table(simbad_url):
    """
        access simbad page for the table list with objects around some coordinates
        region is coded in simbad_url
    :param simbad_url:
    :return:
    """
    with requests.Session() as s:
        time.sleep(0.5)
        html_content = s.get(simbad_url).text
    soup = BeautifulSoup(html_content, "lxml")
    table = soup.find("tbody", attrs={"class": "datatable"})
    if table is not None:
        rows = table.find_all("tr")
        return rows, soup
    else:
        return None, soup


def do_main_loop(ra, dec):
    """
        Example:
            do_main_loop(ra=88.792939, dec=7.407064)
    :param ra: in degrees
    :param dec: in degrees
    :return:
    """
    global papers_total, objects_articles_dict, objects_df, known_refcodes
    global simbad_top_k_objects, simbad_refcodes_per_object
    # example: http://simbad.u-strasbg.fr/simbad/sim-coo?Coord=88.792939%2C+7.407064&CooFrame=FK5&CooEpoch=2000&CooEqui=2000&CooDefinedFrames=none&Radius=2.0&Radius.unit=arcmin&submit=submit+query&CoordList
    simbad_url = portals_urls.simbad(ra, dec)
    print("[x] Reading simbad table of object results..")
    rows, _ = read_simbad_webpage_table(simbad_url)
    list_object_data = read_simbad_table_object_info(rows=rows)
    print("[x] Considering first 10 objects close to the given coordinates")
    for object_dict, simbad_bibcode_url in list_object_data[:simbad_top_k_objects]:
        try:
            obj_id = object_dict['OBJID']
            if obj_id in objects_articles_dict.keys() and not settings.redownload_simbad:
                print(" .. already in cache .. ")
            else:
                # fix object_dict data by getting missing information from the detail page:
                detail_data = get_simbad_detail_identifier(obj_id)
                object_dict = astrocalc.fillup_object_dict_by_detail_page(object_dict, detail_data)
                # example: http://simbad.u-strasbg.fr/simbad/sim-id-refs?Ident=*%20alf%20Ori&Name=*%20alf%20Ori
                simbad_bibcode_url = "http:" + simbad_bibcode_url
                rows, soup = read_simbad_webpage_table(simbad_bibcode_url)
                ref_codes = None
                if rows is not None:  # simbad shows a table with results
                    art_obj_df = read_simbad_table_biblio(rows)
                    print("[x] Using first {} references close to the position".format(simbad_refcodes_per_object))
                    ref_codes = art_obj_df.refcode.iloc[:simbad_refcodes_per_object].values.tolist()
                else:  # there must have been one page abstract
                    view_ads_link = list(
                        filter(lambda t: t.text.strip() == 'View the reference in ADS', soup.find_all("a")))
                    if len(view_ads_link) != 0:
                        view_ads_link = view_ads_link[0]
                        ref_codes = [view_ads_link['href'].split("/")[-1]]
                if ref_codes is not None:
                    unknown_ref_codes = list(filter(lambda x: x not in known_refcodes, ref_codes))
                    papers = ned_table.get_ads_data_by_refcodes(ref_codes, known_refcodes)
                    # global objects update
                    known_refcodes.extend(ref_codes)
                    known_refcodes = list(set(known_refcodes))
                    papers_total.extend(papers)
                    objects_articles_dict[obj_id] = ref_codes
                    obj_df = object_dict.copy()
                    for key in obj_df.keys():
                        obj_df[key] = [obj_df[key]]
                    obj_df = pd.DataFrame(obj_df)
                    obj_df = obj_df[["OBJID", "CLASS", "SUBCLASS", "PLUG_RA", "PLUG_DEC",
                                     "SPEC_U", "SPEC_G", "SPEC_R", "SPEC_I", "SPEC_Z"]]
                    if objects_df is None:
                        objects_df = obj_df
                    else:
                        objects_df = pd.concat([objects_df, obj_df])
                        objects_df = cmng.keep_last_object_record(objects_df)
                    if len(unknown_ref_codes) != 0:
                        cmng.save_work(papers_total, objects_articles_dict, objects_df)
                        time.sleep(1)
        except:
            traceback.print_exc()
            input("... enter to continue ... ")


def show_datarecords():
    papers_df = pd.DataFrame(papers_total)
    papers_df.columns = ['refcode', "title", "description", "link", "keywords"]
    print(papers_df)
    pprint(objects_articles_dict)
    print(objects_df)


"""
    Start job by looking in ./data and load cached pickled files, if previous done work exists.
"""
papers_total, objects_articles_dict, objects_df = cmng.load_work()
if len(papers_total) != 0:
    known_refcodes = pd.DataFrame(papers_total)[0].values.tolist()
else:
    known_refcodes = []

"""
    Usage examples either copy/paste from CDS portal or give the hms-dms format:
         python read_simbad.py '05h46m45.800s' '+00d04m45.00s'
         python read_simbad.py  05 34 31.940 +22 00 52.20
"""
if __name__ == "__main__":
    ra = None
    dec = None
    if len(sys.argv) == 3:
        # ra = '05h55m10.30536s'; dec = '+07d24m25.4304s'
        ra = sys.argv[1]
        dec = sys.argv[2]
    elif len(sys.argv) == 7:
        print(sys.argv)
        ra = "{}h{}m{}s".format(*sys.argv[1:4])
        dec = "{}d{}m{}s".format(*sys.argv[4:])
    else:
        print("[x] Example usage: python read_simbad.py '05h55m10.30536s', '+07d24m25.4304s'")
        print("=> gets data for Betelgeuse by using Simbad website")
        print("[x] Alternative: python read_simbad.py 05 55 10.30536 +07 24 25.4304")

    if ra is not None and dec is not None:
        coord = SkyCoord(ra, dec, frame='icrs')
        ra, dec = coord.ra.deg, coord.dec.deg
        print(ra, dec)
        do_main_loop(ra, dec)
