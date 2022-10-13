import time

from astropy.io import fits
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
import os
import pickle
import traceback

from astropy import units as u
from astropy.coordinates import SkyCoord

import ned_table
import portals_urls
import cache_manager as cmng
import settings

# Data Model: https://data.sdss.org/datamodel/files/BOSS_SPECTRO_REDUX/RUN2D/spectra/PLATE4/spec.html
# https://dr12.sdss.org/advancedSearch
fpath = settings.fpath

# if FITS files have been moved to archive, and some time later a renewal of the object data is desired,
# then set this to true, it will replace existing data by newly downloaded
ASK_BEFORE_IGNORE = True

def load_data(spec_filename="spec-5733-56575-0948.fits"):
    """
        load just one spectra file
    :param spec_filename:
    :return:
    """
    hdu = fits.open(fpath + spec_filename)
    return hdu


def get_header(hdu):
    """
        process header of FITS file
    :param hdu:
    :return:
    """
    # Header data
    header = hdu[0].header
    cols_header = ["PLUG_DEC", "PLUG_RA", "MJD", "PLATEID", "FIBERID"]
    header_dict = {}
    for col in cols_header:
        header_dict[col] = header[col]
    c = SkyCoord(ra=140.0824 * u.degree, dec=49.478306 * u.degree)
    coords_str = c.to_string('hmsdms').replace("h", ":").replace("m", ":").replace("d", ":")
    header_dict["RA_DEC_HMSDMS"] = coords_str
    return header, header_dict


def get_coadd(hdu):
    """
    Coadded Spectrum from spPlate - flux values:
        # https://data.sdss.org/datamodel/files/BOSS_SPECTRO_REDUX/RUN2D/spectra/PLATE4/spec.html
        loglam	float32	log10(wavelength [Å])
        model	float32	pipeline best model fit used for classification and redshift
    :param hdu:
    :return:
    """
    data_coadd = pd.DataFrame(hdu[1].data.tolist(), columns=hdu[1].columns.names)
    return data_coadd


def get_spall(hdu):
    """
    https://data.sdss.org/datamodel/files/BOSS_SPECTRO_REDUX/RUN2D/PLATE4/RUN1D/spZbest.html
    SPECTROFLUX 5-element array of integrated flux in each of the 5 SDSS imaging
                filters (ugriz); the units are nanomaggies, which is 1 at 22.5
                magnitude; convert to magnitudes with
                  22.5 - 2.5 * LOG_10(SPECTROFLUX);
    SPECTROSYNFLUX: Same as SPECTROFLUX, but measured using the best-fit
                    synthetic eigen-spectrum rather than the actual data points
    """
    data_spall = pd.DataFrame(hdu[2].data.tolist(), columns=hdu[2].columns.names)
    data_spall = data_spall.iloc[0].to_dict()
    cols_spall = ["SPECTROSYNFLUX", "SN_MEDIAN_ALL", "CLASS", "SUBCLASS", "Z", "Z_ERR",
                  "BESTOBJID", "OBJID", "SPECOBJID", "TARGETOBJID"]
    spall_dict = {}
    for col in cols_spall:
        if col not in data_spall.keys():
            spall_dict[col] = None
        else:
            spall_dict[col] = data_spall[col]
    object_id = spall_dict["BESTOBJID"] if spall_dict['BESTOBJID'] is not None else spall_dict["OBJID"]
    if not ('bytes' in str(type(object_id)) or 'str' in str(type(object_id))):
        print("************************************ WARNING ************************************")
        print("\nWARNING: cannot get the object id ... ")
        time.sleep(1)
        object_id = None
    else:
        object_id = object_id.strip()
    spall_dict['BESTOBJID'] = object_id
    spall_dict['OBJID'] = object_id
    if not ('bytes' in str(type(object_id)) or 'str' in str(type(object_id))):
        spall_dict['SPECOBJID'] = str(spall_dict['SPECOBJID']).encode("utf-8")
    spall_dict['SPECTROSYNFLUX'] = [22.5 - 2.5 * np.log10(SPECTROFLUX) for SPECTROFLUX in spall_dict['SPECTROSYNFLUX']]
    return data_spall, spall_dict


def get_spzlines(hdu):
    """
        https://data.sdss.org/datamodel/files/BOSS_SPECTRO_REDUX/RUN2D/PLATE4/RUN1D/spZline.html
    :param hdu:
    """
    data_spzline = pd.DataFrame(hdu[3].data.tolist(), columns=hdu[3].columns.names)
    cols_spzline = ["PLATE", "MJD", "FIBERID", "LINENAME", "LINEWAVE", "LINEZ", "LINECHI2"]
    return data_spzline[cols_spzline]


"""
    URLs crafting specific to SDSS sdss.org website,
    required parameters are available in downloaded fits file
"""


def optical_spectra(header_dict):
    """
        access DR12 detail page for spectra
    :param header_dict:
    :return:
    """
    url = "https://dr12.sdss.org/spectrumDetail?mjd={}&fiber={}&plateid={}".format(
        header_dict['MJD'],
        header_dict['FIBERID'],
        header_dict['PLATEID']
    )
    print("\nOptical Spectra:")
    print(url)
    return url


def explore(spall_dict):
    """
        access skyserver for the object to explore
    :param spall_dict:
    :return:
    """
    url = "http://skyserver.sdss.org/dr16/en/tools/explore/Summary.aspx?id={}".format(
        (spall_dict['OBJID'] if 'BESTOBJID' not in spall_dict.keys() else spall_dict['BESTOBJID']).decode("utf-8")
    )
    print("\nExplore:")
    print(url)
    return url


def spectra(spall_dict):
    """
        access skyserver for details on this object
    :param spall_dict:
    :return:
    """
    url = "http://skyserver.sdss.org/dr16/en/get/SpecById.ashx?id={}".format(
        spall_dict['SPECOBJID']
    )
    print("\nSpectra:")
    print(url)
    return url


def show_urls(header_dict, spall_dict):
    """
        header_dict["PLUG_RA"], header_dict["PLUG_DEC"]
    :param header_dict:
    :param spall_dict:
    :return:
    """
    portals_urls.imaging(header_dict["PLUG_RA"], header_dict["PLUG_DEC"])
    optical_spectra(header_dict)
    portals_urls.cas(header_dict["PLUG_RA"], header_dict["PLUG_DEC"])
    explore(spall_dict)
    spectra(spall_dict)
    portals_urls.simbad(header_dict["PLUG_RA"], header_dict["PLUG_DEC"])
    portals_urls.cds(header_dict["PLUG_RA"], header_dict["PLUG_DEC"])
    portals_urls.ned(header_dict["PLUG_RA"], header_dict["PLUG_DEC"])


def show_data(header_dict, spall_dict, data_spzline, data_coadd, doplot=True):
    """
        prints out python objects for one processed object
    :param header_dict: header
    :param spall_dict: spectra
    :param data_spzline: data for the absorbtion/emission lines
    :param data_coadd: coadded data
    :param doplot: performs a plot, same data as shown on explore detail page
    :return:
    """
    pprint(header_dict)
    pprint(spall_dict)
    print(data_spzline)
    show_urls(header_dict, spall_dict)
    if doplot:
        _ = data_coadd.plot.scatter(x="loglam", y="model", s=1, alpha=0.3)
        plt.show()


def get_all_specfile_names():
    """ gather data from local data directory """
    fnames = []
    for subdir, dirs, files in os.walk("./data/"):
        if "archive" not in subdir:
            for fname in files:
                if '.fits' in fname:
                    fnames.append(fname.strip())
    return fnames


def get_spectral_line_record(data_spzline):
    temp = data_spzline[["LINENAME", "LINECHI2"]]
    temp.index = temp["LINENAME"]
    res = temp[["LINECHI2"]].transpose()
    res.columns = res.columns.tolist()
    res = res.reset_index(drop=True)
    return res


def string_strip(value):
    """
        strips string value from any whitespaces, makes sure it stays in byte format
    """
    if not ('bytes' in str(type(value)) or 'str' in str(type(value))):
        value = str(value).strip().encode("utf-8")
    else:
        value = value.strip()
    return value


def get_data_record(header_dict, spall_dict, data_spzline, set_object_id=None):
    """
        gets a dataframe record form for this object
    :param header_dict:
    :param spall_dict:
    :param data_spzline:
    :param set_object_id:
    :return:
    """
    object_id = set_object_id if set_object_id is not None else string_strip(spall_dict['BESTOBJID'])
    header_part = [header_dict['PLATEID'], header_dict['MJD'], header_dict['FIBERID'],
                   header_dict['PLUG_RA'], header_dict['PLUG_DEC']]
    spall_part = [string_strip(spall_dict["CLASS"]), string_strip(spall_dict['SUBCLASS']),
                  object_id,
                  string_strip(spall_dict["SPECOBJID"]),
                  spall_dict["SN_MEDIAN_ALL"], spall_dict["Z"]] \
                 + spall_dict["SPECTROSYNFLUX"]
    spzline_part = get_spectral_line_record(data_spzline)
    object_record = [
        header_part
        + spall_part
        + [0 if t <= 1.0 else 1.0 for t in spzline_part.iloc[0].values.tolist()]
    ]
    obj_record_columns = [
                             "PLATEID", "MJD", "FIBERID", "PLUG_RA", "PLUG_DEC",
                             "CLASS", "SUBCLASS", "OBJID", "SPECOBJID", "SN_MEDIAN_ALL", "Z",
                             "SPEC_U", "SPEC_G", "SPEC_R", "SPEC_I", "SPEC_Z"
                         ] + [t.decode("utf-8").strip() for t in spzline_part.columns.tolist()]
    record_df = pd.DataFrame(object_record)
    record_df.columns = obj_record_columns
    print(record_df.iloc[0])
    return record_df


def append_object(obj_df):
    """
        Appends a new object to objects_df dataframe
        Removes duplicates, if there are any
    :param obj_df:
    :return:
    """
    global objects_df
    if objects_df is None:
        objects_df = obj_df
    else:
        objects_df = pd.concat([objects_df, obj_df])
        objects_df = objects_df.reset_index(drop=True)
        objects_df = cmng.keep_last_object_record(objects_df)

def get_object_id_value(spall_dict):
    """
        get the best value for OBJID
    :param spall_dict:
    :return:
    """
    if "BESTOBJID" in spall_dict.keys():
        return spall_dict["BESTOBJID"]
    else:
        return spall_dict["OBJID"]


def do_main_loop(all_specfiles):
    """
        Performs the main loop for processing downloaded spectroscopic measurements
        - script stays civilized, slows down crawling, so as not to overload the servers with requests
        - can fail while trying to connect to server
    :param all_specfiles:
    :return:
    """
    global papers_total, objects_articles_dict, objects_df, known_refcodes

    count_files = 0
    for specfile_name in all_specfiles:
        ok_save = False
        try:
            hdu = load_data(spec_filename=specfile_name)
            header, header_dict = get_header(hdu)
            data_spall, spall_dict = get_spall(hdu)
            object_id = get_object_id_value(spall_dict)
            object_id = string_strip(object_id)

            if object_id in objects_df['OBJID'].values and not ASK_BEFORE_IGNORE:
                continue

            data_coadd = get_coadd(hdu)
            data_spzline = get_spzlines(hdu)
            show_data(header_dict, spall_dict, data_spzline, data_coadd, doplot=False)

            # create a pandas dataframe object and append it to objects_df collection
            obj_id = object_id
            obj_df = get_data_record(header_dict, spall_dict, data_spzline, set_object_id=object_id)
            append_object(obj_df)

            if obj_id not in objects_articles_dict.keys():
                print("{} File out of {} so far; pausing for 1s .. ".format(all_specfiles.index(specfile_name),
                                                                            len(all_specfiles)))
                print(
                    "\n************************************************************************************************************")
                print("FITS File:", specfile_name)
                objects_articles_dict[obj_id] = []
                papers, refcodes = ned_table.get_research_papers(
                    portals_urls.ned(header_dict["PLUG_RA"], header_dict["PLUG_DEC"]), known_refcodes
                )
                known_refcodes.extend(refcodes)
                known_refcodes = list(set(known_refcodes))
                if len(papers) != 0:
                    papers_total.extend(papers)
                objects_articles_dict[obj_id] = refcodes
                count_files += 1
                ok_save = True
            else:
                print(" .. already in cache .. ")
        except:
            traceback.print_exc()
            # wait for an enter, maybe got banned? if so should increase delay time
            # at midnight servers can be down for maintenance
            # input("... enter to continue ... ")

        if ok_save:
            cmng.save_work(papers_total, objects_articles_dict, objects_df)
            time.sleep(1.0)

    # when done move current FITS files to archive
    if count_files != 0:
        print("[x] Successfully processed {} FITS files, "
              "ready to move them to ./data/archive folder:".format(count_files))
        input("Enter to continue ... ")
        os.system("mv ./data/*.fits ./data/archive")


def show_datarecords():
    """
        shows content of objects already downloaded
    """
    papers_df = pd.DataFrame(papers_total)
    papers_df.columns = ['refcode', "title", "description", "link", "keywords"]

    print("\n[x] papers_df: ")
    print(papers_df)

    print("\n[x] Dict object for articles with maping to objects:")
    pprint(objects_articles_dict)

    print("\n[x] DataFrame objects_df:")
    print(objects_df)


"""
    Start job by looking in ./data and load cached pickle files, if previous done work exists.
"""
# papers_total = []; objects_df = None; objects_articles_dict = {}
papers_total, objects_articles_dict, objects_df = cmng.load_work()
if len(papers_total) != 0:
    known_refcodes = pd.DataFrame(papers_total)[0].values.tolist()
else:
    known_refcodes = []

# Get current list of spec-files from ./data
all_specfiles = get_all_specfile_names()

if __name__ == "__main__":
    do_main_loop(all_specfiles)
