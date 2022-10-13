
"""
URLs crafting:
Specific to SDSS, since they need special parameters found in fits files:
- Optical Spectra: https://dr12.sdss.org/spectrumDetail?mjd=52247&fiber=45&plateid=766
- Explore: http://skyserver.sdss.org/dr16/en/tools/explore/Summary.aspx?id=1237654652032516161
- Spectra: http://skyserver.sdss.org/dr16/en/get/SpecById.ashx?id=862451735845693440

Not specific to SDSS, can be used if coordinates are available:
- Imaging: https://dr12.sdss.org/fields/raDec?ra=140.49233988467287&dec=50.294630489476404
- CAS: http://skyserver.sdss.org/dr16/en/tools/chart/navi.aspx?ra=139.19916&dec=49.780379
- SIMBAD: http://simbad.u-strasbg.fr/simbad/sim-coo?Coord=139.199168322%2C+49.780391239&CooFrame=FK5
            &CooEpoch=2000&CooEqui=2000&CooDefinedFrames=none&Radius=2&Radius.unit=arcmin&submit=submit+query&CoordList=
- CDS Portal: http://cdsportal.u-strasbg.fr/?target=139.199168322%2C%2049.780391239
- NED: http://ned.ipac.caltech.edu/cgi-bin/nph-objsearch?search_type=Near+Position+Search&in_csys=Equatorial
        &in_equinox=J2000.0&obj_sort=Distance+to+search+center&lon=139.1991683d&lat=49.7803912d&radius=1.0
- ADS Portal: https://ui.adsabs.harvard.edu/abs/2011MNRAS.410..166L/abstract
    - Description
    - date
    - Link zu arXiv
    - Keywords
"""

RA = None
DEC = None

def set_radec(ra, dec):
    global RA, DEC
    RA = ra
    DEC = dec

def cds(ra, dec):
    url = "http://cdsportal.u-strasbg.fr/?target={}%2C%20{}".format(
        ra,
        dec
    )
    print("\nCDS Portal:")
    print(url)
    return url

def simbad(ra, dec):
    # "Coord=139.199168322%2C+49.780391239&" \
    url = "http://simbad.u-strasbg.fr/simbad/sim-coo?" \
          "Coord={}%2C+{}&" \
          "CooFrame=FK5&CooEpoch=2000&CooEqui=2000&CooDefinedFrames=none&" \
          "Radius=2&" \
          "Radius.unit=arcmin&" \
          "submit=submit+query&CoordList=".format(
        ra,
        dec
    )
    print("\nSIMBAD:")
    print(url)
    return url

def simbad_ident(object_id):
    import urllib
    dataurl = "httpRadio=Get&output.format=ASCII_SPREADSHEET&output.file=on&output.max=10&" \
              "list.idsel=on&list.idopt=FIRST&list.idcat=&list.otypesel=on&otypedisp=3&list.coo1=on&" \
              "frame1=ICRS&epoch1=J2000&coodisp1=s2&frame2=FK5&epoch2=J2000&equi2=2000&coodisp2=s2&" \
              "frame3=FK4&epoch3=B1950&equi3=1950&coodisp3=s2&frame4=Gal&epoch4=J2000&equi4=2000&coodisp4=d2&" \
              "obj.pmsel=on&obj.plxsel=on&obj.rvsel=on&rvRedshift=on&rvRadvel=on&rvCZ=on&" \
              "obj.fluxsel=on&list.fluxsel=on&U=on&B=on&V=on&R=on&G=on&I=on&J=on&H=on&K=on&u=on&g=on&r=on&i=on&z=on&" \
              "obj.spsel=on&list.spsel=on&obj.mtsel=on&obj.sizesel=on&bibyear1=1990&bibyear2=%24currentYear&" \
              "bibjnls=&bibdisplay=bibnum&bibcom=off&notedisplay=S&obj.messel=on&list.mescat=&mesdisplay=N&save=SAVE"
    url = "http://simbad.u-strasbg.fr/simbad/sim-id?Ident={}&" \
          "NbIdent=1&Radius=2&Radius.unit=arcmin&submit=submit+id&{}".format(
        urllib.parse.quote(object_id),
        dataurl
    )
    print("\nSIMBAD (identifier search for 1 object):")
    print(url)
    return url


def ned(ra, dec):
    url = "http://ned.ipac.caltech.edu/cgi-bin/nph-objsearch?" \
          "search_type=Near+Position+Search&in_csys=Equatorial&in_equinox=J2000.0&" \
          "obj_sort=Distance+to+search+center&" \
          "lon={}d&lat={}d" \
          "&radius=2.0".format(
        ra,
        dec
    )
    print("\nNED:")
    print(url)
    return url

def imaging(ra, dec):
    url = "https://dr12.sdss.org/fields/raDec?ra={}&dec={}".format(
        ra, dec
    )
    print("\nImaging:")
    print(url)
    return url


def cas(ra, dec):
    url = "http://skyserver.sdss.org/dr16/en/tools/chart/navi.aspx?ra={}&dec={}".format(
        ra,
        dec
    )
    print("\nCAS:")
    print(url)
    return url


def show_urls():
    global RA, DEC

    url_img = imaging(RA, DEC)
    url_cas = cas(RA, DEC)
    url_simbad = simbad(RA, DEC)
    url_cds = cds(RA, DEC)
    url_ned = ned(RA, DEC)

    return url_img, url_cas, url_simbad, url_cds, url_ned


# TODO: Extragalactic Distances Datenbank? http://edd.ifa.hawaii.edu/dfirst.php?