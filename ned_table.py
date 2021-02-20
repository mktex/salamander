from bs4 import BeautifulSoup
import requests
import time
import traceback

url_base = "http://ned.ipac.caltech.edu"

def get_refcodes(soup):
    table = soup.find_all("table")  # attrs={"name": "objlist"}
    all_links = table[0].find_all('tr')[0].find_all("a", attrs={"target": "ned_dw"})
    all_links = list(filter(lambda x:
                            "/cgi-bin/datasearch?search_type=Ref_id" in str(x),
                            all_links
                            ))
    print("[x] Reference links found:", len(all_links))
    ref_codes = []
    for link in all_links:
        print("\nURL: {}".format(link))
        print(".. pause for 2s ..")
        time.sleep(1)
        url_check = url_base + link['href']
        print("[x] Requesting url:")
        print(url_check)
        with requests.Session() as s:
            time.sleep(0.5)
            detail_page = s.get(url_check).text
        soup_detail = BeautifulSoup(detail_page, "lxml")
        table_detail = soup_detail.find_all("table")  # attrs={"name": "objlist"}
        paper_links = table_detail[0].find_all('tr')[0].find_all("a", attrs={"target": "ned_dw"})
        paper_refcodes = [t.getText() for t in paper_links]
        print("=> Papers:", paper_refcodes)
        ref_codes.extend(
            paper_refcodes
        )
        ref_codes = list(set(ref_codes))
        print("Ref. codes updated: {}".format(ref_codes))
    print("\n[x] Articles found: {}\n".format(len(ref_codes)))
    return ref_codes

def get_abstract(refcode):
    try:
        print("=================================================================================")
        ads_url = "https://ui.adsabs.harvard.edu/abs/{}/abstract".format(refcode)
        print(ads_url)
        with requests.Session() as s:
            time.sleep(0.5)
            ads_page = s.get(ads_url).text
        ads_soup = BeautifulSoup(ads_page, "lxml")
        title_elem = ads_soup.find_all("h2", attrs={"class": "s-abstract-title"})
        title = title_elem[0].getText().strip()
        abstract_text_elem = ads_soup.find_all("div", attrs={"class": "s-abstract-text"})
        description = abstract_text_elem[0].getText().replace("Abstract", "").strip()
        print("TITLE:", title)
        print("DESC:", description)
    except:
        traceback.print_exc()
        return None, None, None
    return title, description, ads_soup


def get_arxiv(ads_soup, refcode):
    arxiv_elem = ads_soup.find_all("dl", attrs={"class": "s-abstract-dl-horizontal"})
    arxiv_link = list(filter(lambda x:
                             "/arXiv:" in x['href'],
                             arxiv_elem[0].find_all("a")
                             ))
    if len(arxiv_link) != 0:
        arxiv_link = "https://arxiv.org/abs/{}".format(
            arxiv_link[0].getText().replace("arXiv:", "").strip()
        )
    else:
        arxiv_link = "https://ui.adsabs.harvard.edu/abs/{}/abstract".format(refcode)
    return arxiv_link


def get_keywords(ads_soup):
    keywords_elem = ads_soup.find_all("ul", attrs={"class": "list-inline"})
    keywords = []
    for ke in keywords_elem:
        keywords.extend([t.getText().replace(";", "").strip() for t in ke.find_all("li")])
    return keywords


def get_ads_data_by_refcodes(_ref_codes, known_refcodes):
    """
    :param _ref_codes:
    :param known_refcodes:
    :return:
    """
    ref_codes = list(filter(lambda x: x not in known_refcodes, _ref_codes))
    papers = []
    if len(ref_codes) != 0:
        print("\n[x] Extracting for each article ADS description and tags .. ")
        for refcode in ref_codes:
            title, description, ads_soup = get_abstract(refcode)
            if title is None:
                _ref_codes = list(filter(lambda x: x != refcode, _ref_codes))
                continue
            arxiv_link = get_arxiv(ads_soup, refcode)
            keywords = get_keywords(ads_soup)
            papers.append(
                [refcode, title, description, arxiv_link, keywords]
            )
    return papers


def get_research_papers(url, known_refcodes):
    """
        Example url:
            http://ned.ipac.caltech.edu/cgi-bin/nph-objsearch?search_type=Near+Position+Search&in_csys=Equatorial&in_equinox=J2000.0&obj_sort=Distance+to+search+center&lon=140.11575d&lat=49.390577d&radius=5.0

        Example usage:
            papers = get_research_papers(ned())
    :param url:
    :return:
    """
    with requests.Session() as s:
        time.sleep(0.5)
        html_content = s.get(url).text
    soup = BeautifulSoup(html_content, "lxml")
    _ref_codes = get_refcodes(soup)
    papers = get_ads_data_by_refcodes(_ref_codes, known_refcodes)
    return papers, _ref_codes