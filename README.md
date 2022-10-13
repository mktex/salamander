### Overview:
Details on the implementation find at blog article
[Blog post](http://machine-learning.bfht.eu/learning-about-the-universe-as-a-hobby-astronomer-heres-one-way-to-go-about-it-i-of-iv)

There are two options on new objects:

1. Is the object in SDSS database, then:

-  get the download.txt and place it under ./data folder:
<pre>
    python download_sdss.py
</pre>
For a simple trial one FITS file is left under ./data. Follow procedure
described in blog article part II ([Data collection](http://machine-learning.bfht.eu/learning-about-the-universe-as-a-hobby-astronomer-data-collection-and-analysis-ii-of-iv)) to try out newly
region of sky.

-  process data from SDSS:
<pre>
    python read_sdss.py
</pre>


2. If not in SDSS then in Simbad:

-  download from Simbad by coordiates:
<pre>
    python read_simbad.py 09 12 19.497 -24 10 21.36
</pre>


3. Getting associations to scientific papers on specific astronomical
   objects:

 - Give 10 recommendations for one object:
<pre>
    python reco.py papers 'salamander' 'NGC 2784' 10
</pre>


 - Give 5 recommendations per object and save results to a CSV:
<pre>
    python main_reco.py process 'salamander' 5 'NGC 2292' 'NGC 2613' 'NGC 3115'
</pre>


 - Rebuild model:
<pre>
    python main_reco.py update 'salamander'
</pre>


Data processed so far as well as model are saved in pickl files under
./data folder.

Original FITS files were not uploaded due to size.

Creating a new conda environment and required Python modules:
conda create -n salamander python=3.7  
conda activate salamander conda install progressbar2  
conda install numpy scipy scikit-learn pandas seaborn pymc3  
conda install nltk
conda install astropy
conda install beautifulsoup4
conda install requests
conda install lxml

