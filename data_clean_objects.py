
def append_to_replacements_dict(newval, oldkeys):
    """ extends dictionary by pairs of (oldkey, newval) """
    global replacements_dict
    for oldkey in oldkeys:
        replacements_dict[oldkey] = newval


def append_to_replacements_by_startswith_dict(newval, oldkeys):
    """ extends dictionary by pairs of (oldkey, newval) """
    global replacements_by_startswith_dict
    for oldkey in oldkeys:
        replacements_by_startswith_dict[oldkey] = newval


# ******************************************************************************************************************** #
# remove by replacements, e.g. "submillimetre" -> "submillimeter",
replacements_dict = {
    "submillimetre": "submillimeter",
    "x rays": "x ray",

    "catalogues": "catalog",
    "catalogs": "catalog",
    "astronomical data bases miscellaneous": "catalog",
    'celestial objects catalog': "catalog",
    'atlases': 'catalog',

    'surveys': 'survey',
    'survey galaxies fundamental parameters galaxies statistics': 'survey',
    "surveys: galaxies: fundamental parameters: galaxies: statistics": "surveys",

    'v m': 'star clusters',
    'ω centauri': 'star clusters',

    'γ dor': 'blue stragglers',
    'zodiacal dust': 'circumstellar matter',

    'supermassive black holes': 'black holes',
    'chandrasekhar limit': 'black holes',
    'intermediate mass black holes': 'black holes'

}


# replacements by beginning strings, e.g. 'supernovae are cool' -> 'supernovae'
replacements_by_startswith_dict = {

    'ultraviolet': 'ultraviolet',
    'submillimeter': 'submillimeter',
    'radio': 'radio',
    'infrared': 'infrared',
    'near infrared': 'infrared',
    'gamma ray': 'gamma ray',
    'x ray': 'x ray sources',

    'planetary systems': 'planetary systems',
    'protoplanetary': 'planetary systems',
    'planets': 'planetary systems',
    'exoplanet': 'planetary systems',
    'minor planets': 'planetary systems',

}


# a list of tags to be ignored
ignore_tag_list = [
    'terzan', 'sncs', 'snbb', 'ptfkmb', 'ptfdav', 'ptfbij', 'psr j', 'psr b',
    'p puppis', 'omega centauri', 'ophiuchi clouds', 'addenda', 'and ic', 'babić', 'balloons',
    'car ob', 'cgcg', 'delta scuti', 'melotte', 'galah collaboration', 'gaia collaboration',
    'dinh v trung', 'ddo system', 'dark ages', 'collinder', 'co pis', 'clouds', 'chi persei',
    'chronology', 'magellanic clouds', 'jcmt gould belt survey team', 'request for observations',
    'individual galaxies m', 'individual ic', 'individual m', 'berkeley', 'm group',
    'dlt collaboration', 'cygnus constellation', 'crab nebula', 'gum nebula', 'gould belt',
    'equatorial regions', 'photon dominated region pdr', 'polar regions', 'regions',
    'lodén', 'classifications', 'eclipses', 'errata', 'forbidden transitions', 'herbig ae be',
    'spirits buu', 'mergers', 'mapping', 'lunar occultation', 'reliability', 'outflows',
    'northern hemisphere', 'northern sky', 'morphology', 'epic', 'error analysis',
    'scattering', 'mindstep consortium', 'magic collaboration', 'magnetohydrodynamics mhd',
    'lcogt robonet consortium', 'hitomi collaboration', 'harmonics', 'retrograde orbits',
    'dust', 'extinction', 'extra planar gas', 'dynamics', 'formaldehyde',
    'charge coupled devices', 'centroids',  'atmospheric effects', 'gas collaboration',
    'frequencies', 'frequency distribution', 'normal density functions', 'high resolution', 'instabilities',
    'interactions', 'linear polarization', 'luminous intensity', 'periodic variations', 'scattering',
    'scaling relations', 'retrograde orbits', 'synthetic apertures', 'transient',
]

# ******************************************************************************************************************** #
append_to_replacements_dict('stellar studies',
                 ['red giant stars', 'red giants', 'white dwarf stars',
                  "hertzsprung russell and c m diagrams",
                  "hertzsprung russell and colour magnitude diagrams",
                  "white dwarfs", 'wolf rayet stars', 'triple stars', 'variable stars', 'variables other',
                  'a stars', 'halo stars',  'o stars', 'ob associations', 'ob stars',
                  'symbiotic stars', 'supermassive stars', 'supergiant stars', 'neutron stars',
                  "stars: magnetic fields", 'tars oscillations',
                  'photosphere', 'peculiar stars', 'db stars',  'b stars', 'be formation', 'subdwarfs',
                  'm dwarf stars', 'm stars', 'distribution of stars',
                  'blue stars', 'blue stragglers', 'asymptotic giant branch stars', 'binary stars',
                  'brown dwarfs', 'k stars', 'giant stars', 'f stars',
                  'first stars', 'horizontal branch stars', 'main sequence stars',
                  'population ii stars', 'pre main sequence stars',
                  'eclipsing stellar studies', 'low mass x ray stellar studies', 'post stellar studies',
                  'pre stellar studies', 'long period variables', 'main sequence'
                  ])

append_to_replacements_dict('statistics',
                 ['statistical', 'statistics computation', 'statistics machine learning',
                  'astrostatistics', 'spatial distribution', 'correlation',
                  'tables data', 'density distribution', 'fluorescence', 'flux density',
])

append_to_replacements_dict('instrumentation',
                 ['very large array vla', 'very large telescope', 'virtual observatory tools'])

append_to_replacements_dict('star clusters',
                 ['young massive clusters'])

append_to_replacements_dict('galaxies',
                ['peculiar galaxies', 'nearby galaxies', 'compact galaxies', 'irregular galaxies',
                 'interacting galaxies', 'dwarf galaxies', 'disk galaxies',
                 'continuum galaxies', 'elliptical galaxies', 'galaxies',
                 'low surface brightness galaxies', 'sagittarius dwarf spheroidal galaxy'
                ])



# ******************************************************************************************************************** #
append_to_replacements_by_startswith_dict('astrophysics',
                ['astrophysics', 'tidal', 'cosmology', 'gravitational len', 'accretion', 'methods',
                 'astronomy', 'time', 'mass to light', 'astronomical', 'thermal emission',
                 'atomic', 'celestial mechanics', 'polarization', 'nuclear ', 'high energy',
                 'astroparticle physics', 'acceleration of particles', 'asteroseismology', 'magnetic ',
                 'light curve', 'line ', 'light transmission', 'cosmic ', 'dark ', 'electron ',
                 'evolution', 'equation', 'cosmological parameters', 'general relativity and quantum cosmology',
                 'primordial nucleosynthesis', 'nuclei', 'nucleosynthesis', 'n body simulations',
                 'large scale structure of universe', 'many body problem', 'relativistic processes',
                 'centimeter waves', 'gravitational waves', 'shock waves', 'missing mass astrophysics',
                 'millimiter astronomy',  'hst photometry', 'electrophotometry', 'equipartition theorem',
                 'gravitation',])

append_to_replacements_by_startswith_dict('astrometry',
                ['velocity', 'three dimensional', 'distance', 'parallax', 'astrometric', 'angular velocity',
                 'techniques', 'visual binaries', 'photographic', 'photometric', 'luminosity',
                 'kinematics', 'superhigh frequencies', 'space astrometry', 'mass',
                 'absolute magnitude', 'color ', 'orbit determination and improvement',
                 'multiple stars star positions',
                 'reference systems', 'reference stars', 'washington system', 'celestial reference systems',
                 'proper motions', 'red shift', 'radial ', 'observational', 'occultations', 'optical',
                 'point sources', 'point spread functions', 'hydrodynamic', 'position location',
                 'photoelectric magnitudes', 'faint objects'
                 ])

append_to_replacements_by_startswith_dict('spectral analysis',
                ['spectroscopic', 'spectral', 'spectrum', 'photometry', 'brightness', 'wavelengths',
                 'radiation mechanisms', 'visible spectrum', 'ubv ', 'surface photometry',
                 'balmer series', 'spectrophotometry',
                 'absorption spectra', 'angular resolution', 'masers', 'water masers', 'k lines',
                 'emission', 'continuous spectra',
                 'continuous radiation', 'diffuse radiation', 'extragalactic radio sources',
                 'far ultraviolet radiation', 'key words radiation mechanisms non thermal', 'radiant flux density',
                 'radiative transfer', 'radio', 'relic radiation', 'temperature gradients'])

append_to_replacements_by_startswith_dict('stellar studies',
                ['stars binaries', 'stellar', 'stars', 'star', 'binaries',
                 'astrometric and interferometric binaries', 'hertzsprung', 'supergiants', 'stellar studies',
                 'cepheid', 'sun ', 'young star', 'young stellar',
                 'sstellar', 'solar', 'rr lyrae stellar',
                 'cataclysmic', 'dstellar'])

append_to_replacements_by_startswith_dict('instrumentation',
                ['schmidt cameras', 'schmidt telescope', 'cassegrain', 'instrumentation', 'telescopes',
                 'cameras', 'calibrating', 'ccd', 'satellite borne photography', 'hubble', 'ligo',
                 'bolometers', 'fabry perot interferometers'
                 ])

append_to_replacements_by_startswith_dict('galaxies',
                ['galaxy', 'galaxies', 'galactic', 'milky way', 'spiral', 'seyfert', 'barred ',
                 'local group'])

append_to_replacements_by_startswith_dict('star clusters',
                ['orion', 'globular', 'globular clusters', 'open clusters',
                 'clusters', 'pleiades', 'open cluster', 'open star clusters'])

append_to_replacements_by_startswith_dict('chemistry',
                ['carbon', 'nitrogen', 'neon', 'sulfur', 'oxygen', 'molecular',
                 'hydrogen', 'helium', 'metallic',
                 'abundance', 'argon', 'calcium', 'chemical', 'h ', 'hii ', 'ho ',
                 'neutral hydrogen', 'astrochemistry', 'reionization'])

append_to_replacements_by_startswith_dict('statistics',
                 ['astrostatistics', 'statistics', 'bayesian ', 'sampling', 'data '])

append_to_replacements_by_startswith_dict('black holes',
                ['black hole', 'astrophysical black holes', 'quasars', 'active galactic nuclei',
                 'active galaxies', 'agb and post agb', 'bl lacertae'])

append_to_replacements_by_startswith_dict('catalog',
                ['catalogue', 'catalog', 'sky survey', 'southern sky', 'sdss collaboration', 'survey'])

append_to_replacements_by_startswith_dict('novae and supernovae',
                ['supernovae', 'supernova', 'planetary nebulae', 'novae',
                 'transients supernovae', 'supernovae general',
                 'pulsars', 'binary pulsars', 'core collapse supernovae',
                 'millisecond pulsar', 'millisecond pulsars', 'compact objects', 'dense matter',
                 ])

append_to_replacements_by_startswith_dict('nebulae',
                ['nebulae', 'reflection nebulae', 'filamentary nebulae'])

append_to_replacements_by_startswith_dict('interstellar medium',
                ['ism', 'interstellar', 'intergalactic medium', 'interplanetary medium'])

append_to_replacements_by_startswith_dict('early or late types',
                ['early ', 'late '])

append_to_replacements_by_startswith_dict("asteroids and meteorids",
                ['asteroids general',  'meteorites', 'meteoroids', 'meteors'])

append_to_replacements_by_startswith_dict("data analysis",
                ['russell and colour', 'hr diagram', 'magnitude diagrams',
                 'computer science machine learning', 'markov chain monte carlo',
                 'image processing', 'imaging techniques', 'computational methods'])