import astropy.units as u
import numpy as np
from astropy.constants import c
from astropy.cosmology import WMAP9 as cosmo

KMS = u.km / u.s
C_KMS = c.to(KMS)


def choose_one(a, b, c):
    return a if a is not None else (b if b is not None else c)


def _redshift_to_velocity(redshift):
    """
    # from source code:
        - astropy.coordinates.spectral_coordinate
        - http://www.astro.ucla.edu/~wright/cosmo_01.htm
    z = (delta(Lambda) / lambda_0) - 1
    Convert a relativistic redshift to a velocity.
    """
    zponesq = (1 + redshift) ** 2
    return (C_KMS * (zponesq - 1) / (zponesq + 1))


def get_velocity_and_distance(redshift):
    """
        Gets values of velocity in km/s and distance in light-years based on redshift
        v = c * z
        v = H * d; d Distanz in Mpc
    :param redshift:
    :return:
    """
    v = _redshift_to_velocity(redshift=redshift)
    dist = (v / cosmo.H(0)).to(u.Mpc)
    return np.round(v.value, 2), np.round(dist.value, 2)


def approximate_spectral_list(spl_input):
    """
        First attempt to approximate the spectra values, based on uncomplete downloaded measurements
        example:
            [None, 123, 1.0, None, 12.0, 11.3, None]
            => [123, 123, 1.0, 6.5, 12.0, 11.3, 11.3]
    :param spl_input: values list for passbands in their order ['u', 'b', 'v', 'g', 'r', 'i', 'z']
    :return:
    """
    from copy import deepcopy
    spl = deepcopy(spl_input)
    for i in range(0, len(spl)):
        if spl[i] is None:
            if i == 0:
                spl[i] = spl[1]
            elif i == len(spl) - 1:
                spl[i] = spl[len(spl) - 2]
            else:
                spl[i] = ((spl[i - 1] + spl[i + 1]) / 2.0) if (spl[i - 1] is not None and spl[i + 1] is not None) \
                    else choose_one(spl[i - 1], spl[i + 1], None)
    return spl


def fillup_object_dict_by_detail_page(object_dict_input, detail_data):
    """
        Overrides object_dict values with those in detail_data
        object_dict will be one of the elements of list_object_data returned from method read_simbad_table_object_info
        Example object_id:
            object_dict_input = {'Z': None,
                'SPEC_G': 11.7,
                'SPEC_I': 12.0,
                'SPEC_R': None,
                'SPEC_U': None,
                'SPEC_Z': 11.3,
             }
    :param object_dict_input: dictionary with object information
    :param detail_data: dictionary with object information
    :return:
    """
    from copy import deepcopy
    object_dict = deepcopy(object_dict_input)
    # first and second choices
    for key in ['u', 'b', 'v', 'g', 'r', 'i', 'z']:
        if ("SPEC_" + key.upper()) not in object_dict.keys():
            object_dict['SPEC_' + key.upper()] = None
    spectra_list = []
    for spectra in ['u', 'b', 'v', 'g', 'r', 'i', 'z']:
        spectra_list.append(
            choose_one(
                object_dict['SPEC_' + spectra.upper()],
                detail_data['Flux ' + spectra.upper()],
                detail_data['Flux ' + spectra])
        )
    spectra_list = approximate_spectral_list(spectra_list)
    res = dict(list(zip(['u', 'b', 'v', 'g', 'r', 'i', 'z'], spectra_list)))
    for key in res.keys():
        object_dict['SPEC_{}'.format(key.upper())] = res[key]
    object_dict["Z"] = detail_data["Redshift"]
    return object_dict
