"""
Copyright (C) 2023, Khun Sang Phukon


This code requires https://git.ligo.org/khun.phukon/lalsuite/-/tree/taylorF2ecc_coeff/

"""

import re
import numpy
import pycbc
import pycbc.libutils
import pycbc.tmpltbank
from pycbc.tmpltbank.lambda_mapping import generate_mapping, get_chirp_params, ethinca_order_from_string
from pycbc.tmpltbank.lambda_mapping import pycbcValidOrdersHelpDescriptions
from lal import MTSUN_SI, PI, CreateREAL8Vector, CreateDict

lalsimulation = pycbc.libutils.import_optional('lalsimulation')


ValidEccOrders = ['zeroPN', 'onePN', 'onePointFivePN', 'twoPN', 'twoPointFivePN', 'threePN' ]

EccValidOrdersHelpDescriptions="""
     * zeroPN: Will only include the dominant term
     * onePN: Will only the correction at 1PN
     * onePointFivePN: Will include correction terms to 1.5PN.
     * twoPN: Will include correction terms to 2PN.
     * twoPointFivePN: Will include correction terms to 2.5PN.
     * threePN: Will include correction terms to 3PN.
"""

LAL_MAX_ECC_PN_ORDER = 6


def update_dictionary(qc_dictionary, eccentric_dictionary, ecc_map_only=False):
    """
    qc_dictionary: dictionary of lambda parameter for quasicircular PN co-efficients
    eccentric_dictionary: dictionary of lambda parameter for eccentric PN co-efficients
    
    This function will do nothing if we only want dictionary of eccentric phase's lambda params
    """
    if ecc_map_only is False:
        eccentric_dictionary.update(qc_dictionary)


def generate_mapping_ecc(pn_order, ecc_pn_order, ecc_map_only=False):
    """
    This function will take an order string and return a mapping between
    components in the metric and the various Lambda components (QC PN and Ecc PN). 

    This must be
    used (and consistently used) when generating the metric *and* when
    transforming to/from the xi_i coordinates to the lambda_i coordinates.

    Parameters
    ----------
    ecc_pn_order : string
        A string containing  PN order of eccentric corrections. Valid values are
        given by ValidEccOrders.
    pn_order : string
        A string containing  PN order of quasicircular corrections. Valid values
        are given by pycbcValidTmpltbankOrders

    Returns
    --------
    mapping : dictionary
        A mapping between the active Lambda terms and index in the metric
    """

    qc_mapping_dict = generate_mapping(pn_order)
    qc_mapping_len =  len(qc_mapping_dict.keys())
    mapping = {}
    mapping['LambdaEcc0'] = qc_mapping_len + 0
    if ecc_pn_order == 'zeroPN':
        update_dictionary(qc_mapping_dict, mapping, ecc_map_only=ecc_map_only)
        return mapping
    mapping['LambdaEcc2'] = qc_mapping_len + 1
    if ecc_pn_order == 'onePN':
        update_dictionary(qc_mapping_dict, mapping, ecc_map_only=ecc_map_only)
        return mapping
    mapping['LambdaEcc3'] = qc_mapping_len + 2
    if ecc_pn_order == 'onePointFivePN':
        update_dictionary(qc_mapping_dict, mapping, ecc_map_only=ecc_map_only)
        return mapping
    mapping['LambdaEcc4'] = qc_mapping_len + 3
    if ecc_pn_order == 'twoPN':
        update_dictionary(qc_mapping_dict, mapping, ecc_map_only=ecc_map_only)
        return mapping
    mapping['LambdaEcc5'] = qc_mapping_len + 4
    if ecc_pn_order == 'twoPointFivePN':
        update_dictionary(qc_mapping_dict, mapping, ecc_map_only=ecc_map_only)
        return mapping
    mapping['LambdaEcc6'] = qc_mapping_len + 5
    mapping['LogLambdaEcc6'] = qc_mapping_len + 6
    if ecc_pn_order == 'threePN':
        update_dictionary(qc_mapping_dict, mapping, ecc_map_only=ecc_map_only)
        return mapping
    raise ValueError("Eccentricity PN  Order %s is not understood." %(ecc_pn_order))


def generate_inverse_mapping_ecc(pn_order, ecc_pn_order, ecc_map_only=False):
    """Genereate a lambda entry -> PN order map.

    This function will generate the opposite of generate mapping. So where
    generate_mapping gives dict[key] = item this will give
    dict[item] = key. Valid PN orders are:
    {}

    Parameters
    ----------
    ecc_pn_order : string
        A string containing  PN order of eccentric corrections. Valid values are
        given by ValidEccOrders.
    pn_order : string
        A string containing  PN order of quasicircular corrections. Valid values
        are given by pycbc.tmplbank.pycbcValidTmpltbankOrders

    Returns
    --------
    mapping : dictionary
        An inverse mapping between the active Lambda terms and index in the
        metric
    """
    mapping = generate_mapping_ecc(pn_order, ecc_pn_order, ecc_map_only=ecc_map_only)
    inv_mapping = {}
    for key,value in mapping.items():
        inv_mapping[value] = key

    return inv_mapping

generate_inverse_mapping_ecc.__doc__ = \
    generate_inverse_mapping_ecc.__doc__.format(EccValidOrdersHelpDescriptions)




def get_chirp_params_ecc(mass1, mass2, spin1z, spin2z, ecc, f0, f_ecc, order, order_ecc, 
                     quadparam1=None, quadparam2=None, lambda1=None,
                     lambda2=None):
    """
    Take a set of masses and spins and convert to the various lambda
    coordinates that describe the orbital phase. Accepted PN orders are:
    {}

    Parameters
    ----------
    mass1 : float or array
        Mass1 of input(s).
    mass2 : float or array
        Mass2 of input(s).
    spin1z : float or array
        Parallel spin component(s) of body 1.
    spin2z : float or array
        Parallel spin component(s) of body 2.
    ecc : float or array
        Eccentricity of orbit
    f0 : float
        This is an arbitrary scaling factor introduced to avoid the potential
        for numerical overflow when calculating this. Generally the default
        value (70) is safe here. **IMPORTANT, if you want to calculate the
        ethinca metric components later this MUST be set equal to f_low.**
        This value must also be used consistently (ie. don't change its value
        when calling different functions!).
    f_ecc : float
        Frequency at which eccentricity is  defined
    order : string
        The Post-Newtonian order that is used to translate from masses and
        spins to the lambda_i parameters of quasi-circular phase. Valid orders given 
        by pycbc.tmplbank.pycbcValidTmpltbankOrders.
    order_ecc : string
        The Post-Newtonian order that is used to translate masses and 
        eccentricity to the lambda_i parameters of eccentric phase. Valid orders given above. 
    Returns
    --------
    lambdas : list of floats or numpy.arrays
        The lambda coordinates for the input system(s)
    """
    lambdas_qc = get_chirp_params(mass1, mass2, spin1z, spin2z, f0, order, 
                     quadparam1=quadparam1, quadparam2=quadparam2, 
                     lambda1=lambda1, lambda2=lambda2)

    # Determine whether array or single value input
    try:
        num_points = len(lambdas_qc[0])
    except TypeError:
        num_points = 1
        mass1 = numpy.array([mass1])
        mass2 = numpy.array([mass2])
        ecc = numpy.array([ecc])
    if len(ecc) != len(mass1):
        raise ValueError('Dimension of eccentricity input values does not match with that of the rest of parameters')

    mass1_v = CreateREAL8Vector(num_points)
    mass1_v.data[:] = mass1[:]
    mass2_v = CreateREAL8Vector(num_points)
    mass2_v.data[:] = mass2[:]
    ecc_v = CreateREAL8Vector(num_points)
    ecc_v.data[:] = ecc[:]

    ecc_order_num = ethinca_order_from_string(order_ecc)

    phasing_arr_ecc = lalsimulation.SimInspiralTaylorF2EccAlignedPhasingArray\
                                    (mass1_v, mass2_v, ecc_v, f_ecc, ecc_order_num)

    ecc_vec_len = LAL_MAX_ECC_PN_ORDER + 1
    phasing_ecc_vs = numpy.zeros([num_points, ecc_vec_len])
    phasing_ecc_vlogvs = numpy.zeros([num_points, ecc_vec_len])

    lng = num_points
    jmp = num_points * ecc_vec_len

    for vc_id in range(ecc_vec_len):
        phasing_ecc_vs[:,vc_id] = phasing_arr_ecc.data[lng*vc_id : lng*(vc_id+1)]
        phasing_ecc_vlogvs[:,vc_id] = phasing_arr_ecc.data[jmp + lng*vc_id : jmp + lng*(vc_id+1)]

    pim = PI * (mass1 + mass2)*MTSUN_SI
    pmf = pim * f0
    pmf13 = pmf**(1./3.)
    logpim13 = numpy.log((pim)**(1./3.))

    mapping = generate_inverse_mapping_ecc(order, order_ecc, ecc_map_only=True)

    qc_mapping_dict = generate_mapping(order)
    qc_mapping_len =  len(qc_mapping_dict.keys())

    lambdaEcc_str = '^LambdaEcc([0-6]+)'
    loglambdaEcc_str = '^LogLambdaEcc([0-6]+)'
  
    lambdas = []
    for map_key in list(mapping.values()):
        # RE magic engage!
        rematch = re.match(lambdaEcc_str, map_key)
        if rematch:
            ecc_order = int(rematch.groups()[0])
            term = phasing_ecc_vs[:,ecc_order]
            term = term + logpim13 * phasing_ecc_vlogvs[:,ecc_order]     
            lambdas.append(term * pmf13**(-5 - 19/3 + ecc_order) )
            continue
        rematch = re.match(loglambdaEcc_str, map_key)
        if rematch:
            ecc_order = int(rematch.groups()[0])
            lambdas.append(phasing_ecc_vlogvs[:,ecc_order] * logpim13 * pmf13**(-5 - 19/3 + ecc_order))
            continue
        err_msg = "Failed to parse " +  map_key
        raise ValueError(err_msg)
    if num_points == 1:
        return list(lambdas_qc) + [l[0] for l in lambdas]
    else:
        return list(lambdas_qc) + list(lambdas)

get_chirp_params_ecc.__doc__ = \
    get_chirp_params_ecc.__doc__.format(pycbcValidOrdersHelpDescriptions)
