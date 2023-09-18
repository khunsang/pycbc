import logging
from pycbc.tmpltbank.option_utils import insert_metric_calculation_options, metricParameters 
from pycbc.tmpltbank.calc_moments import calculate_moment, interpolate_psd, get_moments
from pycbc.tmpltbank.lambda_mapping import get_ethinca_orders, pycbcValidOrdersHelpDescriptions 
from pycbc.tmpltbank.coord_utils import (get_mu_params, get_covaried_params, massRangeParameters, 
                                            get_random_mass_point_particles)
from pycbc.tmpltbank.lambda_mapping_eccentric import EccValidOrdersHelpDescriptions, get_chirp_params_ecc
from pycbc.types import positive_float
import numpy

def verify_eccentric_metric_calculation_options(opts, parser):
    """
    Parses the eccentric metric calculation options given and verifies that they are
    correct.

    Parameters
    ----------
    opts : argparse.Values instance
        Result of parsing the input options with OptionParser
    parser : object
        The OptionParser instance.
    """
    if not opts.pn_order:
        parser.error("Must supply --pn-order")
    if not opts.ecc_pn_order:
        parser.error("Must supply --ecc-pn-order")

class massRangeParametersEccentric(massRangeParameters):
    """
    This class holds all of the options that are parsed in the function
    insert_mass_range_option_group, insert_eccentricity_range_options
    and all products produced using these options. It can also be initialized
    from the __init__ function providing directly the options normally
    provided on the command line
    """
    def __init__(self, minMass1, maxMass1, minMass2, maxMass2, 
                 minEccentricity, maxEccentricity, 
                 maxNSSpinMag=0, maxBHSpinMag=0, maxTotMass=None,
                 minTotMass=None, maxEta=None, minEta=0,
                 max_chirp_mass=None, min_chirp_mass=None,
                 ns_bh_boundary_mass=None, nsbhFlag=False,
                 remnant_mass_threshold=None, ns_eos=None, use_eos_max_ns_mass=False,
                 delta_bh_spin=None, delta_ns_mass=None):
        self.minEccentricity = minEccentricity
        self.maxEccentricity = maxEccentricity
        super(massRangeParametersEccentric, self).__init__(minMass1, maxMass1, minMass2, maxMass2,
                                                           maxNSSpinMag=maxNSSpinMag, maxBHSpinMag=maxBHSpinMag,
                                                           maxTotMass=maxTotMass, minTotMass=minTotMass, 
                                                           maxEta=maxEta, minEta=minEta, max_chirp_mass=max_chirp_mass,
                                                           min_chirp_mass=min_chirp_mass, ns_bh_boundary_mass=ns_bh_boundary_mass,
                                                           nsbhFlag=nsbhFlag, remnant_mass_threshold=remnant_mass_threshold,
                                                           ns_eos=ns_eos, use_eos_max_ns_mass=use_eos_max_ns_mass,
                                                           delta_bh_spin=delta_bh_spin, delta_ns_mass=delta_ns_mass)
        if minEccentricity > maxEccentricity:
            raise ValueError("Minimum eccentricity cannot be greater than maximum eccentricity. Check option")
    @classmethod
    def from_argparse(cls, opts, nonSpin=False):
        """
        Initialize an instance of the massRangeParametersEccentric class from an
        argparse.OptionParser instance. This assumes that
        insert_mass_range_option_group, insert_eccentricity_range_options,
        verify_mass_range_options, verify_eccentricity_options 
        have already been called before initializing the class.
        """
        if nonSpin:
            return cls(opts.min_mass1, opts.max_mass1, opts.min_mass2,
                       opts.max_mass2, opts.min_ecc, opts.max_ecc,  maxTotMass=opts.max_total_mass,
                       minTotMass=opts.min_total_mass, maxEta=opts.max_eta,
                       minEta=opts.min_eta, max_chirp_mass=opts.max_chirp_mass,
                       min_chirp_mass=opts.min_chirp_mass,
                       remnant_mass_threshold=opts.remnant_mass_threshold,
                       ns_eos=opts.ns_eos, use_eos_max_ns_mass=opts.use_eos_max_ns_mass,
                       delta_bh_spin=opts.delta_bh_spin, delta_ns_mass=opts.delta_ns_mass)
        else:
            return cls(opts.min_mass1, opts.max_mass1, opts.min_mass2,
                       opts.max_mass2, opts.min_ecc, opts.max_ecc,  maxTotMass=opts.max_total_mass,
                       minTotMass=opts.min_total_mass, maxEta=opts.max_eta,
                       minEta=opts.min_eta, maxNSSpinMag=opts.max_ns_spin_mag,
                       maxBHSpinMag=opts.max_bh_spin_mag,
                       nsbhFlag=opts.nsbh_flag,
                       max_chirp_mass=opts.max_chirp_mass,
                       min_chirp_mass=opts.min_chirp_mass,
                       ns_bh_boundary_mass=opts.ns_bh_boundary_mass,
                       remnant_mass_threshold=opts.remnant_mass_threshold,
                       ns_eos=opts.ns_eos, use_eos_max_ns_mass=opts.use_eos_max_ns_mass,
                       delta_bh_spin=opts.delta_bh_spin, delta_ns_mass=opts.delta_ns_mass)

    def is_outside_range_all_params(self, mass1, mass2, spin1z, spin2z, eccentricity):
        """
        Test if a given location in mass1, mass2, spin1z, spin2z, eccentricity  is 
        within the range of parameters allowed by the massParams object.
        """
        #Mass, spin test
        if self.is_outside_range(mass1, mass2, spin1z, spin2z)==1:
            return 1
        # Eccentricity test
        if eccentricity * 1.001 < self.minEccentricity:
            return 1
        if eccentricity * 1.001 > self.maxEccentricity:
            return 1
        return 0

class metricParametersEccentric(metricParameters):
    """
    This class holds all of the options that are parsed in the function
    insert_ecc_options_in_metric_calculation_options
    and all products produced using these options. It can also be initialized
    from the __init__ function, providing directly the options normally
    provided on the command line.
    """
    def __init__(self, pnOrder, eccpnOrder, fLow, fUpper, fEcc, deltaF, f0=70,
                 write_metric=False):
        self.fEcc=fEcc
        self.eccpnOrder = eccpnOrder
        super(metricParametersEccentric, self).__init__(pnOrder, fLow, fUpper, deltaF, f0=f0,
                 write_metric=write_metric)
    
    @classmethod
    def from_argparse(cls, opts):
        """
        Initialize an instance of the metricParametersEccentric class from an
        argparse.OptionParser instance. This assumes that
        insert_ecc_options_in_metric_calculation_options
        and
        verify_eccentric_metric_calculation_options
        have already been called before initializing the class.
        """
        return cls(opts.pn_order, opts.ecc_pn_order, opts.f_low, opts.f_upper, opts.f_ecc,\
                    opts.delta_f,f0=opts.f0, write_metric=opts.write_metric)



def get_random_mass_eccentric_point_particles(numPoints, massRangeParams):
    """
    This function will generate a large set of points within the chosen mass, spin 
    and eccentricity space. 

    Parameters
    ----------
    numPoints : int
        Number of systems to simulate
    massRangeParams : massRangeParametersEccentricity instance
        Instance holding all the details of mass ranges and spin ranges.

    Returns
    --------
    mass1 : float
        Mass of heavier body.
    mass2 : float
        Mass of lighter body.
    spin1z : float
        Spin of body 1.
    spin2z : float
        Spin of body 2.
    eccentricity: float
        Eccentricity of orbit
    """
    mass1, mass2, spin1z, spin2z = get_random_mass_point_particles(numPoints, massRangeParams)
    size = len(mass1)
    minEcc = massRangeParams.minEccentricity
    maxEcc = massRangeParams.maxEccentricity
    eccentricity = numpy.random.random(size) * (maxEcc - minEcc) + minEcc
    return mass1, mass2, spin1z, spin2z, eccentricity


def estimate_mass_range_ecc(numPoints, massRangeParams, metricParams, fUpper,\
                        covary=True):
    """
    This function will generate a large set of points with random masses, spins and 
    eccentricities and translate these points
    into the xi_i coordinate system for the given upper frequency cutoff.

    Parameters
    ----------
    numPoints : int
        Number of systems to simulate
    massRangeParams : massRangeParametersEccentric instance
        Instance holding all the details of mass ranges and spin ranges.
    metricParams : metricParametersEccentric instance
        Structure holding all the options for construction of the metric
        and the eigenvalues, eigenvectors and covariance matrix
        needed to manipulate the space.
    fUpper : float
        The value of fUpper to use when getting the mu coordinates from the
        lambda coordinates. This must be a key in metricParams.evals and
        metricParams.evecs (ie. we must know how to do the transformation for
        the given value of fUpper). It also must be a key in
        metricParams.evecsCV if covary=True.
    covary : boolean, optional (default = True)
        If this is given then evecsCV will be used to rotate from the Cartesian
        coordinate system into the principal coordinate direction (xi_i). If
        not given then points in the original Cartesian coordinates are
        returned.


    Returns
    -------
    xis : numpy.array
        A list of the positions of each point in the xi_i coordinate system.
    """
    vals_set = get_random_mass_eccentric_point_particles(numPoints, massRangeParams)
    mass1 = vals_set[0]
    mass2 = vals_set[1]
    spin1z = vals_set[2]
    spin2z = vals_set[3]
    eccentricity = vals_set[4]
    if covary:
        lambdas = get_cov_params_ecc(mass1, mass2, spin1z, spin2z, eccentricity, metricParams,
                                 fUpper)
    else:
        lambdas = get_conv_params_ecc(mass1, mass2, spin1z, spin2z, eccentricity, metricParams,
                                  fUpper)

    return numpy.array(lambdas)


def get_conv_params_ecc(mass1, mass2, spin1z, spin2z, eccentricity, metricParams, fUpper,
                    lambda1=None, lambda2=None, quadparam1=None,
                    quadparam2=None):
    """
    Function to convert between masses, spins and eccentricity and locations in the mu
    parameter space. Mu = Cartesian metric, but not principal components.

    Parameters
    -----------
    mass1 : float
        Mass of heavier body.
    mass2 : float
        Mass of lighter body.
    spin1z : float
        Spin of body 1.
    spin2z : float
        Spin of body 2.
    eccentricity : float
        Eccentricity of the orbit
    metricParams : metricParameters instance
        Structure holding all the options for construction of the metric
        and the eigenvalues, eigenvectors and covariance matrix
        needed to manipulate the space.
    fUpper : float
        The value of fUpper to use when getting the mu coordinates from the
        lambda coordinates. This must be a key in metricParams.evals and
        metricParams.evecs (ie. we must know how to do the transformation for
        the given value of fUpper)

    Returns
    --------
    mus : list of floats or numpy.arrays
        Position of the system(s) in the mu coordinate system
    """

    # Do this by masses -> lambdas
    lambdas = get_chirp_params_ecc(mass1, mass2, spin1z, spin2z, eccentricity,
                                   metricParams.f0, metricParams.fEcc,
                                   metricParams.pnOrder, metricParams.eccpnOrder,
                                   lambda1=lambda1, lambda2=lambda2,
                                   quadparam1=quadparam1, quadparam2=quadparam2)
    # and lambdas -> mus
    mus = get_mu_params(lambdas, metricParams, fUpper)
    return mus

def get_cov_params_ecc(mass1, mass2, spin1z, spin2z, eccentricity, metricParams, fUpper,
                   lambda1=None, lambda2=None, quadparam1=None,
                   quadparam2=None):
    """
    Function to convert between masses and spins and locations in the xi
    parameter space. Xi = Cartesian metric and rotated to principal components.

    Parameters
    -----------
    mass1 : float
        Mass of heavier body.
    mass2 : float
        Mass of lighter body.
    spin1z : float
        Spin of body 1.
    spin2z : float
        Spin of body 2.
    eccentricity : float
        Eccentricity of the orbit
    metricParams : metricParametersEccentric instance
        Structure holding all the options for construction of the metric
        and the eigenvalues, eigenvectors and covariance matrix
        needed to manipulate the space.
    fUpper : float
        The value of fUpper to use when getting the mu coordinates from the
        lambda coordinates. This must be a key in metricParams.evals,
        metricParams.evecs and metricParams.evecsCV
        (ie. we must know how to do the transformation for
        the given value of fUpper)

    Returns
    --------
    xis : list of floats or numpy.arrays
        Position of the system(s) in the xi coordinate system
    """

    # Do this by doing masses - > lambdas -> mus
    mus = get_conv_params_ecc(mass1, mass2, spin1z, spin2z, eccentricity,
                              metricParams, fUpper,
                              lambda1=lambda1, lambda2=lambda2,
                              quadparam1=quadparam1, quadparam2=quadparam2)
    # and then mus -> xis
    xis = get_covaried_params(mus, metricParams.evecsCV[fUpper])
    return xis

def get_point_distance_ecc(point1, point2, metricParams, fUpper, number_coordinate=-1):
    """
    Function to calculate the mismatch between two points, supplied in terms
    of the masses, spins and eccentricity  using the xi_i parameter space metric to
    approximate the mismatch of the two points. Can also take one of the points
    as an array of points and return an array of mismatches (but only one can
    be an array!)

    point1 : List of floats or numpy.arrays
        point1[0] contains the mass(es) of the heaviest body(ies).
        point1[1] contains the mass(es) of the smallest body(ies).
        point1[2] contains the spin(es) of the heaviest body(ies).
        point1[3] contains the spin(es) of the smallest body(ies).
        point1[4] contains the eccentriciti(es) of the orbit(s)
    point2 : List of floats
        point2[0] contains the mass of the heaviest body.
        point2[1] contains the mass of the smallest body.
        point2[2] contains the spin of the heaviest body.
        point2[3] contains the spin of the smallest body.
        point2[4] contains the eccentriciti(es) of the orbit(s)
    metricParams : metricParametersEccentric instance
        Structure holding all the options for construction of the metric
        and the eigenvalues, eigenvectors and covariance matrix
        needed to manipulate the space.
    fUpper : float
        The value of fUpper to use when getting the mu coordinates from the
        lambda coordinates. This must be a key in metricParams.evals,
        metricParams.evecs and metricParams.evecsCV
        (ie. we must know how to do the transformation for
        the given value of fUpper)

    Returns
    --------
    dist : float or numpy.array
        Distance between the point2 and all points in point1
    xis1 : List of floats or numpy.arrays
        Position of the input point1(s) in the xi_i parameter space
    xis2 : List of floats
        Position of the input point2 in the xi_i parameter space
    """
    aMass1 = point1[0]
    aMass2 = point1[1]
    aSpin1 = point1[2]
    aSpin2 = point1[3]
    aEccentricity = point1[4] 

    bMass1 = point2[0]
    bMass2 = point2[1]
    bSpin1 = point2[2]
    bSpin2 = point2[3]
    bEccentricity = point2[4]

    aXis = get_cov_params_ecc(aMass1, aMass2, aSpin1, aSpin2, aEccentricity, metricParams, fUpper)

    bXis = get_cov_params_ecc(bMass1, bMass2, bSpin1, bSpin2, bEccentricity, metricParams, fUpper)

    dist = (aXis[0] - bXis[0])**2
    if number_coordinate == -1:
        for i in range(1,len(aXis)):
            dist += (aXis[i] - bXis[i])**2
    else:
        for i in range(1,number_coordinate):
            dist += (aXis[i] - bXis[i])**2
    return dist, aXis, bXis


def insert_eccentricity_range_options(parser):
    """
    Adds options to specify eccentricity ranges in the bank
    generation code to an argparser as an OptionGroup.
    """
    eccOpts = parser.add_argument_group("Options related to eccentricity "
                                        "limits for bank generation")
    eccOpts.add_argument("--min-ecc", action="store", type=positive_float, 
                         required=True,
                         help="Minimum value of eccentricity: must be positive and <0.1")
    eccOpts.add_argument("--max-ecc", action="store", type=positive_float,
                         required=True,
                         help="Maximum value of eccentricity: must be positive and <0.1")
    return eccOpts


def verify_eccentricity_options(opts, parser):
    """
    Parses the metric calculation options given and verifies that they are
    correct.

    Parameters
    ----------
    opts : argparse.Values instance
        Result of parsing the input options with OptionParser
    parser : object
        The OptionParser instance.
    """
    if opts.min_ecc > opts.max_ecc:
        parser.error("min-ecc cannot be greater than max-ecc.")
    if opts.max_ecc > 0.1:
        parser.error("max-ecc or min-ecc cannot be greater than 0.1")

def insert_ecc_options_in_metric_calculation_options(parser):
    """
    Adds the essential options for eccentricity options used to obtain a metric 
    in the bank generation codes to an argparser as an OptionGroup. 
    """
    metricOpts = insert_metric_calculation_options(parser)
    metricOpts.add_argument("--ecc-pn-order", action="store", type=str, 
                            required=True,
                            help="PN order for eccentric corrections."
                                 "Choices: %s"%(EccValidOrdersHelpDescriptions))
    metricOpts.add_argument("--f-ecc", action="store", type=positive_float,
                            default=20, help="Frequency at which the initial eccentricty"
                                             "is defined. Units=Hz") 

    return metricOpts
