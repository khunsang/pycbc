from pycbc.tmpltbank.lambda_mapping_eccentric import generate_mapping_ecc
from pycbc.tmpltbank.lambda_mapping import generate_mapping
from pycbc.tmpltbank.calc_moments import (calculate_moment, get_moments, interpolate_psd,
                                            calculate_metric_comp)
import numpy


def get_moments_ecc(metricParamsEcc, vary_fmax=False, vary_density=None):
    """
    This function will calculate the various integrals (moments) that are
    needed to compute the metric used in template bank placement and
    coincidence.

    Based on pycbc.tmpltbank.get_moments

    Parameters
    -----------
    metricParamsEcc : metricParametersEccentric instance
        Structure holding all the options for construction of the metric.
    vary_fmax : boolean, optional (default False)
        If set to False the metric and rotations are calculated once, for the
        full range of frequency [f_low,f_upper).
        If set to True the metric and rotations are calculated multiple times,
        for frequency ranges [f_low,f_low + i*vary_density), where i starts at
        1 and runs up until f_low + (i+1)*vary_density > f_upper.
        Thus values greater than f_upper are *not* computed.
        The calculation for the full range [f_low,f_upper) is also done.
    vary_density : float, optional
        If vary_fmax is True, this will be used in computing the frequency
        ranges as described for vary_fmax.

    Returns
    --------
    None : None
        **THIS FUNCTION RETURNS NOTHING**
        The following will be **added** to the metricParamsEcc structure
    metricParamsEcc.moments : Moments structure
        This contains the result of all the integrals used in computing the
        metrics above. It can be used for the ethinca components calculation,
        or other similar calculations. This is composed of two compound
        dictionaries. The first entry indicates which moment is being
        calculated and the second entry indicates the upper frequency cutoff
        that was used.

        In all cases x = f/f0.

        For the first entries the options are:

        moments['J%d' %(i)][f_cutoff]
        This stores the integral of
        x**((-i)/3.) * delta X / PSD(x)

        moments['log%d' %(i)][f_cutoff]
        This stores the integral of
        (numpy.log(x**(1./3.))) x**((-i)/3.) * delta X / PSD(x)

        moments['loglog%d' %(i)][f_cutoff]
        This stores the integral of
        (numpy.log(x**(1./3.)))**2 x**((-i)/3.) * delta X / PSD(x)


        The second entry stores the frequency cutoff used when computing
        the integral. See description of the vary_fmax option above.

        All of these values are nomralized by a factor of

        x**((-7)/3.) * delta X / PSD(x)

        The normalization factor can be obtained in

        moments['I7'][f_cutoff]
    """

    psd_amp = metricParamsEcc.psd.data
    psd_f = numpy.arange(len(psd_amp), dtype=float) * metricParamsEcc.deltaF
    new_f, new_amp = interpolate_psd(psd_f, psd_amp, metricParamsEcc.deltaF)

    # Need I7 first as this is the normalization factor
    funct = lambda x,f0: 1
    I7 = calculate_moment(new_f, new_amp, metricParamsEcc.fLow, \
                          metricParamsEcc.fUpper, metricParamsEcc.f0, funct,\
                          vary_fmax=vary_fmax, vary_density=vary_density)

    if metricParamsEcc._moments is None:
        get_moments(metricParamsEcc, vary_fmax=vary_fmax, vary_density=vary_density)
    # Do all the J moments
    moments = {}
    
    # J(q) = I(q)/I(7)
    # Power of x variable, n = (7-q)/3
    # For the ease of storing q is replace by a dummay key 3*q
    # THe for loop is over the the dummy key. However, the power of x are correctly
    # considered, i.e. q is divided by 3

    for i in range(-2,90):
        funct = lambda x,f0: x**((-i/3+7)/3.)
        moments['JEcc%d' %(i)] = calculate_moment(new_f, new_amp, \
                                metricParamsEcc.fLow, metricParamsEcc.fUpper, \
                                metricParamsEcc.f0, funct, norm=I7, \
                                vary_fmax=vary_fmax, vary_density=vary_density)

    # Do the logx multiplied by some power terms
    for i in range(16,90):
        funct = lambda x,f0: (numpy.log((x*f0)**(1./3.))) * x**((-i/3+7)/3.)
        moments['logEcc%d' %(i)] = calculate_moment(new_f, new_amp, \
                                metricParamsEcc.fLow, metricParamsEcc.fUpper, \
                                metricParamsEcc.f0, funct, norm=I7, \
                                vary_fmax=vary_fmax, vary_density=vary_density)

    # Do the loglog term
    for i in range(16,90):
        funct = lambda x,f0: (numpy.log((x*f0)**(1./3.)))**2 * x**((-i/3+7)/3.)
        moments['loglogEcc%d' %(i)] = calculate_moment(new_f, new_amp, \
                                metricParamsEcc.fLow, metricParamsEcc.fUpper, \
                                metricParamsEcc.f0, funct, norm=I7, \
                                vary_fmax=vary_fmax, vary_density=vary_density)

    metricParamsEcc.moments.update(moments)


def determine_eigen_directions_eccentric(metricParams, preserveMoments=False,
                               vary_fmax=False, vary_density=None):
    """
    This function will calculate the coordinate transfomations that are needed
    to rotate from a coordinate system described by the various Lambda
    components in the frequency expansion, to a coordinate system where the
    metric is Cartesian.

    Parameters
    -----------
    metricParams : metricParametersEccentric instance
        Structure holding all the options for construction of the metric.
    preserveMoments : boolean, optional (default False)
        Currently only used for debugging.
        If this is given then if the moments structure is already set
        within metricParams then they will not be recalculated.
    vary_fmax : boolean, optional (default False)
        If set to False the metric and rotations are calculated once, for the
        full range of frequency [f_low,f_upper).
        If set to True the metric and rotations are calculated multiple times,
        for frequency ranges [f_low,f_low + i*vary_density), where i starts at
        1 and runs up until f_low + (i+1)*vary_density > f_upper.
        Thus values greater than f_upper are *not* computed.
        The calculation for the full range [f_low,f_upper) is also done.
    vary_density : float, optional
        If vary_fmax is True, this will be used in computing the frequency
        ranges as described for vary_fmax.

    Returns
    --------
    metricParams : metricParametersEccentric instance
        Structure holding all the options for construction of the metric.
        **THIS FUNCTION ONLY RETURNS THE CLASS**
        The following will be **added** to this structure
    metricParams.evals : Dictionary of numpy.array
        Each entry in the dictionary corresponds to the different frequency
        ranges described in vary_fmax. If vary_fmax = False, the only entry
        will be f_upper, this corresponds to integrals in [f_low,f_upper). This
        entry is always present. Each other entry will use floats as keys to
        the dictionary. These floats give the upper frequency cutoff when it is
        varying.
        Each numpy.array contains the eigenvalues which, with the eigenvectors
        in evecs, are needed to rotate the
        coordinate system to one in which the metric is the identity matrix.
    metricParams.evecs : Dictionary of numpy.matrix
        Each entry in the dictionary is as described under evals.
        Each numpy.matrix contains the eigenvectors which, with the eigenvalues
        in evals, are needed to rotate the
        coordinate system to one in which the metric is the identity matrix.
    metricParams.metric : Dictionary of numpy.matrix
        Each entry in the dictionary is as described under evals.
        Each numpy.matrix contains the metric of the parameter space in the
        Lambda_i coordinate system.
    metricParams.moments : Moments structure
        See the structure documentation for a description of this. This
        contains the result of all the integrals used in computing the metrics
        above. It can be used for the ethinca components calculation, or other
        similar calculations.
    """

    evals = {}
    evecs = {}
    metric = {}
    unmax_metric = {}

    # First step is to get the moments needed to calculate the metric
    if not (metricParams.moments and preserveMoments):
        get_moments_ecc(metricParams, vary_fmax=False, vary_density=None)

    # What values are going to be in the moments
    # J7 is the normalization factor so it *MUST* be present
    list = metricParams.moments['J7'].keys()

    # We start looping over every item in the list of metrics
    for item in list:
        # Here we convert the moments into a form easier to use here
        Js = {}
        for i in range(-7,18):
            Js[i] = metricParams.moments['J%d'%(i)][item]

        logJs = {}
        for i in range(-1,18):
            logJs[i] = metricParams.moments['log%d'%(i)][item]

        loglogJs = {}
        for i in range(-1,18):
            loglogJs[i] = metricParams.moments['loglog%d'%(i)][item]

        logloglogJs = {}
        for i in range(-1,18):
            logloglogJs[i] = metricParams.moments['logloglog%d'%(i)][item]

        loglogloglogJs = {}
        for i in range(-1,18):
            loglogloglogJs[i] = metricParams.moments['loglogloglog%d'%(i)][item]
        
        JEccs = {}
        for i in range(-2,90):
            JEccs[i] = metricParams.moments['JEcc%d'%(i)][item]

        logJEccs = {}
        for i in range(16,90):
            logJEccs[i] = metricParams.moments['logEcc%d'%(i)][item]

        loglogJEccs = {}
        for i in range(16,90):
            loglogJEccs[i] = metricParams.moments['loglogEcc%d'%(i)][item]


        ecc_mapping = generate_mapping_ecc(metricParams.pnOrder, metricParams.eccpnOrder)
        qc_mapping = generate_mapping(metricParams.pnOrder)

        # Calculate the metric
        gs, unmax_metric_curr = calculate_metric_ecc(Js, logJs, loglogJs,
                                                     logloglogJs, loglogloglogJs,
                                                     JEccs, logJEccs, loglogJEccs,
                                                     qc_mapping, ecc_mapping)
        metric[item] = gs
        unmax_metric[item] = unmax_metric_curr

        # And the eigenvalues
        evals[item], evecs[item] = numpy.linalg.eig(gs)

        # Numerical error can lead to small negative eigenvalues.
        for i in range(len(evals[item])):
            if evals[item][i] < 0:
                # Due to numerical imprecision the very small eigenvalues can
                # be negative. Make these positive.
                evals[item][i] = -evals[item][i]
            if evecs[item][i,i] < 0:
                # We demand a convention that all diagonal terms in the matrix
                # of eigenvalues are positive.
                # This is done to help visualization of the spaces (increasing
                # mchirp always goes the same way)
                evecs[item][:,i] = - evecs[item][:,i]

    metricParams.evals = evals
    metricParams.evecs = evecs
    metricParams.metric = metric
    metricParams.time_unprojected_metric = unmax_metric

    return metricParams


def calculate_metric_ecc(Js, logJs, loglogJs, logloglogJs, loglogloglogJs, \
                        JEccs, logJEccs, loglogJEccs, \
                        qc_mapping, ecc_mapping):
    """
    This function will take the various integrals calculated by get_moments and
    convert this into a metric for the appropriate parameter space.

    Parameters
    -----------
    Js : Dictionary
        The list of (log^0 x) * x**(-i/3) integrals computed by get_moments()
        The index is Js[i]
    logJs : Dictionary
        The list of (log^1 x) * x**(-i/3) integrals computed by get_moments()
        The index is logJs[i]
    loglogJs : Dictionary
        The list of (log^2 x) * x**(-i/3) integrals computed by get_moments()
        The index is loglogJs[i]
    logloglogJs : Dictionary
        The list of (log^3 x) * x**(-i/3) integrals computed by get_moments()
        The index is logloglogJs[i]
    loglogloglogJs : Dictionary
        The list of (log^4 x) * x**(-i/3) integrals computed by get_moments()
        The index is loglogloglogJs[i]
    JEccs : Dictionary
        The list of (log^0 x) * x**(-i/9) integrals computed by get_moments_ecc()
        The index is JEccs[i]
    logJEccs : Dictionary
        The list of (log^1 x) * x**(-i/9) integrals computed by get_moments_ecc()
        The index is logJEccs[i]
    loglogJEccs : Dictionary
        The list of (log^2 x) * x**(-i/9) integrals computed by get_moments_ecc()
        The index is loglogJEccs[i]
    qc_mapping : dictionary
        Used to identify which Lambda components (quasi-circular phase only) are active in this parameter
        space and map these to entries in the metric matrix.
    ecc_mapping: dictionary
        Lambda components from both quasi-circular and eccentric phase. 

    Returns
    --------
    metric : numpy.matrix
        The resulting metric.
    """

    # How many dimensions in the parameter space?
    maxLen = len(ecc_mapping.keys())
    qcLen = len(qc_mapping.keys())

    metric = numpy.zeros(shape=(maxLen,maxLen), dtype=float)
    unmax_metric = numpy.zeros(shape=(maxLen+1,maxLen+1), dtype=float)

    #FIXME: +10 is a temporary fix
    for i in range(maxLen+10):
        for j in range(maxLen+10):
            if i < qcLen and j < qcLen:
                calculate_metric_comp(metric, unmax_metric, i, j, Js,
                                           logJs, loglogJs, logloglogJs,
                                           loglogloglogJs, qc_mapping)
            if j >= qcLen:
                calculate_metric_comp_ecc(metric, unmax_metric, i, j, Js, 
                                          logJs, JEccs, logJEccs, loglogJEccs, 
                                          qc_mapping, ecc_mapping)

    return metric, unmax_metric

def calculate_metric_comp_ecc(gs, unmax_metric, i, j, Js, logJs, Jeccs, logJeccs,
                              loglogJeccs, qc_mapping, ecc_mapping):
    """
    Computes part of eccentric and eccentric-noneccentric cross terms the metric.

    Only call this from within
    calculate_metric_ecc(). See the documentation for that function.
    """
    qc_len = len(qc_mapping.keys())
    
    if i >= qc_len and j >= qc_len:
        k = i - qc_len
        l = j - qc_len
        # Eccentric Normal terms
        if 'LambdaEcc%d'%k in ecc_mapping and 'LambdaEcc%d'%l in ecc_mapping:
            gammakl = Jeccs[89-3*k-3*l] - Jeccs[55-3*k]*Jeccs[55-3*l]
            gamma0k = (Jeccs[46-3*k] - Js[4]*Jeccs[55-3*k])
            gamma0l = (Jeccs[46-3*l] - Js[4]*Jeccs[55-3*l])
            gs[ecc_mapping['LambdaEcc%d'%k],ecc_mapping['LambdaEcc%d'%l]] = \
                0.5 * (gammakl - gamma0k*gamma0l/(Js[1] - Js[4]*Js[4]))
            unmax_metric[ecc_mapping['LambdaEcc%d'%k], -1] = gamma0k
            unmax_metric[-1, ecc_mapping['LambdaEcc%d'%l]] = gamma0l
            unmax_metric[ecc_mapping['LambdaEcc%d'%k],ecc_mapping['LambdaEcc%d'%l]] = gammakl
        # Eccentric Normal,Eccentric log cross terms
        if 'LambdaEcc%d'%k in ecc_mapping and 'LogLambdaEcc%d'%l in ecc_mapping:
            gammakl = logJeccs[89-3*k-3*l] - logJeccs[55-3*k] * Jeccs[55-3*l]
            gamma0k = (Jeccs[46-3*k] - Js[4] * Jeccs[55-3*k])
            gamma0l = logJeccs[46-3*l] - logJeccs[55-3*l] * Js[4]
            gs[ecc_mapping['LambdaEcc%d'%k],ecc_mapping['LogLambdaEcc%d'%l]] = \
                gs[ecc_mapping['LogLambdaEcc%d'%l],ecc_mapping['LambdaEcc%d'%k]] = \
                0.5 * (gammakl - gamma0k*gamma0l/(Js[1] - Js[4]*Js[4]))
            unmax_metric[ecc_mapping['LambdaEcc%d'%k], -1] = gamma0k
            unmax_metric[-1, ecc_mapping['LambdaEcc%d'%k]] = gamma0k
            unmax_metric[-1, ecc_mapping['LogLambdaEcc%d'%l]] = gamma0l
            unmax_metric[ecc_mapping['LogLambdaEcc%d'%l], -1] = gamma0l
            unmax_metric[ecc_mapping['LambdaEcc%d'%k],ecc_mapping['LogLambdaEcc%d'%l]] = gammakl
            unmax_metric[ecc_mapping['LogLambdaEcc%d'%l],ecc_mapping['LambdaEcc%d'%k]] = gammakl
        
        # Eccentric Log, Eccentric Log terms
        if 'LogLambdaEcc%d'%k in ecc_mapping and 'LogLambdaEcc%d'%l in ecc_mapping:
            gammakl = loglogJeccs[89-3*k-3*l] - logJeccs[55-3*k] * logJeccs[55-3*l]
            gamma0k = (logJeccs[46-3*k] - Js[4] * logJeccs[55-3*k])
            gamma0l = logJeccs[46-3*l] - logJeccs[55-3*l] * Js[4]
            gs[ecc_mapping['LogLambdaEcc%d'%k],ecc_mapping['LogLambdaEcc%d'%l]] = \
                    0.5 * (gammakl - gamma0k*gamma0l/(Js[1] - Js[4]*Js[4]))
            unmax_metric[ecc_mapping['LogLambdaEcc%d'%k], -1] = gamma0k
            unmax_metric[-1, ecc_mapping['LogLambdaEcc%d'%l]] = gamma0l
            unmax_metric[ecc_mapping['LogLambdaEcc%d'%k],ecc_mapping['LogLambdaEcc%d'%l]] =\
                    gammakl
    
    if j >= qc_len:
        l = j - qc_len
        # Quasi-circular Normal, Eccentric Normal cross terms
        if 'Lambda%d'%i in ecc_mapping and 'LambdaEcc%d'%l in ecc_mapping:
            gammail = Jeccs[70-3*i-3*l] - Jeccs[55-3*l] * Js[12-i]
            gamma0i = (Js[9-i] - Js[4] * Js[12-i])
            gamma0l = Jeccs[46-3*l] - Jeccs[55-3*l] * Js[4]
            gs[ecc_mapping['Lambda%d'%i],ecc_mapping['LambdaEcc%d'%l]] = \
                    gs[ecc_mapping['LambdaEcc%d'%l],ecc_mapping['Lambda%d'%i]] = \
                    0.5 * (gammail - gamma0i*gamma0l/(Js[1] - Js[4]*Js[4]))
            unmax_metric[ecc_mapping['Lambda%d'%i],ecc_mapping['LambdaEcc%d'%l]] = \
                gammail
            unmax_metric[ecc_mapping['LambdaEcc%d'%l],ecc_mapping['Lambda%d'%i]] = \
                gammail
            unmax_metric[ecc_mapping['Lambda%d'%i], -1] = gamma0i
            unmax_metric[-1, ecc_mapping['Lambda%d'%i]] = gamma0i
            unmax_metric[-1, ecc_mapping['LambdaEcc%d'%l]] = gamma0l
            unmax_metric[ecc_mapping['LambdaEcc%d'%l], -1] = gamma0l

        # Quasi-circular Normal, Eccentric Log cross terms
        if 'Lambda%d'%i in ecc_mapping and 'LogLambdaEcc%d'%l in ecc_mapping:
            gammail = logJeccs[70-3*i-3*l] - logJeccs[55-3*l] * Js[12-i]
            gamma0i = (Js[9-i] - Js[4] * Js[12-i])
            gamma0l = logJeccs[46-3*l] - logJeccs[55-3*l] * Js[4]
            gs[ecc_mapping['Lambda%d'%i],ecc_mapping['LogLambdaEcc%d'%l]] = \
                    gs[ecc_mapping['LogLambdaEcc%d'%l],ecc_mapping['Lambda%d'%i]] = \
                    0.5 * (gammail - gamma0i*gamma0l/(Js[1] - Js[4]*Js[4]))
            unmax_metric[ecc_mapping['Lambda%d'%i],ecc_mapping['LogLambdaEcc%d'%l]] = \
                gammail
            unmax_metric[ecc_mapping['LogLambdaEcc%d'%l],ecc_mapping['Lambda%d'%i]] = \
                gammail
            unmax_metric[ecc_mapping['Lambda%d'%i], -1] = gamma0i
            unmax_metric[-1, ecc_mapping['Lambda%d'%i]] = gamma0i
            unmax_metric[-1, ecc_mapping['LogLambdaEcc%d'%l]] = gamma0l
            unmax_metric[ecc_mapping['LogLambdaEcc%d'%l], -1] = gamma0l


        # Quasi-circular log, Eccentric Normal cross terms
        if 'LogLambda%d'%i in ecc_mapping and 'LambdaEcc%d'%l in ecc_mapping:
            gammail = logJeccs[70-3*i-3*l] - Jeccs[55-3*l] * logJs[12-i]
            gamma0i = (logJs[9-i] - Js[4] * logJs[12-i])
            gamma0l = Jeccs[46-3*l] - Jeccs[55-3*l] * Js[4]
            gs[ecc_mapping['LogLambda%d'%i],ecc_mapping['LambdaEcc%d'%l]] = \
                gs[ecc_mapping['LambdaEcc%d'%l],ecc_mapping['LogLambda%d'%i]] = \
                0.5 * (gammail - gamma0i*gamma0l/(Js[1] - Js[4]*Js[4]))
            unmax_metric[ecc_mapping['LogLambda%d'%i], -1] = gamma0i
            unmax_metric[-1, ecc_mapping['LogLambda%d'%i]] = gamma0i
            unmax_metric[-1, ecc_mapping['LambdaEcc%d'%l]] = gamma0l
            unmax_metric[ecc_mapping['LambdaEcc%d'%l], -1] = gamma0l
            unmax_metric[ecc_mapping['LogLambda%d'%i],ecc_mapping['LambdaEcc%d'%l]] = \
                gammail
            unmax_metric[ecc_mapping['LambdaEcc%d'%l],ecc_mapping['LogLambda%d'%i]] = \
                gammail

        # Quasi-circular Log, Eccentric Log cross terms
        if 'LogLambda%d'%i in ecc_mapping and 'LogLambdaEcc%d'%l in ecc_mapping:
            gammail = loglogJeccs[70-3*i-3*l] - logJeccs[55-3*l] * logJs[12-i]
            gamma0i = (logJs[9-i] - Js[4] * logJs[12-i])
            gamma0l = logJeccs[46-3*l] - logJeccs[55-3*l] * Js[4]
            gs[ecc_mapping['LogLambda%d'%i],ecc_mapping['LogLambdaEcc%d'%l]] = \
                    0.5 * (gammail - gamma0i*gamma0l/(Js[1] - Js[4]*Js[4]))
            unmax_metric[ecc_mapping['LogLambda%d'%i], -1] = gamma0i
            unmax_metric[-1, ecc_mapping['LogLambdaEcc%d'%l]] = gamma0l
            unmax_metric[ecc_mapping['LogLambda%d'%i],ecc_mapping['LogLambdaEcc%d'%l]] =\
                gammail
            unmax_metric[ecc_mapping['LogLambdaEcc%d'%l],ecc_mapping['LogLambda%d'%i]] =\
                gammail
