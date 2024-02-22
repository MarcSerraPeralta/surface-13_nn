import matplotlib.pyplot as plt
import numpy as np
import os
import datetime


def _rotate_and_center_data(I, Q, vec0, vec1, phi=0):
    """
    Rotate <I>, <Q> shots in IQ plane around axis defined by <vec0> - <vec1>
    """
    vector = vec1 - vec0
    angle = np.arctan(vector[1] / vector[0])
    rot_matrix = np.array(
        [
            [np.cos(-angle + phi), -np.sin(-angle + phi)],
            [np.sin(-angle + phi), np.cos(-angle + phi)],
        ]
    )
    proc = np.array((I, Q))
    proc = np.dot(rot_matrix, proc)
    return proc.transpose()


def _decision_boundary_points(coefs, intercepts):
    """
    Find points along the decision boundaries of
    LinearDiscriminantAnalysis (LDA).
    This is performed by finding the interception
    of the bounds of LDA. For LDA, these bounds are
    encoded in the coef_ and intercept_ parameters
    of the classifier.
    Each bound <i> is given by the equation:
    y + coef_i[0]/coef_i[1]*x + intercept_i = 0
    Note this only works for LinearDiscriminantAnalysis.
    Other classifiers might have diferent bound models.
    """
    points = {}
    # Cycle through model coeficientsand intercepts.
    # 2-state classifier
    if len(intercepts) == 1:
        m = -coefs[0][0] / coefs[0][1]
        _X = np.array([-10, 10])
        _Y = m * _X - intercepts[0] / coefs[0][1]
        points["left"] = np.array([_X[0], _Y[0]])
        points["right"] = np.array([_X[1], _Y[1]])
    # 3-state classifier
    elif len(intercepts) == 3:
        for i, j in [[0, 1], [1, 2], [0, 2]]:
            c_i = coefs[i]
            int_i = intercepts[i]
            c_j = coefs[j]
            int_j = intercepts[j]
            x = (-int_j / c_j[1] + int_i / c_i[1]) / (
                -c_i[0] / c_i[1] + c_j[0] / c_j[1]
            )
            y = -c_i[0] / c_i[1] * x - int_i / c_i[1]
            points[f"{i}{j}"] = (x, y)
        # Find mean point
        points["mean"] = np.mean([[x, y] for (x, y) in points.values()], axis=0)

    return points


def _Classify_qubit_calibration_shots(Shots_0, Shots_1):
    """
    Train linear discriminant classifier
    to classify Qubit shots in IQ space.
    """
    data = np.concatenate((Shots_0, Shots_1))
    labels = [0 for s in Shots_0] + [1 for s in Shots_1]
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    clf = LinearDiscriminantAnalysis()
    clf.fit(data, labels)
    dec_bounds = _decision_boundary_points(clf.coef_, clf.intercept_)
    # dec_bounds = _decision_boundary_points(clf.coef_, clf.intercept_)
    Fid_dict = {}
    for state, shots in zip(["0", "1"], [Shots_0, Shots_1]):
        _res = clf.predict(shots)
        _fid = np.mean(_res == int(state))
        Fid_dict[state] = _fid
    Fid_dict["avg"] = np.mean([f for f in Fid_dict.values()])
    # Get assignment fidelity matrix
    M = np.zeros((2, 2))
    for i, shots in enumerate([Shots_0, Shots_1]):
        for j, state in enumerate(["0", "1"]):
            _res = clf.predict(shots)
            M[i][j] = np.mean(_res == int(state))
    return clf, Fid_dict, M, dec_bounds


def _Classify_qutrit_calibration_shots(Shots_0, Shots_1, Shots_2):
    """
    Train linear discriminant classifier
    to classify Qutrit shots in IQ space.
    """
    data = np.concatenate((Shots_0, Shots_1, Shots_2))
    labels = [0 for s in Shots_0] + [1 for s in Shots_1] + [2 for s in Shots_2]
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    clf = LinearDiscriminantAnalysis()
    clf.fit(data, labels)
    dec_bounds = _decision_boundary_points(clf.coef_, clf.intercept_)
    Fid_dict = {}
    for state, shots in zip(["0", "1", "2"], [Shots_0, Shots_1, Shots_2]):
        _res = clf.predict(shots)
        _fid = np.mean(_res == int(state))
        Fid_dict[state] = _fid
    Fid_dict["avg"] = np.mean([f for f in Fid_dict.values()])
    # Get assignment fidelity matrix
    M = np.zeros((3, 3))
    for i, shots in enumerate([Shots_0, Shots_1, Shots_2]):
        for j, state in enumerate(["0", "1", "2"]):
            _res = clf.predict(shots)
            M[i][j] = np.mean(_res == int(state))
    return clf, Fid_dict, M, dec_bounds


def _calculate_fid_and_threshold(x0, n0, x1, n1):
    """
    Calculate fidelity and threshold from histogram data:
    x0, n0 is the histogram data of shots 0 (value and occurences),
    x1, n1 is the histogram data of shots 1 (value and occurences).
    """
    # Build cumulative histograms of shots 0
    # and 1 in common bins by interpolation.
    all_x = np.unique(np.sort(np.concatenate((x0, x1))))
    cumsum0, cumsum1 = np.cumsum(n0), np.cumsum(n1)
    ecumsum0 = np.interp(x=all_x, xp=x0, fp=cumsum0, left=0)
    necumsum0 = ecumsum0 / np.max(ecumsum0)
    ecumsum1 = np.interp(x=all_x, xp=x1, fp=cumsum1, left=0)
    necumsum1 = ecumsum1 / np.max(ecumsum1)
    # Calculate optimal threshold and fidelity
    F_vs_th = 1 - (1 - abs(necumsum0 - necumsum1)) / 2
    opt_idxs = np.argwhere(F_vs_th == np.amax(F_vs_th))
    opt_idx = int(round(np.average(opt_idxs)))
    F_assignment_raw = F_vs_th[opt_idx]
    threshold_raw = all_x[opt_idx]
    return F_assignment_raw, threshold_raw


def _get_threshold(Shots_0, Shots_1):
    # Take relavant quadrature
    shots_0 = Shots_0[:, 0]
    shots_1 = Shots_1[:, 0]
    # find range
    _all_shots = np.concatenate((shots_0, shots_1))
    _range = (np.min(_all_shots), np.max(_all_shots))
    # Sort shots in unique values
    x0, n0 = np.unique(shots_0, return_counts=True)
    x1, n1 = np.unique(shots_1, return_counts=True)
    Fid, threshold = _calculate_fid_and_threshold(x0, n0, x1, n1)
    return threshold


def _gauss_pdf(x, x0, sigma):
    return np.exp(-(((x - x0) / sigma) ** 2) / 2)


def double_gauss(x, x0, x1, sigma0, sigma1, A, r):
    _dist0 = A * ((1 - r) * _gauss_pdf(x, x0, sigma0) + r * _gauss_pdf(x, x1, sigma1))
    return _dist0


def _double_gauss_joint(x, x0, x1, sigma0, sigma1, A0, A1, r0, r1):
    _dist0 = double_gauss(x, x0, x1, sigma0, sigma1, A0, r0)
    _dist1 = double_gauss(x, x1, x0, sigma1, sigma0, A1, r1)
    return np.concatenate((_dist0, _dist1))


def _fit_double_gauss(x_vals, hist_0, hist_1):
    """
    Fit two histograms to a double gaussian with
    common parameters. From fitted parameters,
    calculate SNR, Pe0, Pg1, Teff, Ffit and Fdiscr.
    """
    from scipy.optimize import curve_fit

    # Double gaussian model for fitting
    def _gauss_pdf(x, x0, sigma):
        return np.exp(-(((x - x0) / sigma) ** 2) / 2)

    global double_gauss

    def double_gauss(x, x0, x1, sigma0, sigma1, A, r):
        _dist0 = A * (
            (1 - r) * _gauss_pdf(x, x0, sigma0) + r * _gauss_pdf(x, x1, sigma1)
        )
        return _dist0

    # helper function to simultaneously fit both histograms with common parameters
    def _double_gauss_joint(x, x0, x1, sigma0, sigma1, A0, A1, r0, r1):
        _dist0 = double_gauss(x, x0, x1, sigma0, sigma1, A0, r0)
        _dist1 = double_gauss(x, x1, x0, sigma1, sigma0, A1, r1)
        return np.concatenate((_dist0, _dist1))

    # Guess for fit
    pdf_0 = hist_0 / np.sum(hist_0)  # Get prob. distribution
    pdf_1 = hist_1 / np.sum(hist_1)  #
    _x0_guess = np.sum(x_vals * pdf_0)  # calculate mean
    _x1_guess = np.sum(x_vals * pdf_1)  #
    _sigma0_guess = np.sqrt(np.sum((x_vals - _x0_guess) ** 2 * pdf_0))  # calculate std
    _sigma1_guess = np.sqrt(np.sum((x_vals - _x1_guess) ** 2 * pdf_1))  #
    _r0_guess = 0.01
    _r1_guess = 0.05
    _A0_guess = np.max(hist_0)
    _A1_guess = np.max(hist_1)
    p0 = [
        _x0_guess,
        _x1_guess,
        _sigma0_guess,
        _sigma1_guess,
        _A0_guess,
        _A1_guess,
        _r0_guess,
        _r1_guess,
    ]
    # Bounding parameters
    _x0_bound = (-np.inf, np.inf)
    _x1_bound = (-np.inf, np.inf)
    _sigma0_bound = (0, np.inf)
    _sigma1_bound = (0, np.inf)
    _r0_bound = (0, 1)
    _r1_bound = (0, 1)
    _A0_bound = (0, np.inf)
    _A1_bound = (0, np.inf)
    bounds = np.array(
        [
            _x0_bound,
            _x1_bound,
            _sigma0_bound,
            _sigma1_bound,
            _A0_bound,
            _A1_bound,
            _r0_bound,
            _r1_bound,
        ]
    )
    # Fit parameters within bounds
    popt, pcov = curve_fit(
        _double_gauss_joint,
        x_vals,
        np.concatenate((hist_0, hist_1)),
        p0=p0,
        bounds=bounds.transpose(),
    )
    popt0 = popt[[0, 1, 2, 3, 4, 6]]
    popt1 = popt[[1, 0, 3, 2, 5, 7]]
    # Calculate quantities of interest
    SNR = abs(popt0[0] - popt1[0]) / ((abs(popt0[2]) + abs(popt1[2])) / 2)
    P_e0 = popt0[5]
    P_g1 = popt1[5]
    # Fidelity from fit
    _range = (np.min(x_vals), np.max(x_vals))
    _x_data = np.linspace(*_range, 10001)
    _h0 = double_gauss(_x_data, *popt0)  # compute distrubition from
    _h1 = double_gauss(_x_data, *popt1)  # fitted parameters.
    Fid_fit, threshold_fit = _calculate_fid_and_threshold(_x_data, _h0, _x_data, _h1)
    # Discrimination fidelity
    _h0 = double_gauss(_x_data, *popt0[:-1], 0)  # compute distrubition without residual
    _h1 = double_gauss(_x_data, *popt1[:-1], 0)  # excitation of relaxation.
    Fid_discr, threshold_discr = _calculate_fid_and_threshold(
        _x_data, _h0, _x_data, _h1
    )
    # return results
    qoi = {
        "SNR": SNR,
        "P_e0": P_e0,
        "P_g1": P_g1,
        "Fid_fit": Fid_fit,
        "Fid_discr": Fid_discr,
    }
    return popt0, popt1, qoi


def _Analyse_qubit_shots_along_decision_boundaries(
    qubit, Shots_0, Shots_1, dec_bounds, proc_data_dict
):
    """
    Project readout data along axis perpendicular
    to the decision boundaries of classifier and computes
    quantities of interest. These are saved in the <proc_data_dict>.
    """
    ############################
    # Projection along 01 axis.
    ############################
    # Rotate shots over 01 axis
    shots_0 = _rotate_and_center_data(
        Shots_0[:, 0],
        Shots_0[:, 1],
        dec_bounds["left"],
        dec_bounds["right"],
        phi=np.pi / 2,
    )
    shots_1 = _rotate_and_center_data(
        Shots_1[:, 0],
        Shots_1[:, 1],
        dec_bounds["left"],
        dec_bounds["right"],
        phi=np.pi / 2,
    )
    # Take relavant quadrature
    shots_0 = shots_0[:, 0]
    shots_1 = shots_1[:, 0]
    n_shots_1 = len(shots_1)
    # find range
    _all_shots = np.concatenate((shots_0, shots_1))
    _range = (np.min(_all_shots), np.max(_all_shots))
    # Sort shots in unique values
    x0, n0 = np.unique(shots_0, return_counts=True)
    x1, n1 = np.unique(shots_1, return_counts=True)
    Fid_01, threshold_01 = _calculate_fid_and_threshold(x0, n0, x1, n1)
    # Histogram of shots for 1 and 2
    h0, bin_edges = np.histogram(shots_0, bins=100, range=_range)
    h1, bin_edges = np.histogram(shots_1, bins=100, range=_range)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    popt0, popt1, params_01 = _fit_double_gauss(bin_centers, h0, h1)
    # Save processed data
    proc_data_dict[qubit]["projection_qubit"] = {}
    proc_data_dict[qubit]["projection_qubit"]["h0"] = h0
    proc_data_dict[qubit]["projection_qubit"]["h1"] = h1
    proc_data_dict[qubit]["projection_qubit"]["bin_centers"] = bin_centers
    proc_data_dict[qubit]["projection_qubit"]["popt0"] = popt0
    proc_data_dict[qubit]["projection_qubit"]["popt1"] = popt1
    proc_data_dict[qubit]["projection_qubit"]["SNR"] = params_01["SNR"]
    proc_data_dict[qubit]["projection_qubit"]["Fid"] = Fid_01
    proc_data_dict[qubit]["projection_qubit"]["threshold"] = threshold_01


def _Analyse_qutrit_shots_along_decision_boundaries(
    qubit, Shots_0, Shots_1, Shots_2, dec_bounds, proc_data_dict
):
    """
    Project readout data along axis perpendicular
    to the decision boundaries of classifier and computes
    quantities of interest. These are saved in the <proc_data_dict>.
    """
    ############################
    # Projection along 01 axis.
    ############################
    # Rotate shots over 01 axis
    shots_0 = _rotate_and_center_data(
        Shots_0[:, 0],
        Shots_0[:, 1],
        dec_bounds["mean"],
        dec_bounds["01"],
        phi=np.pi / 2,
    )
    shots_1 = _rotate_and_center_data(
        Shots_1[:, 0],
        Shots_1[:, 1],
        dec_bounds["mean"],
        dec_bounds["01"],
        phi=np.pi / 2,
    )
    # Take relavant quadrature
    shots_0 = shots_0[:, 0]
    shots_1 = shots_1[:, 0]
    n_shots_1 = len(shots_1)
    # find range
    _all_shots = np.concatenate((shots_0, shots_1))
    _range = (np.min(_all_shots), np.max(_all_shots))
    # Sort shots in unique values
    x0, n0 = np.unique(shots_0, return_counts=True)
    x1, n1 = np.unique(shots_1, return_counts=True)
    Fid_01, threshold_01 = _calculate_fid_and_threshold(x0, n0, x1, n1)
    # Histogram of shots for 1 and 2
    h0, bin_edges = np.histogram(shots_0, bins=100, range=_range)
    h1, bin_edges = np.histogram(shots_1, bins=100, range=_range)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    popt0, popt1, params_01 = _fit_double_gauss(bin_centers, h0, h1)
    # Save processed data
    proc_data_dict[qubit]["projection_01"] = {}
    proc_data_dict[qubit]["projection_01"]["h0"] = h0
    proc_data_dict[qubit]["projection_01"]["h1"] = h1
    proc_data_dict[qubit]["projection_01"]["bin_centers"] = bin_centers
    proc_data_dict[qubit]["projection_01"]["popt0"] = popt0
    proc_data_dict[qubit]["projection_01"]["popt1"] = popt1
    proc_data_dict[qubit]["projection_01"]["SNR"] = params_01["SNR"]
    proc_data_dict[qubit]["projection_01"]["Fid"] = Fid_01
    proc_data_dict[qubit]["projection_01"]["threshold"] = threshold_01
    ############################
    # Projection along 12 axis.
    ############################
    # Rotate shots over 12 axis
    shots_1 = _rotate_and_center_data(
        Shots_1[:, 0],
        Shots_1[:, 1],
        dec_bounds["mean"],
        dec_bounds["12"],
        phi=np.pi / 2,
    )
    shots_2 = _rotate_and_center_data(
        Shots_2[:, 0],
        Shots_2[:, 1],
        dec_bounds["mean"],
        dec_bounds["12"],
        phi=np.pi / 2,
    )
    # Take relavant quadrature
    shots_1 = shots_1[:, 0]
    shots_2 = shots_2[:, 0]
    n_shots_2 = len(shots_2)
    # find range
    _all_shots = np.concatenate((shots_1, shots_2))
    _range = (np.min(_all_shots), np.max(_all_shots))
    # Sort shots in unique values
    x1, n1 = np.unique(shots_1, return_counts=True)
    x2, n2 = np.unique(shots_2, return_counts=True)
    Fid_12, threshold_12 = _calculate_fid_and_threshold(x1, n1, x2, n2)
    # Histogram of shots for 1 and 2
    h1, bin_edges = np.histogram(shots_1, bins=100, range=_range)
    h2, bin_edges = np.histogram(shots_2, bins=100, range=_range)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    popt1, popt2, params_12 = _fit_double_gauss(bin_centers, h1, h2)
    # Save processed data
    proc_data_dict[qubit]["projection_12"] = {}
    proc_data_dict[qubit]["projection_12"]["h1"] = h1
    proc_data_dict[qubit]["projection_12"]["h2"] = h2
    proc_data_dict[qubit]["projection_12"]["bin_centers"] = bin_centers
    proc_data_dict[qubit]["projection_12"]["popt1"] = popt1
    proc_data_dict[qubit]["projection_12"]["popt2"] = popt2
    proc_data_dict[qubit]["projection_12"]["SNR"] = params_12["SNR"]
    proc_data_dict[qubit]["projection_12"]["Fid"] = Fid_12
    proc_data_dict[qubit]["projection_12"]["threshold"] = threshold_12
    ############################
    # Projection along 02 axis.
    ############################
    # Rotate shots over 02 axis
    shots_0 = _rotate_and_center_data(
        Shots_0[:, 0],
        Shots_0[:, 1],
        dec_bounds["mean"],
        dec_bounds["02"],
        phi=np.pi / 2,
    )
    shots_2 = _rotate_and_center_data(
        Shots_2[:, 0],
        Shots_2[:, 1],
        dec_bounds["mean"],
        dec_bounds["02"],
        phi=np.pi / 2,
    )
    # Take relavant quadrature
    shots_0 = shots_0[:, 0]
    shots_2 = shots_2[:, 0]
    n_shots_2 = len(shots_2)
    # find range
    _all_shots = np.concatenate((shots_0, shots_2))
    _range = (np.min(_all_shots), np.max(_all_shots))
    # Sort shots in unique values
    x0, n0 = np.unique(shots_0, return_counts=True)
    x2, n2 = np.unique(shots_2, return_counts=True)
    Fid_02, threshold_02 = _calculate_fid_and_threshold(x0, n0, x2, n2)
    # Histogram of shots for 1 and 2
    h0, bin_edges = np.histogram(shots_0, bins=100, range=_range)
    h2, bin_edges = np.histogram(shots_2, bins=100, range=_range)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    popt0, popt2, params_02 = _fit_double_gauss(bin_centers, h0, h2)
    # Save processed data
    proc_data_dict[qubit]["projection_02"] = {}
    proc_data_dict[qubit]["projection_02"]["h0"] = h0
    proc_data_dict[qubit]["projection_02"]["h2"] = h2
    proc_data_dict[qubit]["projection_02"]["bin_centers"] = bin_centers
    proc_data_dict[qubit]["projection_02"]["popt0"] = popt0
    proc_data_dict[qubit]["projection_02"]["popt2"] = popt2
    proc_data_dict[qubit]["projection_02"]["SNR"] = params_02["SNR"]
    proc_data_dict[qubit]["projection_02"]["Fid"] = Fid_02
    proc_data_dict[qubit]["projection_02"]["threshold"] = threshold_02


def _get_nearest_neighbors(qubit, map_qubits=None):
    """
    Helper function to determine nearest neighbors of a qubit.
    Default map is surface-17, however other maps are supported.
    """
    if map_qubits == None:
        # Surface-17 layout
        map_qubits = {
            "Z3": [-2, -1],
            "D9": [0, 2],
            "X4": [-1, 2],
            "D8": [-1, 1],
            "Z4": [0, 1],
            "D6": [1, 1],
            "D7": [-2, 0],
            "X3": [-1, 0],
            "D5": [0, 0],
            "X2": [1, 0],
            "D3": [2, 0],
            "D4": [-1, -1],
            "Z1": [0, -1],
            "D2": [1, -1],
            "X1": [1, -2],
            "Z2": [2, 1],
            "D1": [0, -2],
        }
    Neighbor_dict = {}
    Qubits = list(map_qubits.keys())
    Qubits.remove(qubit)
    for q in Qubits:
        V0 = np.array(map_qubits[qubit])  # qubit position
        V1 = np.array(map_qubits[q])
        diff = V1 - V0
        dist = np.sqrt(np.sum((diff) ** 2))
        if any(diff) == 0.0:
            pass
        elif diff[0] == 0.0:
            if diff[1] == 1.0:
                Neighbor_dict[q] = "SW"
            elif diff[1] == -1.0:
                Neighbor_dict[q] = "NE"
        elif diff[1] == 0.0:
            if diff[0] == 1.0:
                Neighbor_dict[q] = "NW"
            elif diff[0] == -1.0:
                Neighbor_dict[q] = "SE"
    return Neighbor_dict


def _calculate_defects(Shots, n_rounds, Data_qubit_meas):
    """
    Shots must be a dictionary with format:
                                  |<---nr_shots--->|
    Shots['round <i>'] = np.array([0/1,......., 0/1])

    Returns defect values in +1/-1 (where -1 corresponds to defect).
    """
    Deffect_rate = {}
    nr_shots = len(Shots["round 1"])
    # M array is measured data
    # P array is parity data
    # D array is defect data
    M_values = np.ones((nr_shots, n_rounds))
    for r in range(n_rounds):
        # Signal leakge events
        _Shots = np.array([s if s != 2 else np.nan for s in Shots[f"round {r+1}"]])
        # Convert to +1 and -1 values
        M_values[:, r] *= 1 - 2 * (_Shots)
    # Append +1 Pauli frame in first round
    P_values = np.hstack((np.ones((nr_shots, 2)), M_values))
    P_values = P_values[:, 1:] * P_values[:, :-1]
    # Compute parity from data-qubit readout
    _final_parity = np.ones((nr_shots, 1))
    for _Data_shots in Data_qubit_meas:
        # convert to +1 and -1 values and reshape
        _Data_shots = 1 - 2 * _Data_shots.reshape(nr_shots, 1)
        _final_parity *= _Data_shots
    # Append to ancilla measured parities
    P_values = np.hstack((P_values, _final_parity))
    # Second derivative of parity to get defects
    D_values = P_values[:, 1:] * P_values[:, :-1]
    return D_values


def _array_to_binary_string(array):
    """
    Converts square array of integers in
    same shape array of binary string.
    """
    n_i, n_j = array.shape
    new_array = np.ones((n_i, n_j), dtype="<U4")
    for i in range(n_i):
        for j in range(n_j):
            if np.isnan(array[i, j]):
                new_array[i, j] = np.nan
            else:
                new_array[i, j] = "{:04b}".format(int(array[i, j]))
    return new_array


def _syndrome_to_pauli_frame_correction(array, Decoder_LUT):
    """
    Converts stabilizer syndrome
    to logical operator correction.
    """
    # Add entry in dictionary for post-selection
    Decoder_LUT["nan"] = np.nan
    # Compute PFU
    n_i, n_j = array.shape
    new_array = np.ones((n_i, n_j))
    for i in range(n_i):
        for j in range(n_j):
            new_array[i, j] = Decoder_LUT[array[i, j]]
    return new_array


def _get_decoding_frames(Defects):
    """
    Convert defects in qubits to syndrome frames
    """
    # Get Rounds in experiment
    _aux = list(Defects.keys())[0]
    _Rounds = [int(k.split("_")[0]) for k in Defects[_aux][0].keys()]
    # Assemble deconding frames in each round
    Decoding_frames = [None for i in range(len(Defects[_aux]))]
    for k in range(len(Defects[_aux])):
        Decoding_frames[k] = {f"{r}_R": None for r in _Rounds}
        for n_rounds in _Rounds:
            # Express defects in 0/1 representation (where 1 = defect)
            _D_Z1_bin = (1 - Defects["Z1"][k][f"{n_rounds}_R"]) / 2
            _D_Z2_bin = (1 - Defects["Z2"][k][f"{n_rounds}_R"]) / 2
            _D_Z3_bin = (1 - Defects["Z3"][k][f"{n_rounds}_R"]) / 2
            _D_Z4_bin = (1 - Defects["Z4"][k][f"{n_rounds}_R"]) / 2
            # Convert full stabilizer syndrome to binary base2 number
            _Frames = (
                _D_Z1_bin * 2**3
                + _D_Z2_bin * 2**2
                + _D_Z3_bin * 2**1
                + _D_Z4_bin * 2**0
            )
            # Express it binary strings (eg: '0101')
            _Frames = _array_to_binary_string(_Frames)
            Decoding_frames[k][f"{n_rounds}_R"] = _Frames
    return Decoding_frames


def _calculate_logical_outcomes(Shots, Qubits):
    """
    Compute logical outcomes from qubit shots
    """
    # Get Rounds in experiment
    _aux = list(Shots[0].keys())[0]
    _Rounds = [int(k.split("_")[0]) for k in Shots[0][_aux].keys()]
    # Compute logical outcomes
    Logical_outcomes = [None for k in Shots.keys()]
    for k in range(len(Shots.keys())):
        Logical_outcomes[k] = {f"{r}_R": None for r in _Rounds}
        for n_rounds in _Rounds:
            # Compute Logical outcomes
            n_shots = len(Shots[k][Qubits[0]][f"{n_rounds}_R"][f"round {n_rounds}"])
            Logical_outcomes[k][f"{n_rounds}_R"] = np.ones((n_shots))
            for q in Qubits:
                Logical_outcomes[k][f"{n_rounds}_R"] *= (
                    1 - 2 * Shots[k][q][f"{n_rounds}_R"][f"round {n_rounds}"]
                )
    return Logical_outcomes


def _decode_space_biased(Dec_frames, Logical_out, Decoder_LUT):
    """
    Takes syndrome information and measured Logical outcomes
    and decodes errors using a decoder LUT to obtain
    a mean logical outcome.
    """
    n_kernels = len(Logical_out)
    _Rounds = [int(k.split("_")[0]) for k in Logical_out[0].keys()]
    # Compute mean logical outcomes after corrections
    Mean_logical_out = [np.ones(len(_Rounds)) for i in range(n_kernels)]
    for k in range(n_kernels):
        for i, n_rounds in enumerate(_Rounds):
            _frames = Dec_frames[k][f"{n_rounds}_R"]
            # get Pauli frame updates
            _corr = _syndrome_to_pauli_frame_correction(_frames, Decoder_LUT)
            # multiply all PFUs
            _corr = np.prod(_corr, axis=1)
            _Logical_out = Logical_out[k][f"{n_rounds}_R"]
            # Apply PFUs to measured logical outcomes
            Mean_logical_out[k][i] = np.nanmean(_Logical_out * _corr)
    return _Rounds, Mean_logical_out


# helper functions
Decoder_LUT = {
    "0000": +1,  # No error
    "0001": -1,  # error on D8 or D9
    "0010": -1,  # error on D7
    "0100": -1,  # error on D3
    "1000": -1,  # error on D1 or D2
    "1001": -1,  # error on D5
    "1010": -1,  # error on D4
    "1100": +1,  # double error
    "0101": -1,  # error on D6
    "0110": +1,  # double error
    "0011": +1,  # double error
    "0111": +1,  # double error
    "1011": +1,  # double error
    "1101": +1,  # double error
    "1110": +1,  # double error
    "1111": +1,  # double error
}


def _QED_ps(rounds, shots, ancilla_qubits, data_qubits):
    """
    performs postselection on errors for quantum error detection
    shots: digitized shots from ma2.Repeated_msmt_anaylsis
    Rounds: e.g. 10
    """
    nr_shots = len(shots[ancilla_qubits[0]][f"{rounds}_R"][f"round {rounds}"])
    # for qubit in data_qubits+ancilla_qubits:
    #     for r in range(1,rounds+1):
    #         shots[qubit][f'{rounds}_R'][f'round {r}'] = (1-2*shots[qubit][f'{rounds}_R'][f'round {r}'])

    for qb in ancilla_qubits:
        # create shot round+1 to add the product of the final data qubit msmt
        shots[qb][f"{rounds}_R"][f"round {rounds+1}"] = np.ones(nr_shots)
    Shots_anci_aux = np.ones(nr_shots)
    Shots_post_selected = {name: np.ones(nr_shots) for name in data_qubits}

    for qb in ancilla_qubits:
        for q in _get_nearest_neighbors(qb).keys():
            shots[qb][f"{rounds}_R"][f"round {rounds+1}"] *= shots[q][f"{rounds}_R"][
                f"round {rounds}"
            ]
        # Post-select shots
        for r in range(1, rounds + 2):
            # shots here must in +1/-1 that is why i have  (1-2*shots) as shots are in 0/1
            Shots_anci_aux *= np.array(
                [
                    +1 if shot == +1 else np.nan
                    for shot in shots[qb][f"{rounds}_R"][f"round {r}"]
                ]
            )
        # PS_fraction_Z makes sense, thus i think Shots_anci_aux makes sense
        PS_fraction_Z = np.sum(~np.isnan(Shots_anci_aux)) / nr_shots
    # post select out all shots where it is nan. same here for the shots
    for name in data_qubits:
        Shots_post_selected[name] = [
            s
            for s in shots[name][f"{rounds}_R"][f"round {rounds}"] * Shots_anci_aux
            if ~np.isnan(s)
        ]
        Logical_operator_post_sel = np.ones(len(Shots_post_selected[name]))
    # compute Logical operator with the postselected shots
    for name in data_qubits:
        Logical_operator_post_sel *= np.array(Shots_post_selected[name])
    # print(f'round {r} PS fraction is {PS_fraction_Z}')
    # print(f'round {r} Log QED is {np.mean(Logical_operator_post_sel)}')
    return Logical_operator_post_sel, PS_fraction_Z


def _pauli_frame_update(s_values, rounds, ancilla_qubits):
    """
    This function extracts pauli frame update from qubit shots per experiment (n_rounds)
    """
    syndromes = sum(
        [
            s_values[qb] * 2 ** (len(ancilla_qubits) - 1 - n)
            for n, qb in enumerate(ancilla_qubits)
        ]
    )
    _nr_shots = s_values[ancilla_qubits[0]].shape[0]
    Pauli_frames = np.ones(_nr_shots, dtype=int)
    for i in range(_nr_shots):
        for r in range(rounds + 1):
            if str(syndromes[i, r]) != "nan":
                Pauli_frames[i] *= Decoder_LUT["{0:04b}".format(int(syndromes[i, r]))]
    return Pauli_frames


def _compute_logical_operator_shots(data_qubits, shots_per_exp, rounds):
    """
    Compute final data qubit stabilizer measurement
    """
    nr_shots = len(
        shots_per_exp[data_qubits[0]][f"{rounds}_R"][f"round {rounds}"][
            ~np.isnan(shots_per_exp[data_qubits[0]][f"{rounds}_R"][f"round {rounds}"])
        ]
    )
    # Calculate Logical operator expectation value (we use the operator X1...X9)
    Logical_operator = np.ones(nr_shots)
    for name in data_qubits:
        Logical_operator *= (
            1
            - 2
            * shots_per_exp[name][f"{rounds}_R"][f"round {rounds}"][
                ~np.isnan(
                    shots_per_exp[data_qubits[0]][f"{rounds}_R"][f"round {rounds}"]
                )
            ]
        )
    return Logical_operator


def Analyze_calibration_shots(
    raw_data_dict,
    Rounds,
    n_kernels,
    heralded_init,
):
    """
    Analyze calibration points in raw data.
    """
    Qubits = [
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
        "D6",
        "D7",
        "D8",
        "D9",
        "Z1",
        "Z2",
        "Z3",
        "Z4",
    ]
    _total_rounds = np.sum(Rounds)
    # Add heralding measurement for experimental shots
    if heralded_init:
        _total_rounds += len(Rounds)
    _cycle = _total_rounds * n_kernels + 3
    # Add heralding measurement for calibration points
    if heralded_init:
        _cycle += 3
    ##################################################################
    ##################################################################
    # Sort measurement channels
    ##################################################################
    # Get qubit names in channel order
    ch_names = [name.decode() for name in raw_data_dict["value_names"]]

    def _find_channel(ch_name):
        for i, name in enumerate(ch_names):
            if ch_name in name:
                return i + 1

    chan_idxs = {q: (_find_channel(f"{q} I"), _find_channel(f"{q} Q")) for q in Qubits}
    # Dictionary that will store raw shots so that they can later be sorted.
    raw_shots = {q: {} for q in Qubits}
    # Processed data dictionary
    proc_data_dict = {}
    # Readout thresholds dictionary
    Thresholds = {}
    ##################################################################
    ##################################################################
    # Sort readout calibration shots and assign them
    ##################################################################
    for q_idx, qubit in enumerate(Qubits):
        proc_data_dict[qubit] = {}
        _ch_I, _ch_Q = chan_idxs[qubit]
        _raw_shots = raw_data_dict["data"][:, [_ch_I, _ch_Q]]
        if heralded_init:
            _shots_0 = _raw_shots[n_kernels * _total_rounds + 1 :: _cycle]
            _shots_1 = _raw_shots[n_kernels * _total_rounds + 3 :: _cycle]
            _shots_2 = _raw_shots[n_kernels * _total_rounds + 5 :: _cycle]
        else:
            _shots_0 = _raw_shots[n_kernels * _total_rounds + 0 :: _cycle]
            _shots_1 = _raw_shots[n_kernels * _total_rounds + 1 :: _cycle]
            _shots_2 = _raw_shots[n_kernels * _total_rounds + 2 :: _cycle]
        # Rotate data over |0> - |1> axis in IQ plane
        center_0 = np.array([np.mean(_shots_0[:, 0]), np.mean(_shots_0[:, 1])])
        center_1 = np.array([np.mean(_shots_1[:, 0]), np.mean(_shots_1[:, 1])])
        center_2 = np.array([np.mean(_shots_2[:, 0]), np.mean(_shots_2[:, 1])])
        raw_shots[qubit] = _rotate_and_center_data(
            _raw_shots[:, 0], _raw_shots[:, 1], center_0, center_1
        )
        if heralded_init:
            Shots_0 = raw_shots[qubit][n_kernels * _total_rounds + 1 :: _cycle]
            Shots_1 = raw_shots[qubit][n_kernels * _total_rounds + 3 :: _cycle]
            Shots_2 = raw_shots[qubit][n_kernels * _total_rounds + 5 :: _cycle]
        else:
            Shots_0 = raw_shots[qubit][n_kernels * _total_rounds + 0 :: _cycle]
            Shots_1 = raw_shots[qubit][n_kernels * _total_rounds + 1 :: _cycle]
            Shots_2 = raw_shots[qubit][n_kernels * _total_rounds + 2 :: _cycle]
        # Save sorted shots
        proc_data_dict[qubit]["Shots_0"] = Shots_0
        proc_data_dict[qubit]["Shots_1"] = Shots_1
        proc_data_dict[qubit]["Shots_2"] = Shots_2
        # Classify qubit and qutrit Shots
        (
            clf_qubit,
            Fid_dict_qubit,
            M_qubit,
            dec_bounds_qubit,
        ) = _Classify_qubit_calibration_shots(Shots_0, Shots_1)
        proc_data_dict[qubit]["classifier_qubit"] = clf_qubit
        proc_data_dict[qubit]["Fid_dict_qubit"] = Fid_dict_qubit
        proc_data_dict[qubit]["Assignment_matrix_qubit"] = M_qubit
        proc_data_dict[qubit]["dec_bounds_qutrit"] = dec_bounds_qubit
        (
            clf_qutrit,
            Fid_dict_qutrit,
            M_qutrit,
            dec_bounds_qutrit,
        ) = _Classify_qutrit_calibration_shots(Shots_0, Shots_1, Shots_2)
        proc_data_dict[qubit]["classifier_qutrit"] = clf_qutrit
        proc_data_dict[qubit]["Fid_dict_qutrit"] = Fid_dict_qutrit
        proc_data_dict[qubit]["Assignment_matrix_qutrit"] = M_qutrit
        proc_data_dict[qubit]["dec_bounds_qutrit"] = dec_bounds_qutrit
        # Post selection on heralding measurement
        if heralded_init:
            # Sort heralding measurement shots
            _ps_shots_0 = raw_shots[qubit][n_kernels * _total_rounds + 0 :: _cycle]
            _ps_shots_1 = raw_shots[qubit][n_kernels * _total_rounds + 2 :: _cycle]
            _ps_shots_2 = raw_shots[qubit][n_kernels * _total_rounds + 4 :: _cycle]

            def _post_select(shots, ps_shots):
                """
                Post-select shots based on outcome of
                heralding measurement.
                """
                _ps_shots = clf_qutrit.predict(ps_shots)
                _mask = np.array([1 if s == 0 else np.nan for s in _ps_shots])
                # print(np.nansum(_mask)/ len(_mask))
                shots = shots[~np.isnan(_mask)]
                return shots

            Shots_0 = _post_select(Shots_0, _ps_shots_0)
            Shots_1 = _post_select(Shots_1, _ps_shots_1)
            Shots_2 = _post_select(Shots_2, _ps_shots_2)
            # Redefine calibration shots (now post-selected)
            proc_data_dict[qubit]["Shots_0"] = Shots_0
            proc_data_dict[qubit]["Shots_1"] = Shots_1
            proc_data_dict[qubit]["Shots_2"] = Shots_2
            # Rerun classifiers on post-selected data
            (
                clf_qubit,
                Fid_dict_qubit,
                M_qubit,
                dec_bounds_qubit,
            ) = _Classify_qubit_calibration_shots(Shots_0, Shots_1)
            proc_data_dict[qubit]["classifier_qubit"] = clf_qubit
            proc_data_dict[qubit]["Fid_dict_qubit"] = Fid_dict_qubit
            proc_data_dict[qubit]["Assignment_matrix_qubit"] = M_qubit
            proc_data_dict[qubit]["dec_bounds_qubit"] = dec_bounds_qubit
            (
                clf_qutrit,
                Fid_dict_qutrit,
                M_qutrit,
                dec_bounds_qutrit,
            ) = _Classify_qutrit_calibration_shots(Shots_0, Shots_1, Shots_2)
            proc_data_dict[qubit]["classifier_qutrit"] = clf_qutrit
            proc_data_dict[qubit]["Fid_dict_qutrit"] = Fid_dict_qutrit
            proc_data_dict[qubit]["Assignment_matrix_qutrit"] = M_qutrit
            proc_data_dict[qubit]["dec_bounds_qutrit"] = dec_bounds_qutrit
        # Analyze qubit shots over decision boundaries
        _Analyse_qubit_shots_along_decision_boundaries(
            qubit, Shots_0, Shots_1, dec_bounds_qubit, proc_data_dict
        )
        # Analyze qutrit shots over decision boundaries
        _Analyse_qutrit_shots_along_decision_boundaries(
            qubit, Shots_0, Shots_1, Shots_2, dec_bounds_qutrit, proc_data_dict
        )
        # Save rotated shots
        proc_data_dict["raw_shots"] = raw_shots
    return proc_data_dict


def Sort_and_analyze_experiment_shots(proc_data_dict, Rounds, n_kernels, heralded_init):
    """
    Sort and Analyze experiment shots.
    """
    Qubits = [
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
        "D6",
        "D7",
        "D8",
        "D9",
        "Z1",
        "Z2",
        "Z3",
        "Z4",
    ]
    _total_rounds = np.sum(Rounds)
    # Add heralding measurement for experimental shots
    if heralded_init:
        _total_rounds += len(Rounds)
    _cycle = _total_rounds * n_kernels + 3
    # Add heralding measurement for calibration points
    if heralded_init:
        _cycle += 3
    raw_shots = proc_data_dict["raw_shots"]
    # Sort experimental shots
    # kernel 0 - Surface-13 without LRUs
    # kernel 1 - Surface-13 with LRUs
    shots_exp = {k: {} for k in range(n_kernels)}
    Shots_qubit = {k: {} for k in range(n_kernels)}
    Shots_qutrit = {k: {} for k in range(n_kernels)}
    for k in range(n_kernels):
        shots_exp[k] = {q: {} for q in Qubits}
        Shots_qubit[k] = {q: {} for q in Qubits}
        Shots_qutrit[k] = {q: {} for q in Qubits}
    for q in Qubits:
        _clf_qubit = proc_data_dict[q]["classifier_qubit"]
        _clf_qutrit = proc_data_dict[q]["classifier_qutrit"]
        for r_idx, n_rounds in enumerate(Rounds):
            for k in range(n_kernels):
                shots_exp[k][q][f"{n_rounds}_R"] = {}
                Shots_qubit[k][q][f"{n_rounds}_R"] = {}
                Shots_qutrit[k][q][f"{n_rounds}_R"] = {}
            # counter for number of shots in previous rounds
            _aux = int(n_kernels * np.sum(Rounds[:r_idx]))
            if heralded_init:
                _aux = int(n_kernels * (np.sum(Rounds[:r_idx]) + len(Rounds[:r_idx])))
            for r in range(n_rounds):
                # Note we are using the rotated shots already
                for k in range(n_kernels):
                    shots_exp[k][q][f"{n_rounds}_R"][f"round {r+1}"] = raw_shots[q][
                        r
                        + k * (n_rounds + heralded_init)
                        + heralded_init
                        + _aux :: _cycle
                    ]
                for k in range(n_kernels):
                    # Perform Qubit assignment
                    Shots_qubit[k][q][f"{n_rounds}_R"][
                        f"round {r+1}"
                    ] = _clf_qubit.predict(
                        shots_exp[k][q][f"{n_rounds}_R"][f"round {r+1}"]
                    )
                    # Perform Qutrit assignment
                    Shots_qutrit[k][q][f"{n_rounds}_R"][
                        f"round {r+1}"
                    ] = _clf_qutrit.predict(
                        shots_exp[k][q][f"{n_rounds}_R"][f"round {r+1}"]
                    )
            # Sort heralding measurement shots
            if heralded_init:
                for k in range(n_kernels):
                    shots_exp[k][q][f"{n_rounds}_R"]["ps"] = raw_shots[q][
                        k * (n_rounds + heralded_init) + _aux :: _cycle
                    ]
                    # Classify heralding shots (round 0 is defined for heralding measurement)
                    Shots_qutrit[k][q][f"{n_rounds}_R"][
                        "round 0"
                    ] = _clf_qutrit.predict(shots_exp[k][q][f"{n_rounds}_R"]["ps"])
                    # Compute post-selection mask
                    # (used to signal heralding: 1-successful init. ; np.nan-unsuccessfull init.)
                    Shots_qutrit[k][q][f"{n_rounds}_R"]["ps"] = np.array(
                        [
                            1 if s == 0 else np.nan
                            for s in Shots_qutrit[k][q][f"{n_rounds}_R"]["round 0"]
                        ]
                    )
    ##################################################################
    ##################################################################
    # Post-selection on heralding measurement
    ##################################################################
    if heralded_init:
        for R in Rounds:
            _n_shots = len(Shots_qutrit[0][q][f"{R}_R"]["ps"])
            _mask = {k: np.ones(_n_shots) for k in range(n_kernels)}
            for q in Qubits:
                for k in range(n_kernels):
                    _mask[k] *= Shots_qutrit[k][q][f"{R}_R"]["ps"]
            for k in range(n_kernels):
                print(
                    f"{R}_R Percentage of post-selected shots {k}: {np.nansum(_mask[k])/len(_mask[k])*100:.2f}%"
                )
            for q in Qubits:
                for r in range(R):
                    for k in range(n_kernels):
                        # Remove marked (unsuccessful initialization) shots in qubit shots
                        Shots_qubit[k][q][f"{R}_R"][f"round {r+1}"] = Shots_qubit[k][q][
                            f"{R}_R"
                        ][f"round {r+1}"][~np.isnan(_mask[k])]
                        # Remove marked (unsuccessful initialization) shots in qutrit shots
                        Shots_qutrit[k][q][f"{R}_R"][f"round {r+1}"] = Shots_qutrit[k][
                            q
                        ][f"{R}_R"][f"round {r+1}"][~np.isnan(_mask[k])]
    ###########################
    ## Leakage postselection
    ###########################
    # postselect leakage runs based on ancilla qubit msmt, all bad shots are not included
    # Shots_qubit_ps : stores shots after removing leakage runs from based on qutrit RO on the ancillas
    Shots_qubit_ps = {k: {} for k in range(n_kernels)}
    # Shots_qubit_ps_all : stores shots after removing leakage runs from based on qutrit RO on the ancillas and from final data-qubit msmt
    Shots_qubit_ps_all = {k: {} for k in range(n_kernels)}
    # postselected fraction of Leakage runs based on ancilla L postselection
    Ps_fraction_L = {k: np.ones(Rounds[-1]) for k in range(n_kernels)}
    # postselected fraction of Leakage runs based on final data qubit msmt
    Ps_fraction_L_D = {k: np.ones(Rounds[-1]) for k in range(n_kernels)}
    # maskl masks Leakage runs based on ancilla L postselection
    _mask_l = {k: {} for k in range(n_kernels)}
    # maskld masks Leakage runs based on final data qubit msmt
    _mask_ld = {k: {} for k in range(n_kernels)}
    for k in range(n_kernels):
        Shots_qubit_ps[k] = {q: {} for q in Qubits}
        Shots_qubit_ps_all[k] = {q: {} for q in Qubits}
    for q in Qubits:
        for k in range(n_kernels):
            Shots_qubit_ps[k][q] = {f"{R}_R": {} for R in Rounds}
            Shots_qubit_ps_all[k][q] = {f"{R}_R": {} for R in Rounds}
            for R in Rounds:
                _n_shots = len(Shots_qutrit[k][q][f"{R}_R"][f"round {1}"])
                _mask_l[k] = np.ones(_n_shots)
                for r in range(R):
                    # only for ancilla qubits. This assumes that the order of Qubits start with Data qubits and finishes with 4 ancillas
                    for qa in Qubits[-4:]:
                        # set shots == 2 in Shots_qutrit for all ancilla qubit over all rounds to nan and cast this for all rounds, not including those shots in all rounds in any calculation
                        _mask_l[k] *= np.array(
                            [
                                1 if s != 2 else np.nan
                                for s in Shots_qutrit[k][qa][f"{R}_R"][f"round {r+1}"]
                            ]
                        )
                    Ps_fraction_L[k][r] = np.nansum(_mask_l[k]) / _n_shots
                    # mark those shots from before to nan and keep them in the shots. This is important for consistency later
                    Shots_qubit_ps[k][q][f"{R}_R"][f"round {r+1}"] = (
                        Shots_qutrit[k][q][f"{R}_R"][f"round {r+1}"] * _mask_l[k]
                    )
    # postselect leakage runs based on final data qubit msmt
    for q in Qubits[:-4]:
        for k in range(n_kernels):
            for R in Rounds:
                _n_shots_d = len(Shots_qubit_ps[k][q][f"{R}_R"][f"round {R}"])
                # set shots == 2 in Shots_qutrit for all data qubit in the final rounds to nan.
                _mask_ld[k] = np.ones(_n_shots_d)
                for qa in Qubits[:-4]:
                    _mask_ld[k] *= np.array(
                        [
                            1 if s != 2 else np.nan
                            for s in Shots_qubit_ps[k][qa][f"{R}_R"][f"round {R}"]
                        ]
                    )
                Ps_fraction_L_D[k] = np.nansum(_mask_ld[k]) / _n_shots_d
                # mark those shots from before to nan and keep them in the shots. This is important for consistency later
                Shots_qubit_ps_all[k][q][f"{R}_R"][f"round {R}"] = (
                    Shots_qubit_ps[k][q][f"{R}_R"][f"round {R}"] * _mask_ld[k]
                )
    # postselect leakage runs based on final data qubit msmt
    # only for data qubits. This assumes that the order of Qubits start with Data qubits and finishes with 4 ancillas
    for q in Qubits[:-4]:
        for k in range(n_kernels):
            for R in Rounds:
                Shots_qubit_ps[k][q][f"{R}_R"][f"round {R}"] = Shots_qubit_ps_all[k][q][
                    f"{R}_R"
                ][f"round {R}"]
    # Save processed data
    proc_data_dict["Shots_qubit"] = Shots_qubit
    proc_data_dict["Shots_qutrit"] = Shots_qutrit
    proc_data_dict["Shots_exp"] = shots_exp
    proc_data_dict["L_PS"] = Ps_fraction_L
    proc_data_dict["L_PS_D"] = Ps_fraction_L_D
    proc_data_dict["Shots_qubit_ps"] = Shots_qubit_ps
    proc_data_dict["Shots_qubit_ps_all"] = Shots_qubit_ps_all
    return proc_data_dict


def Calculate_leakage_population(
    proc_data_dict,
    Rounds,
    n_kernels,
):
    """
    Calculate leakage population as function of rounds
    """
    Qubits = [
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
        "D6",
        "D7",
        "D8",
        "D9",
        "Z1",
        "Z2",
        "Z3",
        "Z4",
    ]
    Shots_qubit = proc_data_dict["Shots_qubit"]
    Shots_qutrit = proc_data_dict["Shots_qutrit"]
    # Estimate leakage populations
    Population = {k: {q: {} for q in Qubits} for k in range(n_kernels)}
    Population_f = {k: {q: {} for q in Qubits} for k in range(n_kernels)}

    def _get_pop_vector(Shots):
        p0 = np.mean(Shots == 0)
        p1 = np.mean(Shots == 1)
        p2 = np.mean(Shots == 2)
        return np.array([p0, p1, p2])

    for q in Qubits:
        M_inv = np.linalg.inv(proc_data_dict[q]["Assignment_matrix_qutrit"])
        if "Z" in q:
            # For ancilla qubits we can calculate
            # leakage in every measurement round.
            for n_rounds in Rounds:
                for k in range(n_kernels):
                    Population[k][q][f"{n_rounds}_R"] = {}
                for r in range(n_rounds):
                    for k in range(n_kernels):
                        _pop_vec = _get_pop_vector(
                            Shots_qutrit[k][q][f"{n_rounds}_R"][f"round {r+1}"]
                        )
                        Population[k][q][f"{n_rounds}_R"][f"round {r+1}"] = np.dot(
                            _pop_vec, M_inv
                        )
            for k in range(n_kernels):
                Population_f[k][q] = np.array(
                    [
                        Population[k][q][f"{Rounds[-1]}_R"][key][2]
                        for key in Population[k][q][f"{Rounds[-1]}_R"].keys()
                    ]
                )
        else:
            # For data qubits we can only calculate
            # leakage in the last measurement round.
            for n_rounds in Rounds:
                for k in range(n_kernels):
                    _pop_vec = _get_pop_vector(
                        Shots_qutrit[k][q][f"{n_rounds}_R"][f"round {n_rounds}"]
                    )
                    Population[k][q][f"{n_rounds}_R"] = np.dot(_pop_vec, M_inv)
            for k in range(n_kernels):
                Population_f[k][q] = np.array(
                    [Population[k][q][key][2] for key in Population[k][q].keys()]
                )
    return Population, Population_f


def Compute_defects(
    proc_data_dict,
    Rounds,
    n_kernels,
):
    """
    Compute defects detected by stabilizer measurements.
    This is done in 3 different ways:
        - Regular defects from qubit shots
        - Defects + leakage signalized in the ancilla readout
          (used for ancilla leakage post selection)
        - Defects taking only data qubit measurements into account
          (used for majority voting error decoding).
    """
    Qubits = [
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
        "D6",
        "D7",
        "D8",
        "D9",
        "Z1",
        "Z2",
        "Z3",
        "Z4",
    ]
    Shots_qubit = proc_data_dict["Shots_qubit"]
    Shots_qutrit = proc_data_dict["Shots_qutrit"]
    # qubit shots based on three-level readout, marking leakage runs (ancilla shot over rounds+final data qubit msmt) over all rounds with nan
    Shots_qubit_ps = proc_data_dict["Shots_qubit_ps"]
    # Calculate defects for each stabilizer (ancilla qubit)
    _Ancilla_qubits = [q for q in Qubits if "Z" in q]
    # defects of ancilla qubits over rounds with qubit shots [Shots_qubit]
    Defects = {q: {k: {} for k in range(n_kernels)} for q in _Ancilla_qubits}
    # defects of ancilla qubits over rounds with qubit shots [Shots_qubit_ps]
    Defects_f = {q: {k: {} for k in range(n_kernels)} for q in _Ancilla_qubits}
    # defect for majority voting with dummy shots, ignoring ancilla outcomes [Shots_qubit]
    Defects_MV = {q: {k: {} for k in range(n_kernels)} for q in _Ancilla_qubits}
    # defect for majority voting with dummy shots_f, ignoring ancilla outcomes [Shots_qubit_ps]
    Defects_MV_f = {q: {k: {} for k in range(n_kernels)} for q in _Ancilla_qubits}
    # defect rate for the defect rate plot [Shots_qubit]
    Defect_rate = {q: {k: {} for k in range(n_kernels)} for q in _Ancilla_qubits}
    # defect rate_f for the defect rate plot [Shots_qubit_ps]
    Defect_rate_f = {q: {k: {} for k in range(n_kernels)} for q in _Ancilla_qubits}
    for q in _Ancilla_qubits:
        for n_rounds in Rounds:
            for k in range(n_kernels):
                # Data qubits measured by each stabilizer
                # (will be used to compute measured data-qubit parity)
                if q == "Z1":
                    stab_data_qubits = ["D1", "D2", "D4", "D5"]
                elif q == "Z2":
                    stab_data_qubits = ["D3", "D6"]
                elif q == "Z3":
                    stab_data_qubits = ["D4", "D7"]
                elif q == "Z4":
                    stab_data_qubits = ["D5", "D6", "D8", "D9"]
                # Sort final data qubit measurement shots
                Data_shots = [
                    Shots_qubit[k][_dq][f"{n_rounds}_R"][f"round {n_rounds}"]
                    for _dq in stab_data_qubits
                ]
                # Sort final data qubit measurement shots based post-selected leakage shots in final data-qubit msmt
                Data_shots_ps = [
                    Shots_qubit_ps[k][_dq][f"{n_rounds}_R"][f"round {n_rounds}"]
                    for _dq in stab_data_qubits
                ]
                # Compute defects
                Defects[q][k][f"{n_rounds}_R"] = _calculate_defects(
                    Shots_qubit[k][q][f"{n_rounds}_R"],
                    n_rounds,
                    Data_qubit_meas=Data_shots,
                )
                # Compute defects while signaling leakage on ancilla qubits
                Defects_f[q][k][f"{n_rounds}_R"] = _calculate_defects(
                    Shots_qubit_ps[k][q][f"{n_rounds}_R"],
                    n_rounds,
                    Data_qubit_meas=Data_shots_ps,
                )
                # Compute defects for Majority voting decoder
                # To disregard ancilla qubit measurements we'll create a dummy array
                # of ancilla qubit shots that are all zero
                _dummy_shots = {
                    key: np.zeros(arr.shape)
                    for key, arr in Shots_qubit[k][q][f"{n_rounds}_R"].items()
                }
                Defects_MV[q][k][f"{n_rounds}_R"] = _calculate_defects(
                    _dummy_shots, n_rounds, Data_qubit_meas=Data_shots
                )
                # Compute defects for Majority voting decoder with Shots_qubit_ps
                # To disregard ancilla qubit measurements we'll create a dummy array
                # of ancilla qubit shots that are all zeros
                _dummy_shots_f = {
                    key: np.zeros(arr.shape)
                    for key, arr in Shots_qubit_ps[k][q][f"{n_rounds}_R"].items()
                }
                Defects_MV_f[q][k][f"{n_rounds}_R"] = _calculate_defects(
                    _dummy_shots_f, n_rounds, Data_qubit_meas=Data_shots_ps
                )
                # Compute defect probability
                Defect_rate[q][k][f"{n_rounds}_R"] = np.mean(
                    (1 - Defects[q][k][f"{n_rounds}_R"]) / 2, axis=0
                )
                # Compute defect probability (post-selecting on ancilla leakage)
                Defect_rate_f[q][k][f"{n_rounds}_R"] = np.nanmean(
                    (1 - Defects_f[q][k][f"{n_rounds}_R"]) / 2, axis=0
                )

    return Defects, Defects_f, Defects_MV, Defects_MV_f, Defect_rate, Defect_rate_f


def _get_datafilepath_from_timestamp(timestamp):
    """
    Return the full filepath of a datafile designated by a timestamp.

    Args:
        timestamp (str)
            formatted as "YYMMHH_hhmmss""
    Return:
        filepath (str)
            the full filepath of a datafile

    Note: there also exist two separate functions that are typically
    combined in analysis to achieve the same effect.
    These are "data_from_time" and "measurement_filename".

    This function is intended to replace both of these and be faster.

    """

    # Not only verifies but also decomposes the timestamp
    daystamp, tstamp = verify_timestamp(timestamp)

    daydir = os.listdir(os.path.join(datadir, daystamp))

    # Looking for the folder starting with the right timestamp
    measdir_names = [item for item in daydir if item.startswith(tstamp)]

    if len(measdir_names) > 1:
        raise ValueError("Timestamp is not unique")
    elif len(measdir_names) == 0:
        raise ValueError("No data at timestamp.")
    measdir_name = measdir_names[0]
    # Naming follows a standard convention
    data_fp = os.path.join(datadir, daystamp, measdir_name, measdir_name + ".hdf5")
    return data_fp


def _get_timestamps_in_range(
    timestamp_start,
    timestamp_end=None,
    label=None,
    exact_label_match=False,
    folder=None,
):
    """
    Input parameters:
        label: a string or list of strings to compare the experiment name to
        exact_label_match: 'True' : the label should exactly match the folder name
        (excluding "timestamp_"). 'False': the label must be a substring of the folder name


    """
    if folder is None:
        folder = datadir

    datetime_start = datetime_from_timestamp(timestamp_start)
    if timestamp_end is None:
        datetime_end = datetime.datetime.today()
    else:
        datetime_end = datetime_from_timestamp(timestamp_end)
    days_delta = (datetime_end.date() - datetime_start.date()).days
    all_timestamps = []
    for day in reversed(list(range(days_delta + 1))):
        date = datetime_start + datetime.timedelta(days=day)
        datemark = timestamp_from_datetime(date)[:8]
        try:
            all_measdirs = [d for d in os.listdir(os.path.join(folder, datemark))]
        except FileNotFoundError:
            # Sometimes, when choosing multiples days, there is a day
            # with no measurements
            all_measdirs = []

        # Remove all hidden folders to prevent errors
        all_measdirs = [d for d in all_measdirs if not d.startswith(".")]

        if exact_label_match:
            if isinstance(label, str):
                label = [label]
            for each_label in label:
                # Remove 'hhmmss_' timestamp and check if exactly equals
                all_measdirs = [x for x in all_measdirs if each_label == x[7:]]
        else:
            if isinstance(label, str):
                label = [label]
            for each_label in label:
                all_measdirs = [x for x in all_measdirs if each_label in x]
        if (date.date() - datetime_start.date()).days == 0:
            # Check if newer than starting timestamp
            timemark_start = timemark_from_datetime(datetime_start)
            all_measdirs = [
                dirname
                for dirname in all_measdirs
                if int(dirname[:6]) >= int(timemark_start)
            ]

        if (date.date() - datetime_end.date()).days == 0:
            # Check if older than ending timestamp
            timemark_end = timemark_from_datetime(datetime_end)
            all_measdirs = [
                dirname
                for dirname in all_measdirs
                if int(dirname[:6]) <= int(timemark_end)
            ]
        timestamps = ["{}_{}".format(datemark, dirname[:6]) for dirname in all_measdirs]
        timestamps.reverse()
        all_timestamps += timestamps
    # Ensures the order of the timestamps is ascending
    all_timestamps.sort()
    if len(all_timestamps) == 0:
        raise ValueError('No matching timestamps found for label "{}"'.format(label))
    return all_timestamps


def run_analysis(raw_data_dict, Rounds, n_kernels, heralded_init):
    ##########################################################
    # Sorting and analyzing data
    ##########################################################
    # Analyze calibration data
    proc_data_dict = Analyze_calibration_shots(
        raw_data_dict, Rounds, n_kernels, heralded_init
    )
    # Sort and bin experiment shots
    proc_data_dict = Sort_and_analyze_experiment_shots(
        proc_data_dict, Rounds, n_kernels, heralded_init
    )
    # Calculate leakage population
    Population, Population_f = Calculate_leakage_population(
        proc_data_dict, Rounds, n_kernels
    )
    ##########################################################
    # Extracting syndrome and logical outcome information
    ##########################################################
    # Compute Defects for each stabilizer
    Defects, Defects_f, Defects_MV, Defect_rate, Defect_rate_f = Compute_defects(
        proc_data_dict, Rounds, n_kernels
    )
    # Compute syndrome frames and Logical expectation values
    Decoding_frames = _get_decoding_frames(Defects)
    Decoding_frames_f = _get_decoding_frames(Defects_f)
    Decoding_frames_MV = _get_decoding_frames(Defects_MV)
    _Data_qubits = ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9"]
    Logical_outcomes = _calculate_logical_outcomes(
        proc_data_dict["Shots_qubit"], _Data_qubits
    )
    ##########################################################
    # Decode errors
    ##########################################################
    Decoder_LUT = {
        "0000": +1,  # No error
        "0001": -1,  # error on D8 or D9
        "0010": -1,  # error on D7
        "0100": -1,  # error on D3
        "1000": -1,  # error on D1 or D2
        "1001": -1,  # error on D5
        "1010": -1,  # error on D4
        "1100": +1,  # double error
        "0101": -1,  # error on D6
        "0110": +1,  # double error
        "0011": +1,  # double error
        "0111": +1,  # double error
        "1011": +1,  # double error
        "1101": +1,  # double error
        "1110": +1,  # double error
        "1111": +1,  # double error
    }
    Rounds, Mean_logical_out_corr = _decode_space_biased(
        Decoding_frames, Logical_outcomes, Decoder_LUT
    )
    Rounds, Mean_logical_out_corr_f = _decode_space_biased(
        Decoding_frames_f, Logical_outcomes, Decoder_LUT
    )
    Rounds, Mean_logical_out_MV = _decode_space_biased(
        Decoding_frames_MV, Logical_outcomes, Decoder_LUT
    )
    ##########################################################
    # Plot logical error versus rounds
    ##########################################################
    # Dynamical decoupling mask (accounts for dynamical decoupling which flips logical expectation value)
    _mask = np.array([1 if r % 2 == 1 else -1 for r in Rounds])
    # Plot
    plt.plot(
        Rounds, (1 + (Mean_logical_out_corr[0] * _mask)) / 2, "C0-", label="LUT wo LRU"
    )
    plt.plot(
        Rounds, (1 + (Mean_logical_out_MV[0] * _mask)) / 2, "C2-", label="MV wo LRU"
    )
    plt.plot(
        Rounds,
        (1 + (Mean_logical_out_corr_f[0] * _mask)) / 2,
        "C0--",
        label="$|2\\rangle$-PS wo LRU",
    )
    # plt.plot(Rounds, (1+(Mean_logical_out_corr[1]*_mask))/2, 'C1-', label='LUT w LRU')
    # plt.plot(Rounds, (1+(Mean_logical_out_MV[1]*_mask))/2, 'C3-', label='MV w LRU')
    # plt.plot(Rounds, (1+(Mean_logical_out_corr_f[1]*_mask))/2, 'C1--', label='$|2\\rangle$-PS w LRU')
    plt.legend()
    plt.ylabel("Logical fidelity")
    plt.xlabel("Rounds")

    return (
        Defect_rate,
        Defect_rate_f,
        Population_f,
        Mean_logical_out_corr,
        Mean_logical_out_corr_f,
        Mean_logical_out_MV,
    )
