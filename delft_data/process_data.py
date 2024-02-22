#!/usr/bin/env python
# coding: utf-8

import gc
import warnings

warnings.filterwarnings("ignore")

# ### Imports

import numpy as np

from copy import deepcopy

from .helper_functions import (
    _rotate_and_center_data,
    _Classify_qubit_calibration_shots,
    _Classify_qutrit_calibration_shots,
    _Analyse_qubit_shots_along_decision_boundaries,
    _Analyse_qutrit_shots_along_decision_boundaries,
    _calculate_defects,
    _array_to_binary_string,
    Analyze_calibration_shots,
    Sort_and_analyze_experiment_shots,
    Calculate_leakage_population,
    Compute_defects,
    _syndrome_to_pauli_frame_correction,
    _get_nearest_neighbors,
    _QED_ps,
    _pauli_frame_update,
    _compute_logical_operator_shots,
    _get_timestamps_in_range,
    _get_datafilepath_from_timestamp,
    _get_decoding_frames,
    _calculate_logical_outcomes,
)

from iq_readout.two_state_classifiers import TwoStateLinearClassifierFit
from iq_readout.three_state_classifiers import ThreeStateLinearClassifier2D


##################################################################
# Experiment settings
##################################################################
Qubits = ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "Z1", "Z2", "Z3", "Z4"]
n_kernels = 2
Rounds = [1, 2, 4, 8, 16]
heralded_init = True
_total_rounds = np.sum(Rounds)
# Add heralding measurement for experimental shots
if heralded_init:
    _total_rounds += len(Rounds)
_cycle = _total_rounds * n_kernels + 3
# Add heralding measurement for calibration points
if heralded_init:
    _cycle += 3


def process_data(raw_data_dict):
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
    print("Processing calibration shots...")
    for q_idx, qubit in enumerate(Qubits):
        print(f"{qubit} ({q_idx+1:2d}/{len(Qubits)})", end="\n")
        proc_data_dict[qubit] = {}
        _ch_I, _ch_Q = chan_idxs[qubit]
        raw_shots[qubit] = raw_data_dict["data"][:, [_ch_I, _ch_Q]]
        if heralded_init:
            Shots_0 = raw_shots[qubit][n_kernels * _total_rounds + 1 :: _cycle]
            Shots_1 = raw_shots[qubit][n_kernels * _total_rounds + 3 :: _cycle]
            Shots_2 = raw_shots[qubit][n_kernels * _total_rounds + 5 :: _cycle]
        else:
            Shots_0 = raw_shots[qubit][n_kernels * _total_rounds + 0 :: _cycle]
            Shots_1 = raw_shots[qubit][n_kernels * _total_rounds + 1 :: _cycle]
            Shots_2 = raw_shots[qubit][n_kernels * _total_rounds + 2 :: _cycle]
        # Classify qubit and qutrit Shots
        clf_qubit = TwoStateLinearClassifierFit().fit(Shots_0, Shots_1)
        clf_qutrit = ThreeStateLinearClassifier2D().fit(Shots_0, Shots_1, Shots_2)
        proc_data_dict[qubit]["classifier_qubit_before_ps"] = clf_qubit
        proc_data_dict[qubit]["classifier_qutrit_before_ps"] = clf_qutrit

        # Measurement fidelity
        y0, y1 = clf_qubit.predict(Shots_0), clf_qubit.predict(Shots_1)
        fid = 0.5 * (np.average(y0 == 0) + np.average(y1 == 1))
        print(f"Fmeas(2-state       )={fid:0.4f}", end=" ")

        y0, y1 = clf_qutrit.predict(Shots_0), clf_qutrit.predict(Shots_1)
        fid = 0.5 * (np.average(y0 == 0) + np.average(y1 == 1))
        print(f"Fmeas(3-state, 0&1      )={fid:0.4f}")

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
            # Measurement fidelity
            y0, y1 = clf_qubit.predict(Shots_0), clf_qubit.predict(Shots_1)
            fid = 0.5 * (np.average(y0 == 0) + np.average(y1 == 1))
            print(f"Fmeas(2-state, PS   )={fid:0.4f}", end=" ")
            y0, y1 = clf_qutrit.predict(Shots_0), clf_qutrit.predict(Shots_1)
            fid = 0.5 * (np.average(y0 == 0) + np.average(y1 == 1))
            print(f"Fmeas(3-state, 0&1 PS   )={fid:0.4f}")

            # Classify qubit and qutrit Shots
            clf_qubit = TwoStateLinearClassifierFit().fit(Shots_0, Shots_1)
            clf_qutrit = ThreeStateLinearClassifier2D().fit(Shots_0, Shots_1, Shots_2)
            proc_data_dict[qubit]["classifier_qubit"] = clf_qubit
            proc_data_dict[qubit]["classifier_qutrit"] = clf_qutrit

            # Measurement fidelity
            y0, y1 = clf_qubit.predict(Shots_0), clf_qubit.predict(Shots_1)
            fid = 0.5 * (np.average(y0 == 0) + np.average(y1 == 1))
            print(f"Fmeas(2-state_PS, PS)={fid:0.4f}", end=" ")
            y0, y1 = clf_qutrit.predict(Shots_0), clf_qutrit.predict(Shots_1)
            fid = 0.5 * (np.average(y0 == 0) + np.average(y1 == 1))
            print(f"Fmeas(3-state_PS, 0&1 PS)={fid:0.4f}")
            print(clf_qubit.params())

    del raw_data_dict
    gc.collect()

    # ### Analyze experiment shots
    #
    # Experiment shots will be stored in a three dictionaries.
    #
    #  1. The digitized shots from qubit (2-state) readout:
    #     ```Python
    #     proc_data_dict['Shots_qubit'][k]['<qubit>']['<Rounds>_R']['round <r>'] = array[n_repetitions]
    #     ```
    #  2. The trigitized shots from qutrit (3-state) readout:
    #     ```Python
    #     proc_data_dict['Shots_qutrit'][k]['<qubit>']['<Rounds>_R']['round <r>'] = array[n_repetitions]
    #     ```
    #  3. The raw IQ voltages obtained from readout:
    #     ```Python
    #     proc_data_dict['Shots_exp'][k]['<qubit>']['<Rounds>_R']['round <r>'] = array[n_repetitions, 2]
    #     ```
    #
    # where ```k``` stands for the experiment:
    #
    #     Without LRU: k = 0
    #
    # `<Rounds>` stands for the number of stabilizer measurements in the experiment,
    #
    # `<r>` stands for the QEC round of the experiment. For data qubits the only relevant shots are the last round of the experiment `<Rounds>=<r>`. All other shots have no meaning.
    #
    # Note: All shots are post-selected based on an initialization measurement that heralds successful initialization in the ground state.

    # In[6]:

    ##################################################################
    # Sort and Analyze experiment shots, post-select
    # on leakage and calculate defect rate
    ##################################################################
    print("Processing shots...")
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
    for q_idx, q in enumerate(Qubits):
        print(f"{q} ({q_idx+1}/{len(Qubits)})", end="\r")
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

    del raw_shots
    gc.collect()

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
                        # Remove marked (unsuccessful initialization) shots in exp shots
                        shots_exp[k][q][f"{R}_R"][f"round {r+1}"] = shots_exp[k][q][
                            f"{R}_R"
                        ][f"round {r+1}"][~np.isnan(_mask[k])]
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
    Shots_qubit_ps = {k: {} for k in range(n_kernels)}
    Ps_fraction_L = {k: np.ones(Rounds[-1]) for k in range(n_kernels)}
    Ps_fraction_L_D = {k: np.ones(Rounds[-1]) for k in range(n_kernels)}
    _mask_l = {k: {} for k in range(n_kernels)}
    _mask_ld = {k: {} for k in range(n_kernels)}
    for k in range(n_kernels):
        Shots_qubit_ps[k] = {q: {} for q in Qubits}
    for q in Qubits:
        for k in range(n_kernels):
            Shots_qubit_ps[k][q] = {f"{R}_R": {} for R in Rounds}
            for R in Rounds:
                _n_shots = len(Shots_qutrit[k][q][f"{R}_R"][f"round {1}"])
                _mask_l[k] = np.ones(_n_shots)
                for r in range(R):
                    for qa in Qubits[-4:]:
                        _mask_l[k] *= np.array(
                            [
                                1 if s != 2 else np.nan
                                for s in Shots_qutrit[k][qa][f"{R}_R"][f"round {r+1}"]
                            ]
                        )
                    Ps_fraction_L[k][r] = np.nansum(_mask_l[k]) / _n_shots
                    Shots_qubit_ps[k][q][f"{R}_R"][f"round {r+1}"] = (
                        Shots_qutrit[k][q][f"{R}_R"][f"round {r+1}"] * _mask_l[k]
                    )

    # postselect leakage runs based on final data qubit msmt
    for k in range(n_kernels):
        for R in Rounds:
            _n_shots_d = len(Shots_qubit_ps[k][q][f"{R}_R"][f"round {R}"])
            _mask_ld[k] = np.ones(_n_shots_d)
            for qa in Qubits[:-4]:
                _mask_ld[k] *= np.array(
                    [
                        1 if s != 2 else np.nan
                        for s in Shots_qubit_ps[k][qa][f"{R}_R"][f"round {R}"]
                    ]
                )
            Ps_fraction_L_D[k] = np.nansum(_mask_ld[k]) / _n_shots_d
            for r in range(1, R + 1):
                for q in Qubits:
                    Shots_qubit_ps[k][q][f"{R}_R"][f"round {r}"] *= _mask_ld[k]

    ################################################
    # LEAKAGE POST SELECTION CODE HAS BEEN EDITED
    ################################################

    # Save processed data
    proc_data_dict["Shots_qubit"] = Shots_qubit
    proc_data_dict["Shots_qutrit"] = Shots_qutrit
    proc_data_dict["Shots_exp"] = shots_exp
    proc_data_dict["Shots_qubit_ps"] = Shots_qubit_ps

    return proc_data_dict
