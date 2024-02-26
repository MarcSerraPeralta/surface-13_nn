import matplotlib.pyplot as plt
import numpy as np

def _gauss_pdf(x, x0, sigma):
    return np.exp(-((x-x0)/sigma)**2/2)

def double_gauss(x, x0, x1, sigma0, sigma1, A, r):
    _dist0 = A*( (1-r)*_gauss_pdf(x, x0, sigma0) + r*_gauss_pdf(x, x1, sigma1) )
    return _dist0

def qubit_ssro_IQ_projection_plotfn(
    proc_data_dict,
    qubit):            
    # Select parameters from processed data dictionary
    shots_0 = proc_data_dict[qubit]['Shots_0']
    shots_1 = proc_data_dict[qubit]['Shots_1']
    projection_01 = proc_data_dict[qubit]['projection_qubit']
    classifier = proc_data_dict[qubit]['classifier_qubit']
    dec_bounds = proc_data_dict[qubit]['dec_bounds_qubit']
    Fid_dict = proc_data_dict[qubit]['Fid_dict_qubit']
    # Set axis
    fig = plt.figure(figsize=(10*.65,5*.65), dpi=100)
    axs = [fig.add_subplot(121),
           fig.add_subplot(122)]
    # Fit 2D gaussians
    from scipy.optimize import curve_fit
    def twoD_Gaussian(data, amplitude, x0, y0, sigma_x, sigma_y, theta):
        x, y = data
        x0 = float(x0)
        y0 = float(y0)    
        a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
        b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
        c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
        g = amplitude*np.exp( - (a*((x-x0)**2) + 2*b*(x-x0)*(y-y0) 
                                + c*((y-y0)**2)))
        return g.ravel()
    def _fit_2D_gaussian(X, Y):
        counts, _x, _y = np.histogram2d(X, Y, bins=[100, 100], density=True)
        x = (_x[:-1] + _x[1:]) / 2
        y = (_y[:-1] + _y[1:]) / 2
        _x, _y = np.meshgrid(_x, _y)
        x, y = np.meshgrid(x, y)
        p0 = [counts.max(), np.mean(X), np.mean(Y), np.std(X), np.std(Y), 0]
        popt, pcov = curve_fit(twoD_Gaussian, (x, y), counts.T.ravel(), p0=p0)
        return popt
    popt_0 = _fit_2D_gaussian(shots_0[:,0], shots_0[:,1])
    popt_1 = _fit_2D_gaussian(shots_1[:,0], shots_1[:,1])
    # Plot stuff
    axs[0].plot(shots_0[:,0], shots_0[:,1], '.', color='C0', alpha=0.05)
    axs[0].plot(shots_1[:,0], shots_1[:,1], '.', color='C3', alpha=0.05)
    axs[0].plot([0, popt_0[1]], [0, popt_0[2]], '--', color='k', lw=.5)
    axs[0].plot([0, popt_1[1]], [0, popt_1[2]], '--', color='k', lw=.5)
    axs[0].plot(popt_0[1], popt_0[2], '.', color='C0', label='ground')
    axs[0].plot(popt_1[1], popt_1[2], '.', color='C3', label='excited')
    axs[0].plot(popt_0[1], popt_0[2], 'x', color='white')
    axs[0].plot(popt_1[1], popt_1[2], 'x', color='white')
    # Draw 4sigma ellipse around mean
    from matplotlib.patches import Ellipse
    circle_0 = Ellipse((popt_0[1], popt_0[2]),
                      width=4*popt_0[3], height=4*popt_0[4],
                      angle=-popt_0[5]*180/np.pi,
                      ec='white', fc='none', ls='--', lw=1.25, zorder=10)
    axs[0].add_patch(circle_0)
    circle_1 = Ellipse((popt_1[1], popt_1[2]),
                      width=4*popt_1[3], height=4*popt_1[4],
                      angle=-popt_1[5]*180/np.pi,
                      ec='white', fc='none', ls='--', lw=1.25, zorder=10)
    axs[0].add_patch(circle_1)
    # Plot classifier zones
    from matplotlib.patches import Polygon
    _all_shots = np.concatenate((shots_0, shots_1))
    _lim = np.max([ np.max(np.abs(_all_shots[:,0]))*1.1, np.max(np.abs(_all_shots[:,1]))*1.1 ])
    # Find limits of classifier zones
    c_0 = [popt_0[1], popt_0[2]]
    c_1 = [popt_1[1], popt_1[2]]
    c_mean = np.mean([c_0, c_1], axis=0)
    c_0_limit = c_mean+(c_0-c_mean)*1e3
    c_1_limit = c_mean+(c_1-c_mean)*1e3
    # Plot 0 area
    _points = [dec_bounds['left'], dec_bounds['right'], c_0_limit]
    _patch = Polygon(_points, color='C0', alpha=0.2, lw=0)
    axs[0].add_patch(_patch)
    # Plot 1 area
    _points = [dec_bounds['left'], dec_bounds['right'], c_1_limit]
    _patch = Polygon(_points, color='C3', alpha=0.2, lw=0)
    axs[0].add_patch(_patch)
    # Plot decision boundary
    _x0, _y0 = dec_bounds['left']
    _x1, _y1 = dec_bounds['right']
    axs[0].plot([_x0, _x1], [_y0, _y1], 'k--', lw=1)
    axs[0].set_xlim(-_lim, _lim)
    axs[0].set_ylim(-_lim, _lim)
    axs[0].legend(frameon=False)
    axs[0].set_xlabel('Integrated voltage I')
    axs[0].set_ylabel('Integrated voltage Q')
    axs[0].set_title(f'IQ plot qubit {qubit}')
    # fig.suptitle(f'{timestamp}\n')
    ##########################
    # Plot projection
    ##########################
    _bin_c = projection_01['bin_centers']
    bin_width = _bin_c[1]-_bin_c[0]
    axs[1].bar(_bin_c, projection_01['h0'], bin_width, fc='C0', alpha=0.4)
    axs[1].bar(_bin_c, projection_01['h1'], bin_width, fc='C3', alpha=0.4)
    axs[1].plot(_bin_c, double_gauss(_bin_c, *projection_01['popt0']), '-C0')
    axs[1].plot(_bin_c, double_gauss(_bin_c, *projection_01['popt1']), '-C3')
    axs[1].axvline(projection_01['threshold'], ls='--', color='k', lw=1)
    axs[1].text(projection_01['popt0'][0], projection_01['popt0'][4]/2,
                r'$|g\rangle$', ha='center', va='center', color='C0')
    axs[1].text(projection_01['popt1'][0], projection_01['popt1'][4]/2,
                r'$|e\rangle$', ha='center', va='center', color='C3')
    axs[1].set_xticklabels([])
    axs[1].set_xlim(_bin_c[0], _bin_c[-1])
    axs[1].set_ylim(bottom=0)
    axs[1].set_title('Projection of data')
    # Write fidelity textbox
    text = '\n'.join(('Assignment fidelity:',
                      f'$F_g$ : {Fid_dict["0"]*100:.1f}%',
                      f'$F_e$ : {Fid_dict["1"]*100:.1f}%',
                      f'$F_\mathrm{"{avg}"}$ : {Fid_dict["avg"]*100:.1f}%'))
    props = dict(boxstyle='round', facecolor='gray', alpha=.2)
    axs[1].text(1.05, 1, text, transform=axs[1].transAxes,
                verticalalignment='top', bbox=props)

def qutrit_ssro_IQ_projection_plotfn(
    proc_data_dict,
    qubit):            
    # Select parameters from processed data dictionary
    shots_0 = proc_data_dict[qubit]['Shots_0']
    shots_1 = proc_data_dict[qubit]['Shots_1']
    shots_2 = proc_data_dict[qubit]['Shots_2']
    projection_01 = proc_data_dict[qubit]['projection_01']
    projection_12 = proc_data_dict[qubit]['projection_12']
    projection_02 = proc_data_dict[qubit]['projection_02']
    classifier = proc_data_dict[qubit]['classifier_qutrit']
    dec_bounds = proc_data_dict[qubit]['dec_bounds_qutrit']
    Fid_dict = proc_data_dict[qubit]['Fid_dict_qutrit']
    # Set axis
    fig = plt.figure(figsize=(10*.65,5*.65), dpi=100)
    axs = [fig.add_subplot(121),
           fig.add_subplot(322),
           fig.add_subplot(324),
           fig.add_subplot(326)]
    # Fit 2D gaussians
    from scipy.optimize import curve_fit
    def twoD_Gaussian(data, amplitude, x0, y0, sigma_x, sigma_y, theta):
        x, y = data
        x0 = float(x0)
        y0 = float(y0)    
        a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
        b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
        c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
        g = amplitude*np.exp( - (a*((x-x0)**2) + 2*b*(x-x0)*(y-y0) 
                                + c*((y-y0)**2)))
        return g.ravel()
    def _fit_2D_gaussian(X, Y):
        counts, _x, _y = np.histogram2d(X, Y, bins=[100, 100], density=True)
        x = (_x[:-1] + _x[1:]) / 2
        y = (_y[:-1] + _y[1:]) / 2
        _x, _y = np.meshgrid(_x, _y)
        x, y = np.meshgrid(x, y)
        p0 = [counts.max(), np.mean(X), np.mean(Y), np.std(X), np.std(Y), 0]
        popt, pcov = curve_fit(twoD_Gaussian, (x, y), counts.T.ravel(), p0=p0)
        return popt
    popt_0 = _fit_2D_gaussian(shots_0[:,0], shots_0[:,1])
    popt_1 = _fit_2D_gaussian(shots_1[:,0], shots_1[:,1])
    popt_2 = _fit_2D_gaussian(shots_2[:,0], shots_2[:,1])
    # Plot stuff
    axs[0].plot(shots_0[:,0], shots_0[:,1], '.', color='C0', alpha=0.05)
    axs[0].plot(shots_1[:,0], shots_1[:,1], '.', color='C3', alpha=0.05)
    axs[0].plot(shots_2[:,0], shots_2[:,1], '.', color='C2', alpha=0.05)
    axs[0].plot([0, popt_0[1]], [0, popt_0[2]], '--', color='k', lw=.5)
    axs[0].plot([0, popt_1[1]], [0, popt_1[2]], '--', color='k', lw=.5)
    axs[0].plot([0, popt_2[1]], [0, popt_2[2]], '--', color='k', lw=.5)
    axs[0].plot(popt_0[1], popt_0[2], '.', color='C0', label='ground')
    axs[0].plot(popt_1[1], popt_1[2], '.', color='C3', label='excited')
    axs[0].plot(popt_2[1], popt_2[2], '.', color='C2', label='$2^\mathrm{nd}$ excited')
    axs[0].plot(popt_0[1], popt_0[2], 'x', color='white')
    axs[0].plot(popt_1[1], popt_1[2], 'x', color='white')
    axs[0].plot(popt_2[1], popt_2[2], 'x', color='white')
    # Draw 4sigma ellipse around mean
    from matplotlib.patches import Ellipse
    circle_0 = Ellipse((popt_0[1], popt_0[2]),
                      width=4*popt_0[3], height=4*popt_0[4],
                      angle=-popt_0[5]*180/np.pi,
                      ec='white', fc='none', ls='--', lw=1.25, zorder=10)
    axs[0].add_patch(circle_0)
    circle_1 = Ellipse((popt_1[1], popt_1[2]),
                      width=4*popt_1[3], height=4*popt_1[4],
                      angle=-popt_1[5]*180/np.pi,
                      ec='white', fc='none', ls='--', lw=1.25, zorder=10)
    axs[0].add_patch(circle_1)
    circle_2 = Ellipse((popt_2[1], popt_2[2]),
                      width=4*popt_2[3], height=4*popt_2[4],
                      angle=-popt_2[5]*180/np.pi,
                      ec='white', fc='none', ls='--', lw=1.25, zorder=10)
    axs[0].add_patch(circle_2)
    # Plot classifier zones
    from matplotlib.patches import Polygon
    _all_shots = np.concatenate((shots_0, shots_1, shots_2))
    _lim = np.max([ np.max(np.abs(_all_shots[:,0]))*1.1, np.max(np.abs(_all_shots[:,1]))*1.1 ])
    Lim_points = {}
    for bound in ['01', '12', '02']:
        dec_bounds['mean']
        _x0, _y0 = dec_bounds['mean']
        _x1, _y1 = dec_bounds[bound]
        a = (_y1-_y0)/(_x1-_x0)
        b = _y0 - a*_x0
        _xlim = 1e2*np.sign(_x1-_x0)
        _ylim = a*_xlim + b
        Lim_points[bound] = _xlim, _ylim
    # Plot 0 area
    _points = [dec_bounds['mean'], Lim_points['01'], Lim_points['02']]
    _patch = Polygon(_points, color='C0', alpha=0.2, lw=0)
    axs[0].add_patch(_patch)
    # Plot 1 area
    _points = [dec_bounds['mean'], Lim_points['01'], Lim_points['12']]
    _patch = Polygon(_points, color='C3', alpha=0.2, lw=0)
    axs[0].add_patch(_patch)
    # Plot 2 area
    _points = [dec_bounds['mean'], Lim_points['02'], Lim_points['12']]
    _patch = Polygon(_points, color='C2', alpha=0.2, lw=0)
    axs[0].add_patch(_patch)
    # Plot decision boundary
    for bound in ['01', '12', '02']:
        _x0, _y0 = dec_bounds['mean']
        _x1, _y1 = Lim_points[bound]
        axs[0].plot([_x0, _x1], [_y0, _y1], 'k--', lw=1)
    axs[0].set_xlim(-_lim, _lim)
    axs[0].set_ylim(-_lim, _lim)
    axs[0].legend(frameon=False)
    axs[0].set_xlabel('Integrated voltage I')
    axs[0].set_ylabel('Integrated voltage Q')
    axs[0].set_title(f'IQ plot qubit {qubit}')
    # fig.suptitle(f'{timestamp}\n')
    ##########################
    # Plot projections
    ##########################
    # 01 projection
    _bin_c = projection_01['bin_centers']
    bin_width = _bin_c[1]-_bin_c[0]
    axs[1].bar(_bin_c, projection_01['h0'], bin_width, fc='C0', alpha=0.4)
    axs[1].bar(_bin_c, projection_01['h1'], bin_width, fc='C3', alpha=0.4)
    axs[1].plot(_bin_c, double_gauss(_bin_c, *projection_01['popt0']), '-C0')
    axs[1].plot(_bin_c, double_gauss(_bin_c, *projection_01['popt1']), '-C3')
    axs[1].axvline(projection_01['threshold'], ls='--', color='k', lw=1)
    text = '\n'.join((f'Fid.  : {projection_01["Fid"]*100:.1f}%',
                      f'SNR : {projection_01["SNR"]:.1f}'))
    props = dict(boxstyle='round', facecolor='gray', alpha=0)
    axs[1].text(.775, .9, text, transform=axs[1].transAxes,
                verticalalignment='top', bbox=props, fontsize=7)
    axs[1].text(projection_01['popt0'][0], projection_01['popt0'][4]/2,
                r'$|g\rangle$', ha='center', va='center', color='C0')
    axs[1].text(projection_01['popt1'][0], projection_01['popt1'][4]/2,
                r'$|e\rangle$', ha='center', va='center', color='C3')
    axs[1].set_xticklabels([])
    axs[1].set_xlim(_bin_c[0], _bin_c[-1])
    axs[1].set_ylim(bottom=0)
    axs[1].set_title('Projection of data')
    # 12 projection
    _bin_c = projection_12['bin_centers']
    bin_width = _bin_c[1]-_bin_c[0]
    axs[2].bar(_bin_c, projection_12['h1'], bin_width, fc='C3', alpha=0.4)
    axs[2].bar(_bin_c, projection_12['h2'], bin_width, fc='C2', alpha=0.4)
    axs[2].plot(_bin_c, double_gauss(_bin_c, *projection_12['popt1']), '-C3')
    axs[2].plot(_bin_c, double_gauss(_bin_c, *projection_12['popt2']), '-C2')
    axs[2].axvline(projection_12['threshold'], ls='--', color='k', lw=1)
    text = '\n'.join((f'Fid.  : {projection_12["Fid"]*100:.1f}%',
                      f'SNR : {projection_12["SNR"]:.1f}'))
    props = dict(boxstyle='round', facecolor='gray', alpha=0)
    axs[2].text(.775, .9, text, transform=axs[2].transAxes,
                verticalalignment='top', bbox=props, fontsize=7)
    axs[2].text(projection_12['popt1'][0], projection_12['popt1'][4]/2,
                r'$|e\rangle$', ha='center', va='center', color='C3')
    axs[2].text(projection_12['popt2'][0], projection_12['popt2'][4]/2,
                r'$|f\rangle$', ha='center', va='center', color='C2')
    axs[2].set_xticklabels([])
    axs[2].set_xlim(_bin_c[0], _bin_c[-1])
    axs[2].set_ylim(bottom=0)
    # 02 projection
    _bin_c = projection_02['bin_centers']
    bin_width = _bin_c[1]-_bin_c[0]
    axs[3].bar(_bin_c, projection_02['h0'], bin_width, fc='C0', alpha=0.4)
    axs[3].bar(_bin_c, projection_02['h2'], bin_width, fc='C2', alpha=0.4)
    axs[3].plot(_bin_c, double_gauss(_bin_c, *projection_02['popt0']), '-C0')
    axs[3].plot(_bin_c, double_gauss(_bin_c, *projection_02['popt2']), '-C2')
    axs[3].axvline(projection_02['threshold'], ls='--', color='k', lw=1)
    text = '\n'.join((f'Fid.  : {projection_02["Fid"]*100:.1f}%',
                      f'SNR : {projection_02["SNR"]:.1f}'))
    props = dict(boxstyle='round', facecolor='gray', alpha=0)
    axs[3].text(.775, .9, text, transform=axs[3].transAxes,
                verticalalignment='top', bbox=props, fontsize=7)
    axs[3].text(projection_02['popt0'][0], projection_02['popt0'][4]/2,
                r'$|g\rangle$', ha='center', va='center', color='C0')
    axs[3].text(projection_02['popt2'][0], projection_02['popt2'][4]/2,
                r'$|f\rangle$', ha='center', va='center', color='C2')
    axs[3].set_xticklabels([])
    axs[3].set_xlim(_bin_c[0], _bin_c[-1])
    axs[3].set_ylim(bottom=0)
    axs[3].set_xlabel('Integrated voltage')
    # Write fidelity textbox
    text = '\n'.join(('Assignment fidelity:',
                      f'$F_g$ : {Fid_dict["0"]*100:.1f}%',
                      f'$F_e$ : {Fid_dict["1"]*100:.1f}%',
                      f'$F_f$ : {Fid_dict["2"]*100:.1f}%',
                      f'$F_\mathrm{"{avg}"}$ : {Fid_dict["avg"]*100:.1f}%'))
    props = dict(boxstyle='round', facecolor='gray', alpha=.2)
    axs[1].text(1.05, 1, text, transform=axs[1].transAxes,
                verticalalignment='top', bbox=props)

def _plot_defect_rate_and_leakage(
    stabilizer,
    rounds,
    Defect_rate,
    Population_f,
    Defect_rate_f = None):
    
    Stabilizer_map = {'Z1': ['Z1', 'D1', 'D2', 'D4', 'D5'],
                      'Z2': ['Z2', 'D3', 'D6'],
                      'Z3': ['Z3', 'D4', 'D7'],
                      'Z4': ['Z4', 'D5', 'D6', 'D8', 'D9']}
    _qubits = Stabilizer_map[stabilizer]
    fig = plt.figure(figsize=(4,3))
    axs = [fig.add_subplot(111)]
    # Plot defect rate
    _max_rounds = list(Defect_rate[stabilizer][0].keys())[-1]
    _n_rounds =  int(list(Defect_rate[stabilizer][0].keys())[-1][:2])
    axs[0].plot(range(1,_n_rounds+2), Defect_rate[stabilizer][0][_max_rounds], 'C0.-', label='No LRU')
    axs[0].plot(range(1,_n_rounds+2), Defect_rate[stabilizer][1][_max_rounds], 'C1.-', label='With LRU')
    if Defect_rate_f:
        axs[0].plot(range(1,_n_rounds+2), Defect_rate_f[stabilizer][0][_max_rounds], 'C0.--')
        axs[0].plot(range(1,_n_rounds+2), Defect_rate_f[stabilizer][1][_max_rounds], 'C1.--')

    axs[0].set_ylim(0, .5)
    axs[0].legend(frameon=False, loc=4)
    axs[0].set_ylabel('Defect probability')
    axs[0].set_xlabel('Rounds')
    axs[0].set_title('Defect rate')
    axs[0].set_xticks([0, 5, 10, 15])
    # Plot leakage for each qubit
    lims = (0, 0)
    for i, q in enumerate(_qubits):
        axs.append(fig.add_subplot(121, label=f'axis {i}'))
        _pos = axs[-1].get_position()
        _pos = [_pos.x0+.95+i*.4, _pos.y0, _pos.width, _pos.height]
        axs[-1].set_position(_pos)
        if 'Z' in q:
            axs[-1].plot(range(1,_n_rounds+1), Population_f[0][q]*100, 'C0--', marker='.')
            axs[-1].plot(range(1,_n_rounds+1), Population_f[1][q]*100, 'C1-', marker='.' if i!=0 else '')
            axs[-1].set_xticks([0, 4, 8, 15])
        else:  
            axs[-1].plot(rounds, Population_f[0][q]*100, 'C0--', marker='.')
            axs[-1].plot(rounds, Population_f[1][q]*100, 'C1-', marker='.' if i!=0 else '')
            axs[-1].set_xticks(rounds)

        axs[-1].set_title(q)
        axs[-1].set_xlabel('Rounds')
        _lims = axs[-1].get_ylim()
        lims = min(lims[0], _lims[0]), max(lims[1], _lims[1])
    axs[1].set_ylabel('Leakage (%)')
    for i, q in enumerate(_qubits):
        axs[i+1].set_ylim(lims)
        if i>0:
            axs[i+1].set_yticklabels([])
    fn =  f"./Surface_13_defect_rate_{stabilizer}.png"
    fig.savefig(fn, dpi=300, bbox_inches='tight', format='png')
