import numpy as np
import xarray_behave as xb
import itertools
import matplotlib.pyplot as plt

# for glm
from glm_utils.preprocessing import time_delay_embedding
from tempfile import TemporaryDirectory
import scipy.stats as stat
from sklearn import linear_model as lm    # Ridge
import sklearn.model_selection as sms    # train_test_split
import sklearn.metrics as skmet    # classification_report
from sklearn.pipeline import Pipeline

abs_feature_names = ['angles', 'rotational_speed', 'velocity_forward', 'velocity_lateral', 'acceleration_mag']


def assemble_datasets(datename, time_step, root="Z:/#Common", expsetup='backlight', target_sampling_rate=1000):
    dataset = xb.assemble(datename, root=root+f'/{expsetup}', target_sampling_rate=target_sampling_rate)
    metrics_dataset = xb.assemble_metrics(dataset)
    dstimes = dataset.nearest_frame.time[::int(time_step*target_sampling_rate/1000)].values
    return metrics_dataset, dstimes


def chain_analysis(metrics_dataset, dist_thr=90, or_thr=100, ang_thr=50, verbose=False):

    # Load data
    if verbose:
        print(f"   loading specific metrics...")
    rel_orientations = metrics_dataset.rel_features.sel(relative_features='relative_orientation')
    dist = metrics_dataset.rel_features.sel(relative_features='distance')
    rel_angle = metrics_dataset.rel_features.sel(relative_features='relative_angle')

    # Correct rel_orientation to be -180 to 180
    rel_orientations = rel_orientations - ((rel_orientations+180)//360)*360

    # Deduce variables from data dimensions
    nflies = metrics_dataset.flies.size
    nframes = metrics_dataset.time.size

    # Create chain matrices based on conditions
    if verbose:
        print(f"   calculating chain matrices...")
    chain_matrices = np.ones((nframes, nflies, nflies))
    chain_matrices = np.multiply(chain_matrices, dist < dist_thr)
    chain_matrices = np.multiply(chain_matrices, np.abs(rel_angle) < ang_thr)
    chain_matrices = np.multiply(chain_matrices, np.abs(rel_orientations) < or_thr)
    chain_matrices = chain_matrices.astype(bool)

    if verbose:
        print(f"   done")

    return chain_matrices


def get_chain_groups(chain_matrix):
    """Goes through the chain_matrix (the element-wise multiplication of all matrices with parameters
    for chain detection), and checks for common connected members, to find chains (or groups).

    Returns a dictionary with the members of each chain (group)."""

    # get nonzero points from "image"
    points = np.unravel_index(np.flatnonzero(chain_matrix), chain_matrix.shape)
    chain_pairs = np.vstack(points).T.astype(np.int)

    groups_dict = {}
    nn = 0
    for ii, jj in chain_pairs:
        found_group = False
        for groupnn in range(nn):
            if ii in groups_dict[f"t{groupnn}"]:
                found_group = True
                if jj not in groups_dict[f"t{groupnn}"]:
                    groups_dict[f"t{groupnn}"].append(jj)
            elif jj in groups_dict[f"t{groupnn}"]:
                found_group = True
                groups_dict[f"t{groupnn}"].append(ii)
        if not found_group:
            groups_dict[f"t{nn}"] = [ii, jj]
            nn += 1

    removed_key = []
    for a, b in itertools.combinations(groups_dict.keys(), 2):
        if not set(groups_dict[a]).isdisjoint(groups_dict[b]):
            groups_dict[a] = list(set().union(groups_dict[a], groups_dict[b]))
            groups_dict[b] = []
            removed_key.append(b)

    for rk in removed_key:
        del groups_dict[rk]

    nn = 0
    for k in sorted(groups_dict, key=lambda k: len(groups_dict[k]), reverse=True):
        groups_dict[nn] = groups_dict.pop(k)
        nn += 1

    return groups_dict


def triad_analysis(dstimes, chain_matrices, stepsize=1):

    triads_data = {}
    my_chain_matrices = chain_matrices.loc[dstimes, ...].values

    for ii, chain_matrix in enumerate(my_chain_matrices):
        chgroups = get_chain_groups(chain_matrix)
        group_lengths = np.asarray([len(chgroups[gg]) for gg in chgroups])
        for ngg in np.where(group_lengths > 2)[0]:
            for subset in itertools.permutations(chgroups[ngg], 3):
                if chain_matrix[subset[0], subset[1]]*chain_matrix[subset[1], subset[2]] == 1:
                    if f"{subset[0]:02d}{subset[1]:02d}{subset[2]:02d}" in triads_data:
                        triads_data[f"{subset[0]:02d}{subset[1]:02d}{subset[2]:02d}"].append(ii)
                    else:
                        triads_data[f"{subset[0]:02d}{subset[1]:02d}{subset[2]:02d}"] = [ii]

    segmented_triads = {}
    for key in triads_data.keys():
        consecutive_triads = np.split(triads_data[key], np.where(np.diff(triads_data[key]) > stepsize)[0] + 1)
        for nt, ct in enumerate(consecutive_triads):
            segmented_triads[f"{key}_{nt:02d}"] = list(ct)

    return triads_data, segmented_triads


def collect_simple(datename, f1, f2, basis, selected_flies=(0, 1), root="Z:/#Common", expsetup='backlight', time_step=100, target_sampling_rate=50):

    w, ncos = basis.shape
    metrics_dataset, dstimes = assemble_datasets(datename, time_step, root=root, expsetup=expsetup, target_sampling_rate=target_sampling_rate)

    # chaining detection
    chain_matrices = chain_analysis(metrics_dataset)

    # triads
    triads_data, segmented_triads = triad_analysis(dstimes, chain_matrices)

    print(f'{f1} vs {f2}')

    all_AA = []
    all_BB = []

    for key in segmented_triads.keys():

        if len(segmented_triads[key]) > 1:

            # get flies id
            IDlist = [int(key[ikey*2:2+ikey*2]) for ikey in range(3)]
            ii = IDlist[selected_flies[0]]    # predicted
            jj = IDlist[selected_flies[1]]    # predictor

            # get times
            times = [dstimes[idx] for idx in segmented_triads[key]]

            # get feature data (A is to predict, B what will be used for the prediction)
            A = xb.metrics.remove_nan(metrics_dataset.abs_features.sel(absolute_features=f1, flies=ii))
            B = xb.metrics.remove_nan(metrics_dataset.abs_features.sel(absolute_features=f2, flies=jj))

            # find times for time-slicing the data
            t0 = int(np.where(A.time == times[0])[0])
            t1 = int(np.where(A.time == times[-1])[0])

            # time slice the data
            AA = np.ascontiguousarray(A.values)[t0:t1, ...]
            BB = np.dot(time_delay_embedding(x=np.ascontiguousarray(B.values), window_size=w), basis)[t0:t1, ...]

            # append
            all_AA.extend(AA)
            all_BB.extend(BB)

    # concatenate data
    all_AA_array = stat.zscore(np.stack(all_AA, axis=0))
    all_BB_array = np.stack(all_BB, axis=0)

    stds = np.std(all_BB_array, axis=0, ddof=1)
    whitener = np.diag(1/stds)
    all_BB_array = np.dot(all_BB_array, whitener)

    Xs = all_BB_array
    y = all_AA_array

    return Xs, y


def collect_multiple(list_datename, f1, f2, basis, selected_flies=(0, 1), root="Z:/#Common", expsetup='backlight', time_step=100, target_sampling_rate=50):
    all_Xs = np.empty((0, basis.shape[1]))
    all_y = np.empty((0,))
    for ii, datename in enumerate(list_datename):
        temp_Xs, temp_y = collect_simple(datename=datename, f1=f1, f2=f2, basis=basis, selected_flies=selected_flies, root=root, expsetup=expsetup, time_step=time_step, target_sampling_rate=target_sampling_rate)
        all_Xs = np.concatenate((all_Xs, temp_Xs), axis=0)
        all_y = np.concatenate((all_y, temp_y))

    print(f"collected Xs: {all_Xs.shape}, collected y: {all_y.shape}")

    return all_Xs, all_y


def recon_filters(coefficients, basis):
    """Reconstruct time filter from linear regression model.

    Args:
        coefficients (numpy array): Coefficients of the regression model (`model.coef_` or for pipelines `model[-1].coef_`).
        basis (numpy array): Basis matrix used for basis projection. [time, nvectors]

    Returns:
        time_filters (numpy array): Time filter of the model. [nfeature, time]
    """

    w, nbasis = basis.shape
    nfeatures = coefficients.shape[0]//nbasis    # finds number of features comparing number of coefficients and basis dimension

    time_filters = np.empty((nfeatures, w))
    for ii in range(nfeatures):
        time_filters[ii, :] = np.matmul(basis, coefficients[ii*nbasis:(ii+1)*nbasis])

    return time_filters


def try_lasso(X_train, X_test, y_train, y_test, basis, ncv=10, rand_state=42, max_iter=5000):
    Lasso_steps = [('lassoCV', lm.LassoCV(cv=ncv, random_state=rand_state, max_iter=5000))]
    with TemporaryDirectory() as tempdir:
        clf = Pipeline(Lasso_steps, memory=tempdir)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(f'train r2={clf.score(X_train, y_train):1.2}')
        print(f'test r2={clf.score(X_test, y_test):1.2}')
        print(f'explained_variance_score ={skmet.explained_variance_score(y_test, y_pred):1.2}')
        print(f'mean_squared_error ={skmet.mean_squared_error(y_test, y_pred):1.2}')
        # print(np.abs(clf.named_steps['lassoCV'].coef_))
        # print(np.abs(clf.named_steps['lassoCV'].coef_).argsort()[::-1])
        recon = recon_filters(clf.named_steps['lassoCV'].coef_, basis)
        plt.plot(np.squeeze(recon).T)
        plt.show()
    return np.squeeze(recon).T


def compare_flies(list_datename, basis, f1='velocity_forward', f2='velocity_forward', time_step=100, root="Z:/#Common", expsetup='backlight', target_sampling_rate=50, lazy=False):

    Xs1, y = collect_multiple(list_datename, f1=f1, f2=f2, basis=basis, selected_flies=(0, 1), root=root, expsetup=expsetup, time_step=time_step, target_sampling_rate=target_sampling_rate)
    Xs2, _ = collect_multiple(list_datename, f1=f1, f2=f2, basis=basis, selected_flies=(0, 2), root=root, expsetup=expsetup, time_step=time_step, target_sampling_rate=target_sampling_rate)
    merged_Xs = np.concatenate((Xs1, Xs2), axis=1)
    X_train, X_test, y_train, y_test = sms.train_test_split(merged_Xs, y, test_size=0.3, random_state=42)

    nbasis = basis.shape[1]

    recons = []
    print(f"\n ====================== {f2} ====================== \n")
    print(f"\n ---------------------- mixed ---------------------- \n")
    recon = try_lasso(X_train, X_test, y_train, y_test, basis)
    recons.append(recon)
    print(f"\n ---------------------- fly 1 ---------------------- \n")
    recon = try_lasso(X_train[:, :nbasis], X_test[:, :nbasis], y_train, y_test, basis)
    recons.append(recon)
    print(f"\n ---------------------- fly 2 ---------------------- \n")
    recon = try_lasso(X_train[:, nbasis:], X_test[:, nbasis:], y_train, y_test, basis)
    recons.append(recon)

    return recons


def new_triad_analysis(dstimes, chain_matrices, stepsize=1, min_ele_size=2):

    triads_data = {}
    my_chain_matrices = chain_matrices.loc[dstimes, ...].values

    for ii, chain_matrix in enumerate(my_chain_matrices):
        chgroups = get_chain_groups(chain_matrix)
        group_lengths = np.asarray([len(chgroups[gg]) for gg in chgroups])
        for ngg in np.where(group_lengths > 2)[0]:
            for subset in itertools.permutations(chgroups[ngg], 3):
                if chain_matrix[subset[0], subset[1]]*chain_matrix[subset[1], subset[2]] == 1:
                    if f"{subset[0]:02d}{subset[1]:02d}{subset[2]:02d}" in triads_data:
                        triads_data[f"{subset[0]:02d}{subset[1]:02d}{subset[2]:02d}"].append(ii)
                    else:
                        triads_data[f"{subset[0]:02d}{subset[1]:02d}{subset[2]:02d}"] = [ii]

    segmented_triads = {}
    for key in triads_data.keys():
        consecutive_triads = np.split(triads_data[key], np.where(np.diff(triads_data[key]) > stepsize)[0] + 1)
        consecutive_triads = [ele for ele in consecutive_triads if ele.size >= min_ele_size]
        for nt, ct in enumerate(consecutive_triads):
            segmented_triads[f"{key}_{nt:02d}"] = [dstimes[ct[0]], dstimes[ct[-1]]]
    return triads_data, segmented_triads
