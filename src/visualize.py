#!python3

"""plots for the milestone report"""

import h5py
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import gridspec


def plot_df_single_color(data, variables, labels, size=12, labelsize=17, name=None):
    """
    """
    plt.clf()
    input_dim = len(variables)
    cols = min(np.floor(input_dim ** 0.5).astype(int), 4)
    rows = (np.ceil(input_dim / cols)).astype(int)
    gs = gridspec.GridSpec(rows, cols)
    fig = plt.figure(figsize=(size, max(size, rows * 2)))

    for n in range(input_dim):
        ax = fig.add_subplot(gs[n])
        ax.plot(data[variables[n]], marker='.', markerfacecolor='none', alpha=0.7)
        ax.tick_params(axis='x', labelsize=labelsize)
        ax.tick_params(axis='y', labelsize=labelsize)
        plt.ylabel(labels[n], fontsize=labelsize)
        plt.xlabel('Time [s]', fontsize=labelsize)
    plt.tight_layout()
    if name is not None:
        plt.savefig(name, format='png', dpi=300)
    plt.show()
    plt.close()


def plot_df_color_per_unit(data, variables, labels, size=(16, 9), labelsize=17, option='Time', name=None):
    """
    """
    plt.clf()
    input_dim = len(variables)
    # cols = min(np.floor(input_dim ** 0.5).astype(int), 4)
    # rows = (np.ceil(input_dim / cols)).astype(int)
    cols = 4
    rows = 1
    gs = gridspec.GridSpec(rows, cols)
    leg = []
    # fig = plt.figure(figsize=(size, max(size, rows * 2)))
    fig = plt.figure(figsize=size)
    color_dic_unit = {'Unit 1': 'C0', 'Unit 2': 'C1', 'Unit 3': 'C2', 'Unit 4': 'C3', 'Unit 5': 'C4', 'Unit 6': 'C5',
                      'Unit 7': 'C6', 'Unit 8': 'C7', 'Unit 9': 'C8', 'Unit 10': 'C9', 'Unit 11': 'C10',
                      'Unit 12': 'C11', 'Unit 13': 'C12', 'Unit 14': 'C13', 'Unit 15': 'C14', 'Unit 16': 'C15',
                      'Unit 17': 'C16', 'Unit 18': 'C17', 'Unit 19': 'C18', 'Unit 20': 'C19'}

    unit_sel = np.unique(data['unit'])
    for n in range(input_dim):
        ax = fig.add_subplot(gs[n])
        for j in unit_sel:
            data_unit = data.loc[data['unit'] == j]
            if option == 'cycle':
                time_s = data.loc[data['unit'] == j, 'cycle']
                label_x = 'Time [cycle]'
            else:
                time_s = np.arange(len(data_unit))
                label_x = 'Time [s]'
            ax.plot(time_s, data_unit[variables[n]], '-o', color=color_dic_unit['Unit ' + str(int(j))],
                    alpha=0.7, markersize=5)
            ax.tick_params(axis='x', labelsize=labelsize)
            ax.tick_params(axis='y', labelsize=labelsize)
            leg.append('Unit ' + str(int(j)))
        plt.ylabel(labels[n], fontsize=labelsize)
        plt.xlabel(label_x, fontsize=labelsize)
        ax.get_xaxis().set_major_formatter(
            mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        if n == 0:
            ax.get_yaxis().set_major_formatter(
                mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.legend(leg, loc='best', fontsize=labelsize - 2)  # lower left
    plt.tight_layout()
    if name is not None:
        plt.savefig(name, format='png', dpi=300)
    plt.show()
    plt.close()


def plot_kde(leg, variables, labels, size, units, df_W, df_A, labelsize=17, name=None):
    """
    """
    plt.clf()

    input_dim = len(variables)
    # cols = min(np.floor(input_dim ** 0.5).astype(int), 4)
    # rows = (np.ceil(input_dim / cols)).astype(int)
    cols = 4
    rows = 1
    gs = gridspec.GridSpec(rows, cols)

    color_dic_unit = {'Unit 1': 'C0', 'Unit 2': 'C1', 'Unit 3': 'C2', 'Unit 4': 'C3', 'Unit 5': 'C4', 'Unit 6': 'C5',
                      'Unit 7': 'C6', 'Unit 8': 'C7', 'Unit 9': 'C8', 'Unit 10': 'C9', 'Unit 11': 'C10',
                      'Unit 12': 'C11', 'Unit 13': 'C12', 'Unit 14': 'C13', 'Unit 15': 'C14', 'Unit 16': 'C15',
                      'Unit 17': 'C16', 'Unit 18': 'C17', 'Unit 19': 'C18', 'Unit 20': 'C19'}

    # fig = plt.figure(figsize=(size, max(size, rows * 2)))
    fig = plt.figure(figsize=size)

    for n in range(input_dim):
        ax = fig.add_subplot(gs[n])
        for k, elem in enumerate(units):
            sns.kdeplot(df_W.loc[df_A['unit'] == elem, variables[n]],
                        color=color_dic_unit[leg[k]], shade=True, gridsize=100)
            ax.tick_params(axis='x', labelsize=labelsize)
            ax.tick_params(axis='y', labelsize=labelsize)

        ax.get_xaxis().set_major_formatter(
            mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        plt.xlabel(labels[n], fontsize=labelsize)
        plt.ylabel('Density [-]', fontsize=labelsize)
        if n == 0:
            plt.legend(leg, fontsize=labelsize - 4, loc=0)
        else:
            plt.legend(leg, fontsize=labelsize - 4, loc=2)
    plt.tight_layout()
    if name is not None:
        plt.savefig(name, format='png', dpi=300)
    plt.show()
    plt.close()


if __name__ == '__main__':
    filename = '../../data_set/N-CMAPSS_DS02-006.h5'

    with h5py.File(filename, 'r') as hdf:
        # Development set
        W_dev = np.array(hdf.get('W_dev'))  # W
        X_s_dev = np.array(hdf.get('X_s_dev'))  # X_s
        X_v_dev = np.array(hdf.get('X_v_dev'))  # X_v
        T_dev = np.array(hdf.get('T_dev'))  # T
        Y_dev = np.array(hdf.get('Y_dev'))  # RUL
        A_dev = np.array(hdf.get('A_dev'))  # Auxiliary

        # Test set
        W_test = np.array(hdf.get('W_test'))  # W
        X_s_test = np.array(hdf.get('X_s_test'))  # X_s
        X_v_test = np.array(hdf.get('X_v_test'))  # X_v
        T_test = np.array(hdf.get('T_test'))  # T
        Y_test = np.array(hdf.get('Y_test'))  # RUL
        A_test = np.array(hdf.get('A_test'))  # Auxiliary

        # Varnams
        W_var = np.array(hdf.get('W_var'))
        X_s_var = np.array(hdf.get('X_s_var'))
        X_v_var = np.array(hdf.get('X_v_var'))
        T_var = np.array(hdf.get('T_var'))
        A_var = np.array(hdf.get('A_var'))

        # from np.array to list dtype U4/U5
        W_var = list(np.array(W_var, dtype='U20'))
        X_s_var = list(np.array(X_s_var, dtype='U20'))
        X_v_var = list(np.array(X_v_var, dtype='U20'))
        T_var = list(np.array(T_var, dtype='U20'))
        A_var = list(np.array(A_var, dtype='U20'))

    W = np.concatenate((W_dev, W_test), axis=0)
    X_s = np.concatenate((X_s_dev, X_s_test), axis=0)
    X_v = np.concatenate((X_v_dev, X_v_test), axis=0)
    T = np.concatenate((T_dev, T_test), axis=0)
    Y = np.concatenate((Y_dev, Y_test), axis=0)
    A = np.concatenate((A_dev, A_test), axis=0)

    print("W shape: " + str(W.shape))
    print("X_s shape: " + str(X_s.shape))
    print("X_v shape: " + str(X_v.shape))
    print("T shape: " + str(T.shape))
    print("A shape: " + str(A.shape))

    df_A = pd.DataFrame(data=A, columns=A_var)
    df_W = pd.DataFrame(data=W, columns=W_var)
    df_W['unit'] = df_A['unit'].values

    df_W_u = df_W.loc[(df_A.unit == 20) & (df_A.cycle == 1)]
    df_W_u.reset_index(inplace=True, drop=True)

    labels = ['Altitude [ft]', 'Mach Number [-]', 'Throttle Resolver Angle [%]', 'Temperature at fan inlet (T2) [°R]']
    plot_df_color_per_unit(df_W_u, W_var, labels, size=(24, 6), labelsize=19, name='flight_profile.png')

    variables = ['alt', 'Mach', 'TRA', 'T2']
    labels = ['Altitude [ft]', 'Mach Number [-]', 'Throttle Resolver Angle [%]', 'Temperature at fan inlet [°R]']
    size = (24, 6)

    units = list(np.unique(df_A['unit']))
    leg = ['Unit ' + str(int(u)) for u in units]

    plot_kde(leg, variables, labels, size, units, df_W, df_A, labelsize=19, name='kde.png')
