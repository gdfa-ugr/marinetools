import numpy as np
import scipy.signal as scs
import matplotlib.pyplot as plt
import pandas as pd
# from utide import solve, reconstruct
from ..utils import auxiliar, save



def lombscargle_spectra(data, var_, fname=None, max_period=None, ref=1.1, nperiods=5):
    """ Compute the LombScargle Periodogram for uneven sampling

    Args:
        * data: pd.DataFrame with the timeseries
        * var_: variable
        * fname: name of the output file with the power spectral density and frequencies
        * max_period: maximum value to analyse the diferent time periods
        * ref: maximum value to look for the main time periods
        * nperiods: number of main periods
    
    Returns:
        The periodogram and the significant frequencies
    """

    time = (data.index - data.index[0]).total_seconds()/(365.25*24*3600)
    periods = np.linspace(24 * 7 / (24 * 365.25), 2, 1000)
    if max_period is not None:
        periods = np.hstack([periods, np.arange(2.01, max_period, 0.02)])

    freqs = 1.0/periods
    w = 2*np.pi*freqs

    pgram = scs.lombscargle(time, data[var_].values, w, normalize=True)
    pgram = pd.Series(pgram, index=periods)
    signf = auxiliar.moving(pgram, 100)
    signf.columns, signf.index.name = ['PSD'], 'periods'
    signf = signf.loc[signf.index < ref]
    signf = signf.nlargest(nperiods, 'PSD')

    if not fname:
        print(signf)
    else:
        signf.to_excel(fname + '.xlsx')

    return pgram, signf


def fft(data, ci=False):
    """Compute the Fast-Fourier Transform for regular sampling
    """
    
    N = len(data)
    df = 5/(data.index[-1] - data.index[0]).days
    G = np.fft.fft(data)

    # Se selecciona una hoja del espectro
    ent_real = int(np.mod(N/2, 1))
    if ent_real == 0:
        cn = G[1:int(N/2)]/N
    else:
        cn = G[1:int(N+1)/2]/N

    an, bn = 2*np.real(cn), -2*np.imag(cn)
    N_fft = len(cn)
    f, S = np.arange(0, N_fft)*df,  0.5*np.sqrt(an**2.+bn**2.)**2./df
    plt.semilogy(f/5, S, color='black', label=PARAM['nfile'])

    if not ci:
        var = data.var()
        print(var)
        # degrees of freedom
        M = N/2
        phi = (2*(N-1)-M/2.)/M
        # values of chi-squared
        chi_val_99 = chi2.isf(q=1-0.01/2, df=phi)  # /2 for two-sided test
        # chi_val_95 = chi2.isf(q=1-0.05/2, df=phi)
        plt.semilogy(f/5, (var/N)*(chi_val_99/phi)/(df*f**2), color='gray',
                    linestyle='--', label='Red-noise')  # ruido rojo - f**2, ruido rosa - f
        Sf = S[S > (var/N)*(chi_val_99/phi)/(df*f**2)]
        ff = f[S > (var/N)*(chi_val_99/phi)/(df*f**2)]
        plt.semilogy(ff/5, Sf, '.', color='steelblue',
                    label='Significant frequencies')
        # plt.semilogy(f, (var/N)*(chi_val_95/phi)/(df*f**2),color='0.4',linestyle='--')
        ax.set_xlabel(r'Freqs. ($\mathbf{d^{-1}}$)', fontweight='bold')
        ax.set_ylabel(r'PSD (mm$\mathbf{^{2}}$d)', fontweight='bold')
    plt.show()
    return


# def harmonic():
#     nivel.dropna(inplace=True)
#     i = nivel.index[0]
#     aux = nivel.loc[i:i + timedelta(PARAM['wlen'])].values
#     time = mdates.date2num(nivel.loc[i:i + timedelta(PARAM['wlen'])].index.to_pydatetime())
#     print(i)
#     # print(aux)

#     if len(aux) >= 0.9*PARAM['wlen']*24:
#         out = solve(time, aux,
#             lat=lat,
#             nodal=True,
#             trend=False,
#             method='ols',
#             conf_int='linear',
#             Rayleigh_min=0.95,
#             verbose=False)
#         if len(out['A']) == 35:
#             t_name.append(i+timedelta(PARAM['step']/2.))
#             amp.append(out['A'])
#             amp_e.append(out['A_ci'])
#             pha.append(out['g'])
#             pha_e.append(out['g_ci'])
#             comp_name = [i for i in out['name']]
#     return


