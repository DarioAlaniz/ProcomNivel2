import numpy as np
import matplotlib.pyplot as plt
import time

def get_filter(name, T, rolloff=None,amplitude=1,norm=False):
    def rc(t, beta,amplitude):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return amplitude*(np.sinc(t)*np.cos(np.pi*beta*t)/(1-(2*beta*t)**2))
    def rrc(t, beta,amplitude):
        return amplitude*(np.sin(np.pi*t*(1-beta))+4*beta*t*np.cos(np.pi*t*(1+beta)))/(np.pi*t*(1-(4*beta*t)**2))
        
    # rolloff is ignored for triang and rect
    if name == 'rect':
        return lambda t: (abs(t/T)<0.5).astype(int)    
    if name == 'triang': 
        return lambda t: (1-abs(t/T)) * (abs(t/T)<1).astype(float)
    elif name == 'rc':
        if norm: return lambda t: rc(t/T, rolloff,amplitude)/np.sqrt(np.sum(rc(t/T, rolloff,amplitude)**2)) 
        else :   return lambda t: rc(t/T, rolloff,amplitude)
    elif name == 'rrc':
        if norm: return lambda t: rrc(t/T, rolloff,amplitude)/np.sqrt(np.sum(rrc(t/T, rolloff,amplitude)**2))
        else :   return lambda t: rrc(t/T, rolloff,amplitude)


def get_signal(g, d):
    """Generate the transmit signal as sum(d[k]*g(t-kT))"""
    t = np.arange(-2*T, (len(d)+2)*T, 1/Fs)
    # g0 = g(np.array([1e-9]))
    xt = sum(d[k]*g(t-k*T) for k in range(len(d)))
    return t, xt

def drawFullEyeDiagram(xt):
    """Draw the eye diagram using all parts of the given signal xt"""
    samples_perT = Fs*T
    samples_perWindow = 2*Fs*T
    parts = []
    startInd = 2*samples_perT   # ignore some transient effects at beginning of signal
    
    for k in range(int(len(xt)/samples_perT) - 6):
        parts.append(xt[startInd + k*samples_perT + np.arange(samples_perWindow)])
    parts = np.array(parts).T
    
    t_part = np.arange(-T, T, 1/Fs)
    plt.plot(t_part, parts, 'b-')

# # function to calculate the spectrum of the input signal
spec = lambda x: abs(np.fft.fftshift(np.fft.fft(x, 4*len(t))))/Fs

T = 1
Fs = 100
Nbaud = 1
oversampling = 1
t = np.arange(-0.5*T*Nbaud, 0.5*T*Nbaud, 1/(Fs*oversampling))
