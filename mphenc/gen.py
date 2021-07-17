#!/usr/bin/env python3
'''
Generate data
'''

import numpy

def rng(src=None):
    '''
    Wash a src of randomness into a random generator
    '''
    if src is None:
        return numpy.random.default_rng()
    if isinstance(src, int):
        return numpy.random.default_rng(src)
    return src

def rayleigh(domain, sigma=1):
    '''
    Return the Rayleigh probability density function on the domain. 
    '''
    isig2=1.0/(sigma*sigma)
    ret = isig2*domain*numpy.exp(-0.5*isig2*domain*domain)
    return ret / numpy.sum(ret)

def spectrum(waves):
    '''
    Return complex spectrum of given waves
    '''
    return numpy.fft.fft(waves)

def white(length, n=None, r=None):
    '''
    Genearate white noise waveforms of given length.

    See waveforms() for n an r.
    '''
    mean_spectrum = [1.0/length]*length
    return waveforms(mean_spectrum, n, r)

def pink(length, tsample=1, n=None, r=None):
    '''
    Genearate pink noise waveforms of given length.

    The tsample is sample time period.

    See waveforms() for n an r.
    '''
    maxf=1.0/tsample
    df = maxf/length
    mean_spectrum = [1/(df*(i+1)) for i in range(length)]
    mean_spectrum /= numpy.sum(mean_spectrum)

    return waveforms(mean_spectrum, n, r)


def waveforms(mean_spectrum, n=None, r = None):
    '''
    Generate waveforms according to mean spectrum.

    If n is given, return array of shape (n, l) of n waveforms each of
    length l, that of the len of the mean_spectrum.  If n is not
    given, a single waveform will be returned.

    A random generator or seed may be given with r.
    '''
    r = rng(r)

    nwave = 1 if n is None else n
    shape = (nwave, len(mean_spectrum))

    amp = r.rayleigh(mean_spectrum, shape)
    phi = r.uniform(0.0, 2.0*numpy.pi, shape)
    spec = amp * numpy.exp(1j*phi)
    cwaves = numpy.fft.ifft(spec)
    waves = numpy.real(cwaves)
    if n is None:
        return waves[0]
    return waves
    
def noise(n=None, r=None):
    nticks = 1000
    tsamp = 0.5e-6
    maxfreq = 1.0/tsamp
    freqpeak = maxfreq/10.0
    baseline = 2**11

    tdom = numpy.arange(0, tsamp*nticks, tsamp)
    fdom = numpy.arange(0, maxfreq, maxfreq/nticks)

    nspec = 1e6*rayleigh(fdom, freqpeak)
    
    nwf = waveforms(nspec, n, r)
    nwfadc = baseline + numpy.array(nwf, dtype=int)
    return nwfadc, tdom

def plot2d(wf):

    import matplotlib.pyplot as plt
    plt.clf()
    plt.imshow(wf, aspect='auto')
    plt.colorbar()

def test2d():
    x = numpy.arange(0,1000,.1)
    y = 1e6*rayleigh(x, 100)
    z = numpy.floor(10*waveforms(y, 100))
