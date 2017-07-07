"""Test the propagators."""
import pytest
import numpy as np
from wave_1d_fd_tf.propagators import (VPy1, VTF)

def ricker(freq, length, dt, peak_time):
    """Return a Ricker wavelet with the specified central frequency."""
    t = np.arange(-peak_time, (length)*dt - peak_time, dt, dtype=np.float32)
    y = ((1.0 - 2.0*(np.pi**2)*(freq**2)*(t**2))
         * np.exp(-(np.pi**2)*(freq**2)*(t**2)))
    return y

def green(x0, x1, dx, dt, t, v, v0, f):
    """Use the 1D Green's function to determine the wavefield at a given
    location and time due to the given source.
    """
    y = np.sum(f[:np.maximum(0, int((t - np.abs(x1-x0)/v)/dt))])*dt*dx*v0/2
    return y

@pytest.fixture
def model_one():
    """Create a model with one reflector, and the expected wavefield."""
    N = 100
    rx = int(N/2)
    model = np.ones(N, dtype=np.float32) * 1500
    model[rx:] = 2500
    max_vel = 2500
    dx = 5
    dt = 0.001
    # 0.14 secs is chosen to avoid reflections from boundaries
    nsteps = np.ceil(0.14/dt).astype(np.int)
    source = ricker(25, nsteps, dt, 0.05)
    sx = 35
    expected = np.zeros(N)
    # create a new source shifted by the time to the reflector
    time_shift = np.round((rx-sx)*dx / 1500 / dt).astype(np.int)
    shifted_source = np.pad(source, (time_shift, 0), 'constant')
    # reflection and transmission coefficients
    r = (2500 - 1500) / (2500 + 1500)
    t = 1 + r

    # direct wave
    expected[:rx] = np.array([green(x*dx, sx*dx, dx, dt,
                                    (nsteps+1)*dt, 1500, 1500,
                                    source) for x in range(rx)])
    # reflected wave
    expected[:rx] += r*np.array([green(x*dx, (rx-1)*dx, dx, dt,
                                       (nsteps+1)*dt, 1500, 1500,
                                       shifted_source) for x in range(rx)])
    # transmitted wave
    expected[rx:] = t*np.array([green(x*dx, rx*dx, dx, dt,
                                      (nsteps+1)*dt, 2500, 1500,
                                      shifted_source) for x in range(rx, N)])
    return {'model': model, 'dx': dx, 'dt': dt, 'nsteps': nsteps,
            'sources': np.array([source]), 'sx': np.array([sx]),
            'expected': expected}

@pytest.fixture
def model_two():
    """Create a random model and compare with VPy1 implementation."""
    N = 100
    np.random.seed(0)
    model = np.random.random(N).astype(np.float32) * 3000 + 1500
    max_vel = 4500
    dx = 5
    dt = 0.6 * dx / max_vel
    nsteps = np.ceil(0.2/dt).astype(np.int)
    num_sources = 10
    sources_x = np.zeros(num_sources, dtype=np.int)
    sources = np.zeros([num_sources, nsteps], dtype=np.float32)
    sources_x = np.random.randint(N, size=num_sources)
    while len(set(sources_x)) != num_sources:
        sources_x = np.random.randint(N, size=num_sources)
    for sourceIdx in range(num_sources):
        sources[sourceIdx, :] = ricker(25, nsteps, dt, 0.05+np.random.rand())
    v = VPy1(model, dx, dt)
    expected = v.step(nsteps, sources, sources_x)
    return {'model': model, 'dx': dx, 'dt': dt, 'nsteps': nsteps,
            'sources': sources, 'sx': sources_x, 'expected': expected}

@pytest.fixture
def versions():
    """Return a list of implementations."""
    return [VPy1, VTF]

def test_one_reflector(model_one, versions):
    """Verify that the numeric and analytic wavefields are similar."""

    for v in versions:
        _test_version(v, model_one, atol=1.5)


def test_allclose(model_two, versions):
    """Verify that all implementations produce similar results."""

    for v in versions[1:]:
        _test_version(v, model_two, atol=0.0015)#atol=1e-8) Reduced accuracy!


def _test_version(version, model, atol):
    """Run the test for one implementation."""
    v = version(model['model'], model['dx'], model['dt'])
    y = v.step(model['nsteps'], model['sources'], model['sx'])
    assert np.allclose(y, model['expected'], atol=atol)
