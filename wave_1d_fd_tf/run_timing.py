"""Measure the runtime of the propagators."""
from timeit import repeat
import numpy as np
import pandas as pd
from wave_1d_fd_tf.propagators import (VTF)
from wave_1d_fd_tf.test_wave_1d_fd_tf import ricker

def run_timing_num_steps(num_repeat=10, num_steps=range(0, 110, 10), model_size=1000):
    """Time implementations as num_steps varies."""

    versions = _versions()

    times = pd.DataFrame(columns=['version', 'num_steps', 'model_size', 'time'])

    for nsteps in num_steps:
        model = _make_model(model_size, nsteps)
        times = _time_versions(versions, model, num_repeat, times)

    return times


def run_timing_model_size(num_repeat=10, num_steps=20, model_sizes=range(200, 2200, 200)):
    """Time implementations as model size varies."""

    versions = _versions()

    times = pd.DataFrame(columns=['version', 'num_steps', 'model_size', 'time'])

    for N in model_sizes:
        model = _make_model(N, num_steps)
        times = _time_versions(versions, model, num_repeat, times)

    return times


def _versions():
    """Return a list of versions to be timed."""
    return [{'class': VTF, 'name': 'TensorFlow'}]


def _make_model(N, nsteps):
    """Create a model with a given number of elements and time steps."""
    model = np.random.random(N).astype(np.float32) * 3000 + 1500
    max_vel = 4500
    dx = 5
    dt = 0.6 * dx / max_vel
    source = ricker(25, nsteps, dt, 0.05)
    sx = int(N/2)
    return {'model': model, 'dx': dx, 'dt': dt, 'nsteps': nsteps,
            'sources': np.array([source]), 'sx': np.array([sx])}


def _time_versions(versions, model, num_repeat, dataframe):
    """Loop over versions and append the timing results to the dataframe."""
    num_steps = model['nsteps']
    model_size = len(model['model'])
    for v in versions:
        time = _time_version(v['class'], model, num_repeat)
        dataframe = dataframe.append({'version': v['name'],
                                      'num_steps': num_steps,
                                      'model_size': model_size,
                                      'time': time}, ignore_index=True)
    return dataframe


def _time_version(version, model, num_repeat):
    """Time a particular version."""
    v = version(model['model'], model['dx'], model['dt'])

    def closure():
        """Closure over variables so they can be used in repeat below."""
        v.step(model['nsteps'], model['sources'], model['sx'])
 
    return np.min(repeat(closure, number=1, repeat=num_repeat))

if __name__ == '__main__':
    print(run_timing_num_steps())
