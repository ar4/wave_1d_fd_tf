"""1D finite difference wave propagation implemented using TensorFlow
"""
import numpy as np
import tensorflow as tf

class Propagator(object):
    """An 8th order finite difference propagator for the 1D wave equation."""
    def __init__(self, model, dx, dt=None, npad=8):
        self.nx = len(model)
        self.dx = np.float32(dx)
        max_vel = np.max(model)
        if dt:
            self.dt = dt
        else:
            self.dt = 0.6 * self.dx / max_vel
        self.nx_padded = self.nx + 2*npad
        self.model_padded = np.pad(model, (npad, npad), 'edge')
        self.model_padded2_dt2 = self.model_padded**2 * self.dt**2
        self.wavefield = [np.zeros(self.nx_padded, np.float32),
                          np.zeros(self.nx_padded, np.float32)
                         ]
        self.current_wavefield = self.wavefield[0]
        self.previous_wavefield = self.wavefield[1]

class VPy1(Propagator):
    """A simple Python implementation."""
    def step(self, num_steps, sources=None, sources_x=None):
        """Propagate wavefield."""
        for istep in range(num_steps):
            f = self.current_wavefield
            fp = self.previous_wavefield

            for x in range(8, self.nx_padded-8):
                f_xx = (-735*f[x-8]+15360*f[x-7]
                        -156800*f[x-6]+1053696*f[x-5]
                        -5350800*f[x-4]+22830080*f[x-3]
                        -94174080*f[x-2]+538137600*f[x-1]
                        -924708642*f[x+0]
                        +538137600*f[x+1]-94174080*f[x+2]
                        +22830080*f[x+3]-5350800*f[x+4]
                        +1053696*f[x+5]-156800*f[x+6]
                        +15360*f[x+7]-735*f[x+8])/(302702400*self.dx**2)
                fp[x] = (self.model_padded[x]**2 * self.dt**2 * f_xx
                         + 2*f[x] - fp[x])

            for i in range(sources.shape[0]):
                sx = sources_x[i] + 8
                source_amp = sources[i, istep]
                fp[sx] += (self.model_padded[sx]**2 * self.dt**2 * source_amp)

            self.current_wavefield = fp
            self.previous_wavefield = f

        return self.current_wavefield[8:self.nx_padded-8]

class VTF(Propagator):
    """A TensorFlow implementation."""
    def __init__(self, model, dx, dt=None):
        super(VTF, self).__init__(model, dx, dt, npad=0)
        self.sess = tf.Session()
        self.model_padded2_dt2 = tf.constant(self.model_padded2_dt2)
        self.f = tf.placeholder(tf.float32, shape=(self.nx_padded))
        self.fp = tf.placeholder(tf.float32, shape=(self.nx_padded))
        self.sources = tf.placeholder(tf.float32, shape=(None))
        self.sources_x = tf.placeholder(tf.int64, shape=(None))

        fd_kernel = np.array([-735, +15360,
                                   -156800, +1053696,
                                   -5350800, +22830080,
                                   -94174080, +538137600,
                                   -924708642,
                                   +538137600, -94174080,
                                   +22830080, -5350800,
                                   +1053696, -156800,
                                   +15360, -735]/(302702400*self.dx**2),
                                   np.float32)
        fd_kernel = fd_kernel.reshape([-1, 1, 1])
        fd_kernel = tf.constant(fd_kernel)

        def laplace(x):
            return tf.squeeze(tf.nn.conv1d(tf.reshape(x, [1, -1, 1]), fd_kernel, 1, 'SAME'))

        self.fp_ = self.model_padded2_dt2 * laplace(self.f) + 2*self.f - self.fp
        #print(self.model_padded2_dt2, self.f, self.fp_, laplace(self.f))

        sources_v = tf.gather(self.model_padded2_dt2, self.sources_x)
        sources_amp = self.sources * sources_v
        self.fp_ += tf.sparse_to_dense(self.sources_x, [self.nx_padded], sources_amp)

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def step(self, num_steps, sources=None, sources_x=None):
        """Propagate wavefield."""

        ssort = sources_x.argsort()
        sources_sort = sources[ssort, :]
        sources_x_sort = sources_x[ssort]

        for istep in range(num_steps):
            sources_step = sources_sort[:,istep]
            #print(sources_step)
            #print(sources_x)
            #print(sources_step.dtype, sources_step.shape)
            #print(sources_x_step.dtype, sources_x_step.shape)
            #print(self.current_wavefield.dtype, self.current_wavefield.shape)
            #print(self.previous_wavefield.dtype, self.previous_wavefield.shape)

            y = self.sess.run(self.fp_, {self.sources: sources_step,
                                         self.sources_x: sources_x_sort,
                                         self.f: self.current_wavefield,
                                         self.fp: self.previous_wavefield})
            self.previous_wavefield[:] = y[:]

            tmp = self.current_wavefield
            self.current_wavefield = self.previous_wavefield
            self.previous_wavefield = tmp

        if num_steps > 0:
            return y
        else:
            return self.current_wavefield
