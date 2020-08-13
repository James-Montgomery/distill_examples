import numpy as np

import pymc3 as pm
import theano
import theano.tensor as tt

#####

class DataGenerator(object):
    """
    """

    def __init__(self, slope=1.0, amplitude=1.0, frequency=1.0, noise=1.0):
        """
        """
        self.slope = slope
        self.amplitude = amplitude
        self.freqency = frequency
        self.noise = noise

    def mean_function(self, x):
        """
        """
        return self.amplitude * np.sin(self.freqency * x) + self.slope * x

    def generate_mean_function(self, bounds=(-10, 10)):
        """
        """
        lb, ub = bounds
        x = np.linspace(lb, ub, 1000).reshape(-1, 1)
        return x, self.mean_function(x)

    @staticmethod
    def _check_bounds(bounds):
        if not isinstance(bounds, tuple):
            raise ValueError("Bounds input must be a tuple.")
        lb, ub = bounds
        if lb >= ub:
            raise ValueError("Lower bound must be lower than upper bound.")
        return lb, ub

    @staticmethod
    def sample_x(num_samples, bounds, gap):
        """
        """
        lb, ub = bounds
        if gap == False:
            return np.random.uniform(lb, ub, (num_samples, 1))

        ratio = 5 / (ub - lb)
        num_left = np.round((1.0 - ratio) * num_samples, 0).astype(int)
        num_right = np.round(ratio * num_samples, 0).astype(int)
        return np.vstack([
            np.random.uniform(lb, 0, (num_left, 1)),
            np.random.uniform(5.0, ub, (num_right, 1)),
        ])

    def generate_sample_data(self, num_samples, bounds=(-10, 10), gap=False):
        """
        """
        lb, ub = self._check_bounds(bounds)
        x = self.sample_x(num_samples, bounds, gap)
        y = self.mean_function(x)
        y += np.random.normal(0, self.noise, y.shape)
        return x, y


#####

class BayesianNN(object):
    """
    """

    def __init__(self):
        """

        """
        self.model = None
        self.trace = None

    def _build_model(self, x, y):
        """
        """

        if self.model is not None:
            raise Exception("Overwriting previous fit.")

        input_dim = x.shape[1]
        output_dim = y.shape[1]
        ann_input = theano.shared(x)
        ann_output = theano.shared(y)

        n_hidden = 3
        with pm.Model() as neural_network:
            # Weights from input to hidden layer
            weights_in_1 = pm.Normal('w_in_1', 0, sd=1, shape=(input_dim, n_hidden))#, testval=init_1)
            weights_b_1 = pm.Normal('w_b_1', 0, sd=1, shape=(n_hidden))#, testval=init_b_1)

            # Weights from 1st to 2nd layer
            weights_1_2 = pm.Normal('w_1_2', 0, sd=1, shape=(n_hidden, n_hidden))#, testval=init_2)
            weights_b_2 = pm.Normal('w_b_2', 0, sd=1, shape=(n_hidden))#, testval=init_b_2)

            # Weights from hidden layer to output
            weights_2_out = pm.Normal('w_2_out', 0, sd=1, shape=(n_hidden, output_dim))#, testval=init_out)
            weights_b_out = pm.Normal('w_b_out', 0, sd=1, shape=(output_dim))#, testval=init_b_out)

            # Build neural-network using tanh activation function
            act_1 = pm.math.tanh(pm.math.dot(ann_input, weights_in_1) + weights_b_1)
            act_2 = pm.math.tanh(pm.math.dot(act_1, weights_1_2) + weights_b_2)
            act_out = pm.math.dot(act_2, weights_2_out) + weights_b_out

            variance = pm.HalfNormal('uncertainty', sigma=3.0)
            out = pm.Normal('out', mu=act_out,
                            sigma=variance,
                            observed=ann_output)

        self.model = neural_network

    def fit(self, x, y, num_iter=1000, VI=False):
        """
        """
        x = x.astype("float32")
        y = y.astype("float32")
        self._build_model(x, y)

        with self.model:

            if VI is False:
                step = pm.NUTS()
                trace = pm.sample(
                    num_iter,
                    tune=100,
                    cores=1,
                    chains=1,
                    step=step
                )
            elif VI is True:
                approx = pm.fit(n=10000, method='svgd')
                trace = approx.sample(draws=20000)

        self.trace = trace

    def sample_posterior(self, x, samples):
        """
        """
        x = x .astype("float32")
        trace = self.trace
        posterior_samples = trace['w_in_1'].shape[0]
        output = []
        for _ in range(samples):
            i = np.random.randint(posterior_samples)
            act_1 = np.tanh(np.dot(x, trace['w_in_1'][i]) + trace['w_b_1'][i])
            act_2 = np.tanh(np.dot(act_1, trace['w_1_2'][i]) + trace['w_b_2'][i])
            y = np.dot(act_2, trace['w_2_out'][i]) + trace['w_b_out'][i]
            output.append(y)
        return np.asarray(output)
