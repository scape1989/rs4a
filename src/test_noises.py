import unittest
import numpy as np
import tqdm
import torch
import noises_new as noises

class TestSigma(unittest.TestCase):

    def test_sigma(self):
        rel_tol = 1e-2
        nsamples = int(1e4)
        dim = 3 * 32 * 32
        dev = 'cpu'
        configs = [
            dict(noise=noises.UniformNoise),
            dict(noise=noises.GaussianNoise),
            dict(noise=noises.LaplaceNoise),
            dict(noise=noises.UniformBallNoise),
        ]
        for a in [3, 10, 100, 1000]:
            configs.append(
                dict(noise=noises.ParetoNoise, a=a)
            )
        for k in [1, 2, 10, 100]:
            for j in [0, 1, 10, 100, 1000]:
                configs.append(
                    dict(noise=noises.ExpInfNoise, k=k, j=j)
                )
#        for a in [10, 100, 1000]:
#            configs.append(
#                dict(noise=noises.PowerInfNoise, a=a+dim)
#            )
        for c in tqdm.tqdm(configs):
            c['device'] = dev
            c['dim'] = dim
            c['sigma'] = 2
            with self.subTest(config=dict(c)):
                noisecls = c.pop('noise')
                noise = noisecls(**c)
                emp_sigma = noise.sample(torch.zeros(nsamples, dim)).std()
                self.assertAlmostEqual(emp_sigma, noise.sigma,
                                       delta=rel_tol * emp_sigma)

class TestRadii(unittest.TestCase):

    def test_laplace_linf_radii(self):
        noise = noises.LaplaceNoise('cpu', 3*32*32, sigma=1)
        cert1 = noise.certifylinf(torch.arange(0.5, 1, 0.01))
        cert2 = noise.certifylinf(torch.arange(0.5, 1, 0.01), 'integrate')
        self.assertTrue(np.allclose(cert1, cert2, rtol=1e-2))

if __name__ == '__main__':
    unittest.main()
