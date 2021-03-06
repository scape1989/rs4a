import unittest
import numpy as np
import tqdm
import torch
import noises

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
        for k in [1, 2, 10, 50]:
            for j in [0, 1, 10, 100, 1000]:
                configs.append(
                    dict(noise=noises.ExpInfNoise, k=k, j=j)
                )
        for k in [1, 2, 10, 20]:
            for j in [0, 1, 10, 100, 1000]:
                configs.append(
                    dict(noise=noises.Exp2Noise, k=k, j=j)
                )
        for k in [1, 2, 10, 20]:
            configs.append(
                dict(noise=noises.Exp1Noise, k=k)
            )
        for a in [10, 100, 1000]:
            configs.append(
                dict(noise=noises.PowerInfNoise, a=a+dim)
            )
        for k in [1, 2, 10, 100]:
            for j in [0, 1, 10, 100, 1000]:
                configs.append(
                    dict(noise=noises.Exp2Noise, k=k, j=j)
                )
        for k in [1, 2, 10, 100]:
            for a in [10, 100, 1000]:
                a = (dim + a) / k
                configs.append(
                    dict(noise=noises.Power2Noise, k=k, a=a)
                )
        for c in tqdm.tqdm(configs):
            c['device'] = dev
            c['dim'] = dim
            c['sigma'] = 1
            with self.subTest(config=dict(c)):
                noisecls = c.pop('noise')
                noise = noisecls(**c)
                samples = noise.sample(torch.zeros(nsamples, dim))
                self.assertEqual(samples.shape, torch.Size((nsamples, dim)))
                emp_sigma = samples.std()
                self.assertAlmostEqual(emp_sigma, noise.sigma,
                                       delta=rel_tol * emp_sigma)

class TestRadii(unittest.TestCase):

    def test_laplace_linf_radii(self):
        '''Test that the "approx" and "integrate" modes of linf certification
        for Laplace agree with each other.'''
        noise = noises.LaplaceNoise('cpu', 3*32*32, sigma=1)
        cert1 = noise.certifylinf(torch.arange(0.5, 1, 0.01))
        cert2 = noise.certifylinf(torch.arange(0.5, 1, 0.01), 'integrate')
        self.assertTrue(np.allclose(cert1, cert2, rtol=1e-2))

    def test_exp2_l2_radii(self):
        '''Test that for exp(-\|x\|_2), the differential and level set methods
        obtain similar robust radii.'''
        rs = torch.arange(0.5, 1, 0.01)
        noise = noises.Exp2Noise('cpu', 3*32*32, sigma=1)
        cert1 = noise.certifyl2(rs)
        cert2 = noise.certifyl2_levelset(rs)
        self.assertTrue(np.allclose(cert1, cert2, rtol=1e-2))

if __name__ == '__main__':
    unittest.main()
