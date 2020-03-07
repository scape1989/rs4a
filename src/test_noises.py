import unittest
import torch
import noises_new as noises

class TestSigma(unittest.TestCase):

    def test_sigma(self):
        rel_tol = 1e-2
        dim = 3 * 32 * 32
        dev = 'cpu'
        configs = [
            dict(noise=noises.PowerInfNoise, a=dim+10),
            dict(noise=noises.PowerInfNoise, a=dim+1000),
        ]
        for k in [1, 2, 10, 100]:
            for j in [0, 1, 10, 100, 1000]:
                configs.append(
                    dict(noise=noises.ExpInfNoise, k=k, j=j)
                )
        for c in configs:
            c['device'] = dev
            c['dim'] = dim
            c['sigma'] = 1
            with self.subTest(config=c):
                noisecls = c.pop('noise')
                noise = noisecls(**c)
                emp_sigma = noise.sample(torch.zeros(10000, dim)).std()
                self.assertAlmostEqual(emp_sigma, noise.sigma,
                                        delta=rel_tol * emp_sigma)
if __name__ == '__main__':
    unittest.main()