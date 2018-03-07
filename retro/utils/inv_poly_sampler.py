# pylint: disable=wrong-import-position, invalid-name


"""
Inverse transform sampling for a PDF that can be described by a polynomial.
"""


from __future__ import absolute_import, division, print_function


__author__ = 'J.L. Lanfranchi'
__license__ = '''Copyright 2018 Justin L. Lanfranchi

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.'''


import numpy as np
from scipy.optimize import brentq


__all__ = ['InvPolySampler']

__author__ = 'P. Eller, J.L. Lanfranchi'
__license__ = '''Copyright 2017 Philipp Eller and Justin L. Lanfranchi

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.'''


class InvPolySampler(object):
    """
    Inverse transform sampling for a probability density function that can be
    described by a polynomial.

    Parameters
    ----------
    n_samp : int
        Linear interpolation will be performed between this many points sampled
        from the inverse CDF. Note that both endpoints of the range (i.e., 0 and
        1) are included.

    power : float
        The range of the CDF--which is in [0, 1]--is transformed by this power
        in order to sample the CDF more accurately.

    pdf : numpy.polynomial.Polynomial or array-like of coeffs
        If an array-like of coeffs is provided, note that these must be in
        _ascending_ order, as these are passed to
        :class:`numpy.polynomial.Polynomial`.

    domain : 2-seq of float; optional if `pdf` is np.polynomial.Polynomial
        The domain of the PDF (and hence also of the CDF). This is unnecessary
        if `pdf` is a :class:`numpy.polynomial.Polynomial` since this object
        already has a `domain` attribute defined. If both are specified, both
        domains must be equal.

    """
    def __init__(self, n_samp, power, pdf, domain=None):
        self.n_samp = n_samp
        self.prob_binwidth = 1 / (n_samp - 1)
        self.power = power
        self.inv_power = 1 / power

        self.cdf_samples = np.linspace(0, 1, n_samp)**power
        """Sample CDF at y-values regularly spaced in ``y**power``"""

        if domain is None:
            assert isinstance(pdf, np.polynomial.Polynomial)
            domain = pdf.domain
        else:
            if len(domain) != 2:
                raise ValueError('`domain` must be a length-2 sequence; got {}'
                                 .format(domain))
            if isinstance(pdf, np.polynomial.Polynomial):
                assert pdf.domain[0] == domain[0]
                assert pdf.domain[1] == domain[1]

        if domain[0] >= domain[1]:
            raise ValueError(
                '`domain[0]` (lower edge) is greater than or equal to'
                ' `domain[1]` (upper edge): {}'.format(domain)
            )
        self.domain = min(domain), max(domain)

        if not isinstance(pdf, np.polynomial.Polynomial):
            pdf = np.polynomial.Polynomial(pdf, domain=self.domain)

        # To be a valid PDF, must integrate to 1
        pdf_int = pdf.integ()
        pdf = pdf / (pdf_int(self.domain[1]) - pdf_int(self.domain[0]))

        cdf_poly = pdf.integ()
        cdf_poly -= cdf_poly(self.domain[0])

        # Make sure we have a valid CDF
        assert np.isclose(cdf_poly(self.domain[0]), 0)
        assert np.isclose(cdf_poly(self.domain[1]), 1)

        self.pdf_poly = pdf
        self.cdf_poly = cdf_poly

        # Numerically invert the CDF: Find domain locations (x-values) that
        # correspond to the CDF samples (y-values)
        domain_samples = []
        for cdf_y_val in self.cdf_samples:
            try:
                domain_x_val = brentq(
                    f=lambda x: self.cdf_poly(x) - cdf_y_val,
                    a=self.domain[0],
                    b=self.domain[1]
                )
                domain_samples.append(domain_x_val)
            except ValueError:
                print('Failed to find location in domain where CDF = {}'
                      .format(cdf_y_val))
                raise

        self.domain_samples = np.array(domain_samples)
        self.inv_cdf_slopes = (np.diff(self.domain_samples)
                               / np.diff(self.cdf_samples**self.inv_power))

    def __call__(self, val):
        """Perform linear interpolation on `val`, sample(s) in [0, 1]"""
        prob_bin_idx, dx = np.divmod(val**self.inv_power, self.prob_binwidth)
        exceed_max_mask = prob_bin_idx > self.n_samp - 2
        prob_bin_idx[exceed_max_mask] = self.n_samp - 2
        dx[exceed_max_mask] = self.prob_binwidth
        prob_bin_idx = prob_bin_idx.astype(int)
        y0 = self.domain_samples[prob_bin_idx]
        m = self.inv_cdf_slopes[prob_bin_idx]
        y = m*dx + y0
        return y
