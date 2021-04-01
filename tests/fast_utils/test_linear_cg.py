#!/usr/bin/env python3
"""
Taken from GPytorch https://github.com/cornellius-gp/gpytorch/blob/2e55ec6cf6bd70dc96db76d88689dd743fd42ade/test/utils/test_linear_cg.py

With this License:
MIT License

Copyright (c) 2017 Jake Gardner

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import random

import torch

from mgplvm.fast_utils.linear_cg import linear_cg


class TestLinearCG():

    def setup_method(self, method):
        if os.getenv("UNLOCK_SEED") is None or os.getenv(
                "UNLOCK_SEED").lower() == "false":
            self.rng_state = torch.get_rng_state()
            torch.manual_seed(0)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(0)
            random.seed(0)

    def teardown_method(self, method):
        if hasattr(self, "rng_state"):
            torch.set_rng_state(self.rng_state)

    def test_cg(self):
        size = 100
        matrix = torch.randn(size, size, dtype=torch.float64)
        matrix = matrix.matmul(matrix.transpose(-1, -2))
        matrix.div_(matrix.norm())
        matrix.add_(torch.eye(matrix.size(-1), dtype=torch.float64).mul_(1e-1))

        rhs = torch.randn(size, 50, dtype=torch.float64)
        solves = linear_cg(matrix.matmul, rhs=rhs, max_iter=size)

        # Check cg
        matrix_chol = matrix.cholesky()
        actual = torch.cholesky_solve(rhs, matrix_chol)
        assert (torch.allclose(solves, actual, atol=1e-3, rtol=1e-4))

    def test_cg_with_tridiag(self):
        size = 10
        matrix = torch.randn(size, size, dtype=torch.float64)
        matrix = matrix.matmul(matrix.transpose(-1, -2))
        matrix.div_(matrix.norm())
        matrix.add_(torch.eye(matrix.size(-1), dtype=torch.float64).mul_(1e-1))

        rhs = torch.randn(size, 50, dtype=torch.float64)
        solves, t_mats = linear_cg(matrix.matmul,
                                   rhs=rhs,
                                   n_tridiag=5,
                                   max_tridiag_iter=10,
                                   max_iter=size,
                                   tolerance=0,
                                   eps=1e-15)

        # Check cg
        matrix_chol = matrix.cholesky()
        actual = torch.cholesky_solve(rhs, matrix_chol)
        assert (torch.allclose(solves, actual, atol=1e-3, rtol=1e-4))

        # Check tridiag
        eigs = matrix.symeig()[0]
        for i in range(5):
            approx_eigs = t_mats[i].symeig()[0]
            assert (torch.allclose(eigs, approx_eigs, atol=1e-3, rtol=1e-4))

    def test_batch_cg(self):
        batch = 5
        size = 100
        matrix = torch.randn(batch, size, size, dtype=torch.float64)
        matrix = matrix.matmul(matrix.transpose(-1, -2))
        matrix.div_(matrix.norm())
        matrix.add_(torch.eye(matrix.size(-1), dtype=torch.float64).mul_(1e-1))

        rhs = torch.randn(batch, size, 50, dtype=torch.float64)
        solves = linear_cg(matrix.matmul, rhs=rhs, max_iter=size)

        # Check cg
        matrix_chol = torch.cholesky(matrix)
        actual = torch.cholesky_solve(rhs, matrix_chol)
        assert (torch.allclose(solves, actual, atol=1e-3, rtol=1e-4))

    def test_batch_cg_with_tridiag(self):
        batch = 5
        size = 10
        matrix = torch.randn(batch, size, size, dtype=torch.float64)
        matrix = matrix.matmul(matrix.transpose(-1, -2))
        matrix.div_(matrix.norm())
        matrix.add_(torch.eye(matrix.size(-1), dtype=torch.float64).mul_(1e-1))

        rhs = torch.randn(batch, size, 10, dtype=torch.float64)
        solves, t_mats = linear_cg(matrix.matmul,
                                   rhs=rhs,
                                   n_tridiag=8,
                                   max_iter=size,
                                   max_tridiag_iter=10,
                                   tolerance=0,
                                   eps=1e-30)

        # Check cg
        matrix_chol = torch.cholesky(matrix)
        actual = torch.cholesky_solve(rhs, matrix_chol)
        assert (torch.allclose(solves, actual, atol=1e-3, rtol=1e-4))

        # Check tridiag
        for i in range(5):
            eigs = matrix[i].symeig()[0]
            for j in range(8):
                approx_eigs = t_mats[j, i].symeig()[0]
                assert (torch.allclose(eigs, approx_eigs, atol=1e-3, rtol=1e-4))


if __name__ == "__main__":
    tests = TestLinearCG()
    tests.test_batch_cg()
    tests.test_linear_cg()
    tests.test_cg_with_tridiag()
    tests.test_batch_cg_with_tridiag()
    tests.test_cg()
