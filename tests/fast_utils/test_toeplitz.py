#!/usr/bin/env python3
"""
Taken from GPytorch with modifications https://github.com/cornellius-gp/gpytorch/blob/236304cb8420076c31d6de6a267803135055822e/test/utils/test_toeplitz.py

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

import torch
import os

from mgplvm.fast_utils import toeplitz, toeplitz_matmul, sym_toeplitz


class TestToeplitz():

    def setup_method(self, method):
        if os.getenv("UNLOCK_SEED") is None or os.getenv(
                "UNLOCK_SEED").lower() == "false":
            self.rng_state = torch.get_rng_state()
            torch.manual_seed(0)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(0)

    def test_sym_toeplitz_constructs_tensor_from_vector(self):
        c = torch.tensor([1, 6, 4, 5], dtype=torch.float)

        res = sym_toeplitz(c)
        actual = torch.tensor(
            [[1, 6, 4, 5], [6, 1, 6, 4], [4, 6, 1, 6], [5, 4, 6, 1]],
            dtype=torch.float)

        assert (torch.equal(res, actual))

    def test_toeplitz_matmul(self):
        col = torch.tensor([1, 6, 4, 5], dtype=torch.float)
        row = torch.tensor([1, 2, 1, 1], dtype=torch.float)
        rhs_mat = torch.randn(4, 2, dtype=torch.float)

        # Actual
        lhs_mat = toeplitz(col, row)
        actual = torch.matmul(lhs_mat, rhs_mat)

        # Fast toeplitz
        res = toeplitz_matmul(col, row, rhs_mat)
        assert (torch.allclose(res, actual))

    def test_toeplitz_matmul_batch(self):
        cols = torch.tensor([[1, 6, 4, 5], [2, 3, 1, 0], [1, 2, 3, 1]],
                            dtype=torch.float)
        rows = torch.tensor([[1, 2, 1, 1], [2, 0, 0, 1], [1, 5, 1, 0]],
                            dtype=torch.float)

        rhs_mats = torch.randn(3, 4, 2, dtype=torch.float)

        # Actual
        lhs_mats = torch.zeros(3, 4, 4, dtype=torch.float)
        for i, (col, row) in enumerate(zip(cols, rows)):
            lhs_mats[i].copy_(toeplitz(col, row))
        actual = torch.matmul(lhs_mats, rhs_mats)

        # Fast toeplitz
        res = toeplitz_matmul(cols, rows, rhs_mats)
        assert (torch.allclose(res, actual))

    def test_toeplitz_matmul_batchmat(self):
        col = torch.tensor([1, 6, 4, 5], dtype=torch.float)
        row = torch.tensor([1, 2, 1, 1], dtype=torch.float)
        rhs_mat = torch.randn(3, 4, 2, dtype=torch.float)

        # Actual
        lhs_mat = toeplitz(col, row)
        actual = torch.matmul(lhs_mat.unsqueeze(0), rhs_mat)

        # Fast toeplitz
        res = toeplitz_matmul(col.unsqueeze(0), row.unsqueeze(0), rhs_mat)
        assert (torch.allclose(res, actual))


if __name__ == "__main__":
    tests = TestToeplitz
    tests.test_toeplitz_matmul()
    tests.test_toeplitz_matmul_batchmat()
    tests.test_toeplitz_matmul_batch()
    tests.test_sym_toeplitz_constructs_tensor_from_vector()
