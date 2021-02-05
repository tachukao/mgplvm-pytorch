import abc
from ..base import Module
from torch import Tensor


class Kernel(Module, metaclass=abc.ABCMeta):
    """
    Base kernel class
    """

    def __init__(self):
        super().__init__()

    @abc.abstractstaticmethod
    def K(self, x: Tensor, y: Tensor) -> Tensor:
        pass

    @abc.abstractstaticmethod
    def trK(self, x: Tensor) -> Tensor:
        pass

    @abc.abstractstaticmethod
    def diagK(self, x: Tensor) -> Tensor:
        pass

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return self.K(x, y)


#class Combination(Kernel):
#
#    def __init__(self, kernels: List[Kernel]):
#        """
#        Combination Kernels
#
#        Parameters
#        ----------
#        kernels : list of kernels
#
#        Notes
#        -----
#        Implementation largely follows thats described in
#        https://github.com/GPflow/GPflow/blob/develop/gpflow/kernels/base.py
#        """
#        super().__init__()
#        self.kernels = kernels
#
#    def forward(self, x: List[Tensor], y: List[Tensor]) -> Tensor:
#        return self._reduce([k(x, y) for (k, x, y) in zip(self.kernels, x, y)])
#
#    @abc.abstractmethod
#    def _reduce(self, x: List[Tensor]) -> Tensor:
#        pass
#
#    @property
#    def prms(self) -> List[Tuple[Tensor]]:
#        return [k.prms for k in self.kernels]
#
#
#class Sum(Combination):
#
#    def _reduce(self, x: List[Tensor]) -> Tensor:
#        return torch.sum(torch.stack(x, dim=0), dim=0)
#
#    def trK(self, x: Tensor) -> Tensor:
#        """
#        sum_i(alpha_1^2 + alpha_2^2)
#        """
#        alphas = [k.prms[0] for k in self.kernels]
#        sqr_alphas = [torch.square(alpha) for alpha in alphas]
#        sqr_alpha = torch.stack(sqr_alphas).sum(dim=0)
#        return torch.ones(x[0].shape[:-2]).to(
#            sqr_alpha.device) * sqr_alpha * x[0].shape[-1]
#
#
#class Product(Combination):
#
#    def _reduce(self, x: List[Tensor]):
#        return torch.prod(torch.stack(x, dim=0), dim=0)
#
#    def trK(self, x: Tensor) -> Tensor:
#        """
#        sum_i(alpha_1^2 * alpha_2^2)
#        """
#        alphas = [k.prms[0] for k in self.kernels]
#        sqr_alphas = [torch.square(alpha) for alpha in alphas]
#        sqr_alpha = torch.stack(sqr_alphas).prod(dim=0)
#        return torch.ones(x[0].shape[:-2]).to(
#            sqr_alpha.device) * sqr_alpha * x[0].shape[-1]
