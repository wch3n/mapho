"""Standalone exchange compression from Lin, DOI:10.1021/acs.jctc.6b00092.

Only tensors in an orthonormal representation enter this module. It neither
computes Fock exchange nor imports any electronic-structure backend. The caller
supplies both the construction orbitals and their full exchange actions.
"""

from dataclasses import dataclass

import torch
from torch import Tensor


def _check_tensor(value: Tensor, name: str) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype not in (
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    ):
        raise TypeError(f"{name} must use float32/64 or complex64/128")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains nonfinite values")


@dataclass(frozen=True)
class ACEOperator:
    """Represent exchange as ``K(v) = -X @ (X.mH @ v)``.

    ``factors`` has shape (basis_size, rank). Applying the operator does not
    construct a dense basis_size-by-basis_size matrix. Tensor dtype and device
    are preserved, including complex orbitals and CUDA tensors.
    """

    factors: Tensor

    def __post_init__(self) -> None:
        _check_tensor(self.factors, "factors")
        if self.factors.ndim != 2 or self.factors.shape[0] == 0:
            raise ValueError("factors must have shape (positive basis_size, rank)")

    @property
    def rank(self) -> int:
        return self.factors.shape[1]

    @property
    def storage_bytes(self) -> int:
        """Storage of the retained factors, excluding temporary work arrays."""
        return self.factors.numel() * self.factors.element_size()

    def apply(self, vectors: Tensor) -> Tensor:
        """Apply to one vector or a column block in the same representation."""
        _check_tensor(vectors, "vectors")
        if vectors.ndim not in (1, 2) or vectors.shape[0] != self.factors.shape[0]:
            raise ValueError("vectors must have shape (basis_size,) or (basis_size, n)")
        if vectors.dtype != self.factors.dtype or vectors.device != self.factors.device:
            raise ValueError("vectors and factors must share dtype and device")
        return -(self.factors @ (self.factors.mH @ vectors))

    @classmethod
    def from_samples(
        cls,
        orbitals: Tensor,
        exchange_actions: Tensor,
        *,
        rtol: float | None = None,
    ) -> "ACEOperator":
        """Compress a fixed negative exchange operator from ``W = K @ Phi``.

        Columns of Phi must be independent in the exchange metric. This initial
        implementation rejects a singular Gram matrix instead of regularizing
        or truncating it silently. Phi need not itself have orthonormal columns;
        its ambient coefficient representation must use the Euclidean inner
        product. ``rtol`` controls the Hermiticity and interpolation checks.

        Include the desired hybrid fraction in W before calling this function.
        This function does not apply an additional exchange fraction.
        """
        _check_tensor(orbitals, "orbitals")
        _check_tensor(exchange_actions, "exchange_actions")
        if orbitals.ndim != 2 or exchange_actions.shape != orbitals.shape:
            raise ValueError(
                "orbitals and exchange_actions must have the same 2D shape"
            )
        n_basis, n_states = orbitals.shape
        if not 0 < n_states <= n_basis:
            raise ValueError("require 0 < number of construction states <= basis_size")
        if (
            orbitals.dtype != exchange_actions.dtype
            or orbitals.device != exchange_actions.device
        ):
            raise ValueError(
                "orbitals and exchange_actions must share dtype and device"
            )
        if rtol is None:
            rtol = 2e-4 if orbitals.real.dtype == torch.float32 else 1e-10
        if not 0 < rtol < 1:
            raise ValueError("rtol must be finite and between zero and one")

        gram = orbitals.mH @ exchange_actions
        scale = torch.linalg.vector_norm(gram)
        defect = torch.linalg.vector_norm(gram - gram.mH)
        if not bool(torch.isfinite(gram).all()):
            raise ValueError("exchange Gram matrix overflowed")
        if bool(defect > rtol * scale):
            raise ValueError("exchange samples have a non-Hermitian Gram matrix")

        # Lin (2016), Eqs. (10)-(15), with adjoints for complex coefficients.
        # -Phi^H W = L L^H; X = W L^{-H}. Solve without forming an inverse.
        positive_gram = -0.5 * (gram + gram.mH)
        lower, info = torch.linalg.cholesky_ex(positive_gram)
        if int(info) != 0:
            raise ValueError(
                "negative exchange Gram matrix must be positive definite; "
                "check the sign of K and remove dependent or zero-action states"
            )
        factors = torch.linalg.solve_triangular(
            lower, exchange_actions.mH, upper=False
        ).mH
        operator = cls(factors)
        residual = operator.apply(orbitals) - exchange_actions
        relative_scale = torch.linalg.vector_norm(exchange_actions)
        if bool(torch.linalg.vector_norm(residual) > rtol * relative_scale):
            raise ValueError(
                "ACE interpolation failed; construction samples may be ill-conditioned"
            )
        return operator
