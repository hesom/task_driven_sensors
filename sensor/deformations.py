from abc import ABC, abstractmethod

import torch
from typing import Optional
from jaxtyping import Float

class Deformation(ABC):
    """
    Abstract class for deformations
    """
    @abstractmethod
    def __call__(self, X: Float[torch.Tensor, "*batch 2"], theta: Float[torch.Tensor, "n"]) \
        -> tuple[Float[torch.Tensor, "*batch 2"], Float[torch.Tensor, "*batch 2 2"]]:
        """
        Deform the input tensor
        :param X: input tensor, shape (*, 2)
        :param theta: parameter of deformation
        :return: (deformed tensor, jacobian matrix, shape ((*, 2), (*, 2, 2))
        """
        pass

    @abstractmethod
    def dfdt(self, X: Float[torch.Tensor, "*batch 2"], theta: Float[torch.Tensor, "n"]) \
            -> Float[torch.Tensor, "*batch 2 n"]:
        """
        Compute the derivative of the deformation with respect to theta
        :param X: input tensor, shape (*, 2)
        :param theta: parameter of deformation
        :return: derivative of the deformation with respect to theta, shape (*, 2, dim(theta))
        """
        pass

    @abstractmethod
    def jacobian(self, X: Float[torch.Tensor, "*batch 2"], theta: Float[torch.Tensor, "n"]) \
            -> Float[torch.Tensor, "*batch 2 2"]:
        pass

    @abstractmethod
    def numParameters(self) -> int:
        """
        :return: number of parameters of the deformation
        """
        pass

    @abstractmethod
    def getNeutralParameter(self) -> Float[torch.Tensor, "n"]:
        """
        :return: neutral parameter of the deformation
        """
        pass

    def inverse(self, X: Float[torch.Tensor, "*batch 2"], theta: Float[torch.Tensor, "n"]) \
            -> Float[torch.Tensor, "*batch 2"]:
        """
        Computes the inverse of the deformation. Uses Newton method if not overridden
        :param X: input tensor, shape (*, 2)
        :param theta: parameter of deformation
        :return: inverse of the deformation
        """
        x = X.clone()

        for _ in range(5):
            curr, jac = self(x, theta)
            jac = torch.linalg.inv(jac)
            x = x - torch.matmul(jac, curr.unsqueeze(-1) - X.unsqueeze(-1)).squeeze(-1)
        return x

    def getAllDerivatives(self, X: Float[torch.Tensor, "*batch 2"], theta: Float[torch.Tensor, "n"])\
        -> tuple[Float[torch.Tensor, "*batch 2"], Float[torch.Tensor, "*batch 2 2"], Float[torch.Tensor, "*batch 2 n"]]:
            x, jac = self(X, theta)
            dfdt = self.dfdt(X, theta)
            return x, jac, dfdt

    def getGradMask(self, device=None) -> Optional[Float[torch.Tensor, "n"]]:
        """
        Returns an optional gradient mask. Used to freeze some sensor parameters.
        Probably only useful for composed deformations, but defined here for compatibility
        :return: binary mask that can be multiplied elementwise with the gradient
        """
        return None

class CurvilinearDeformation(Deformation):
    """
    From https://math.stackexchange.com/questions/459872/adjustable-sigmoid-curve-s-curve-from-0-0-to-1-1
    Also see https://dhemery.github.io/DHE-Modules/technical/sigmoid/
    Here we actually only need the upper half of the sigmoid
    """

    def __init__(self, h=1.0) -> None:
        super().__init__()
        self.h = h

    def __call__(self, X: Float[torch.Tensor, "*batch 2"], theta: Float[torch.Tensor, "n"]) -> tuple[Float[torch.Tensor, "*batch 2"], Float[torch.Tensor, "*batch 2 2"]]:
        r = torch.norm(X, dim=-1)

        mask = r < self.h
        s = theta[0]
        t = theta[1]

        r_t_x = mask * (s - 1) / (2*s*r - s - 1)
        r_t_y = mask * (t - 1) / (2*t*r - t - 1)

        r_t = torch.stack((r_t_x * X[..., 0] + ~mask*X[..., 0], r_t_y * X[..., 1] + ~mask*X[..., 1]), dim=-1)

        return r_t, self.jacobian(X, theta)

    def jacobian(self, X: Float[torch.Tensor, "*batch 2"], theta: Float[torch.Tensor, "n"]) -> Float[torch.Tensor, "*batch 2 2"]:
        r= torch.norm(X, dim=-1)
        r2 = r**2

        x = X[..., 0]
        y = X[..., 1]
        s = theta[0]
        t = theta[1]

        mask = r < self.h

        j11 = torch.nan_to_num((-2*s*x**2*(s - 1)*r2**(7/2) + (1 - s)*r2**4*(-2*s*r + s + 1))/(r2**4*(-2*s*r + s + 1)**2), nan=1.0, posinf=1.0, neginf=1.0)
        j22 = torch.nan_to_num((-2*t*y**2*(t - 1)*r2**(7/2) + (1 - t)*r2**4*(-2*t*r + t + 1))/(r2**4*(-2*t*r + t + 1)**2), nan=1.0, posinf=1.0, neginf=1.0)
        j12 = torch.nan_to_num(-2*s*x*y*(s - 1)/(r*(-2*s*r + s + 1)**2), nan=0.0, posinf=0.0, neginf=0.0)
        j21 = torch.nan_to_num(-2*t*x*y*(t - 1)/(r*(-2*t*r + t + 1)**2), nan=0.0, posinf=0.0, neginf=0.0)

        j11[~mask] = 1
        j22[~mask] = 1
        j12[~mask] = 0
        j21[~mask] = 0

        return torch.stack((torch.stack((j11, j12), dim=-1), torch.stack((j21, j22), dim=-1)), dim=-2)

    def dfdt(self, X: Float[torch.Tensor, "*batch 2"], theta: Float[torch.Tensor, "n"]) -> Float[torch.Tensor, "*batch 2 n"]:
        r= torch.norm(X, dim=-1)

        x = X[..., 0]
        y = X[..., 1]

        s = theta[0]
        t = theta[1]

        dfdt = torch.zeros((*X.shape, self.numParameters()), device=X.device)

        mask = r < self.h

        dfdt[..., 0, 0] = torch.nan_to_num(2*x*(x**4 + x**2 *(2*y**2 - 1) + y**2*(y**2 - 1)))/(r * (s*(2*x**2 + 2*y**2 -1) - 1)**2)
        dfdt[..., 1, 1] = torch.nan_to_num(2*y*(x**4 + x**2 *(2*y**2 - 1) + y**2*(y**2 - 1)))/(r * (t*(2*x**2 + 2*y**2 -1) - 1)**2)
        dfdt[..., 0, 1] = 0
        dfdt[..., 1, 0] = 0

        dfdt[~mask] = 0

        return dfdt

    def numParameters(self) -> int:
        return 2

    def getNeutralParameter(self) -> Float[torch.Tensor, "n"]:
        return torch.Tensor([0.0, 0.0])
    
class RectangularDeformation(Deformation):
    """
    From https://math.stackexchange.com/questions/459872/adjustable-sigmoid-curve-s-curve-from-0-0-to-1-1
    Also see https://dhemery.github.io/DHE-Modules/technical/sigmoid/
    Here we actually only need the upper half of the sigmoid
    """

    def __init__(self, h=1.0) -> None:
        super().__init__()
        self.h = h

    def __call__(self, X: Float[torch.Tensor, "*batch 2"], theta: Float[torch.Tensor, "n"]) -> tuple[Float[torch.Tensor, "*batch 2"], Float[torch.Tensor, "*batch 2 2"]]:
        r_x = torch.abs(X[..., 0])
        r_y = torch.abs(X[..., 1])

        s = theta[0]
        t = theta[1]

        mask_x = r_x < self.h
        mask_y = r_y < self.h

        r_t_x = mask_x * (s - 1) / (2*s*r_x - s - 1)
        r_t_y = mask_y * (t - 1) / (2*t*r_y - t - 1)

        r_t = torch.stack((r_t_x * X[..., 0] + ~mask_x*X[..., 0], r_t_y * X[..., 1] + ~mask_y*X[..., 1]), dim=-1)

        return r_t, self.jacobian(X, theta)

    def jacobian(self, X: Float[torch.Tensor, "*batch 2"], theta: Float[torch.Tensor, "n"]) -> Float[torch.Tensor, "*batch 2 2"]:
        r_x = torch.abs(X[..., 0])
        r_y = torch.abs(X[..., 1])

        x = X[..., 0]
        y = X[..., 1]
        s = theta[0]
        t = theta[1]

        mask_x = r_x < self.h
        mask_y = r_y < self.h

        j11 = torch.nan_to_num(
            -2*s*x*(s*r_x - r_x)*torch.sign(x)/((2*s*r_x - s - 1)**2*r_x) +
            x*(s*torch.sign(x) - torch.sign(x))/((2*s*r_x - s - 1)*r_x) +
            (s*r_x - r_x)/((2*s*r_x - s - 1)*r_x) -
            (s*r_x - r_x)*torch.sign(x)/(x*(2*s*r_x - s - 1)), nan=1, posinf=1, neginf=1)
        j22 = torch.nan_to_num(
            -2*t*y*(t*r_y - r_y)*torch.sign(y)/((2*t*r_y - t - 1)**2*r_y) +
            y*(t*torch.sign(y) - torch.sign(y))/((2*t*r_y - t - 1)*r_y) +
            (t*r_y - r_y)/((2*t*r_y - t - 1)*r_y) -
            (t*r_y - r_y)*torch.sign(y)/(y*(2*t*r_y - t - 1)), nan=1, posinf=1, neginf=1)
        j21 = torch.zeros_like(j11)
        j12 = torch.zeros_like(j11)

        j11[~mask_x] = 1
        j22[~mask_y] = 1

        return torch.stack((torch.stack((j11, j12), dim=-1), torch.stack((j21, j22), dim=-1)), dim=-2)

    def dfdt(self, X: Float[torch.Tensor, "*batch 2"], theta: Float[torch.Tensor, "n"]) -> Float[torch.Tensor, "*batch 2 n"]:
        r_x = torch.abs(X[..., 0])
        r_y = torch.abs(X[..., 1])

        x = X[..., 0]
        y = X[..., 1]

        s = theta[0]
        t = theta[1]

        dfdt = torch.zeros((*X.shape, self.numParameters()), device=X.device)

        mask_x = r_x < self.h
        mask_y = r_y < self.h

        dfdt[..., 0, 0] = mask_x * torch.nan_to_num(x*(1 - 2*torch.abs(x))*(s*torch.abs(x) - torch.abs(x))/((2*s*torch.abs(x) - s - 1)**2*torch.abs(x)) + x/(2*s*torch.abs(x) - s - 1))
        dfdt[..., 0, 1] = 0
        dfdt[..., 1, 1] = mask_y * torch.nan_to_num(y*(1 - 2*torch.abs(y))*(t*torch.abs(y) - torch.abs(y))/((2*t*torch.abs(y) - t - 1)**2*torch.abs(y)) + y/(2*t*torch.abs(y) - t - 1))
        dfdt[..., 1, 0] = 0

        return dfdt

    def numParameters(self) -> int:
        return 2

    def getNeutralParameter(self) -> Float[torch.Tensor, "n"]:
        return torch.Tensor([0.0, 0.0])
    
    def inverse(self, X: Float[torch.Tensor, "*batch 2"], theta: Float[torch.Tensor, "n"]) -> Float[torch.Tensor, "*batch 2"]:
        return self(X, -theta)[0]

class IdentityDeformation(Deformation):
    def __init__(self):
        super().__init__()

    def __call__(self, X: Float[torch.Tensor, "*batch 2"], theta: Float[torch.Tensor, "n"]) -> tuple[Float[torch.Tensor, "*batch 2"], Float[torch.Tensor, "*batch 2 2"]]:
        return X, self.jacobian(X, theta) 
        
    def dfdt(self, X: Float[torch.Tensor, "*batch 2"], theta: Float[torch.Tensor, "n"]) -> Float[torch.Tensor, "*batch 2 n"]:
        return torch.Tensor([], device=X.device)

    def jacobian(self, X: Float[torch.Tensor, "*batch 2"], theta: Float[torch.Tensor, "n"]) -> Float[torch.Tensor, "*batch 2 2"]:
        X = X.view(-1, 2)
        jac = torch.zeros(X.shape[0], 2, 2, device=X.device)
        jac[..., 0, 0] = 1
        jac[..., 1, 1] = 1
        return jac

    def numParameters(self) -> int:
        return 0

    def getNeutralParameter(self) -> Float[torch.Tensor, "n"]:
        return torch.Tensor([])

    def inverse(self, X: Float[torch.Tensor, "*batch 2"], theta: Float[torch.Tensor, "n"]) -> Float[torch.Tensor, "*batch 2"]:
        return X

class EllipticSigmoidDeformation(Deformation):
    """
    Same as `AnisotropicHalfNormalTunableSigmoidDeformation` but with an elliptic mapping from square to disk first
    From https://math.stackexchange.com/questions/459872/adjustable-sigmoid-curve-s-curve-from-0-0-to-1-1
    Also see https://dhemery.github.io/DHE-Modules/technical/sigmoid/
    Here we actually only need the upper half of the sigmoid
    """

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, X: Float[torch.Tensor, "*batch 2"], theta: Float[torch.Tensor, "n"]) -> tuple[Float[torch.Tensor, "*batch 2"], Float[torch.Tensor, "*batch 2 2"]]:
        x = X[..., 0]
        y = X[..., 1]
        s = theta[0]
        t = theta[1]

        jac_right = self.elliptic_jac(x, y) # J_PSI(x)

        # apply PSI
        x, y = self.square_to_disk(x, y)
        # x = PSI(x)

        # compute J_phi(PSI(x))
        r = torch.sqrt(x**2 + y**2)
        r2 = r**2

        j11 = torch.nan_to_num((-2*s*x**2*(s - 1)*r2**(7/2) + (1 - s)*r2**4*(-2*s*r + s + 1))/(r2**4*(-2*s*r + s + 1)**2), nan=1.0, posinf=1.0, neginf=1.0)
        j22 = torch.nan_to_num((-2*t*y**2*(t - 1)*r2**(7/2) + (1 - t)*r2**4*(-2*t*r + t + 1))/(r2**4*(-2*t*r + t + 1)**2), nan=1.0, posinf=1.0, neginf=1.0)
        j12 = torch.nan_to_num(-2*s*x*y*(s - 1)/(r*(-2*s*r + s + 1)**2), nan=0.0, posinf=0.0, neginf=0.0)
        j21 = torch.nan_to_num(-2*t*x*y*(t - 1)/(r*(-2*t*r + t + 1)**2), nan=0.0, posinf=0.0, neginf=0.0)
        jac_middle = torch.stack((torch.stack((j11, j12), dim=-1), torch.stack((j21, j22), dim=-1)), dim=-2) # J_phi(PSI(x))
        
        # apply phi
        x = (x*(s-1)) / (2*s*r - s - 1)
        y = (y*(t-1)) / (2*t*r - t - 1)
        # x = phi(PSI(x))

        # apply inverse PSI
        x, y = self.disk_to_square(x, y)
        # x = PSI_inv(phi(PSI(x))
        jac_left = self.inv_2x2(self.elliptic_jac(x, y))

        jac = jac_left @ jac_middle @ jac_right

        return torch.stack((x, y), dim=-1), jac

    def square_to_disk(self, x, y):
        u = x * torch.sqrt(1. - y**2/2.)
        v = y * torch.sqrt(1. - x**2/2.)

        return u, v

    def disk_to_square(self, x, y):
        import math
        u = 0.5 * torch.sqrt(2.0 + x**2 - y**2 + 2.0*math.sqrt(2)*x) - 0.5 * torch.sqrt(2.0 + x**2 - y**2 - 2.0*math.sqrt(2)*x)
        v = 0.5 * torch.sqrt(2.0 - x**2 + y**2 + 2.0*math.sqrt(2)*y) - 0.5 * torch.sqrt(2.0 - x**2 + y**2 - 2.0*math.sqrt(2)*y)
        return u, v


    def elliptic_jac(self, x, y) -> Float[torch.Tensor, "*batch 2 2"]:
        j11 = torch.nan_to_num(torch.sqrt(1. - ((y**2) / 2.)), nan=1.0, posinf=1.0, neginf=1.0)
        j22 = torch.nan_to_num(torch.sqrt(1. - ((x**2) / 2.)), nan=1.0, posinf=1.0, neginf=1.0)
        j12 = torch.nan_to_num(-((x*y) / (torch.sqrt(4 - 2*y**2))), nan=0.0, posinf=0.0, neginf=0.0)
        j21 = torch.nan_to_num(-((x*y) / (torch.sqrt(4 - 2*x**2))), nan=0.0, posinf=0.0, neginf=0.0)

        jac = torch.stack((torch.stack((j11, j12), dim=-1), torch.stack((j21, j22), dim=-1)), dim=-2)
        return jac

    def inv_2x2(self, M):
        j11 = M[..., 0, 0]
        j22 = M[..., 1, 1]
        j21 = M[..., 1, 0]
        j12 = M[..., 0, 1]

        det_inv = 1.0 / (j11*j22 - j12*j21)
        
        j11_inv = torch.nan_to_num(j22 * det_inv, nan=1.0, posinf=1.0, neginf=1.0)
        j22_inv = torch.nan_to_num(j11 * det_inv, nan=1.0, posinf=1.0, neginf=1.0)
        j21_inv = torch.nan_to_num(-j21 * det_inv, nan=0.0, posinf=0.0, neginf=0.0)
        j12_inv = torch.nan_to_num(-j12 * det_inv, nan=0.0, posinf=0.0, neginf=0.0)

        jac_inv = torch.stack((torch.stack((j11_inv, j12_inv), dim=-1), torch.stack((j21_inv, j22_inv), dim=-1)), dim=-2)
        
        return jac_inv

    def jacobian(self, X: Float[torch.Tensor, "*batch 2"], theta: Float[torch.Tensor, "n"]) -> Float[torch.Tensor, "*batch 2 2"]:
        return self(X, theta)[1]

    def dfdt(self, X: Float[torch.Tensor, "*batch 2"], theta: Float[torch.Tensor, "n"]) -> Float[torch.Tensor, "*batch 2 n"]:
        x = X[..., 0]
        y = X[..., 1]
        s = theta[0]
        t = theta[1]

        # apply PSI
        x, y = self.square_to_disk(x, y)
        # x = PSI(x)
        r = torch.sqrt(x**2 + y**2)

        dfdt = torch.zeros((*X.shape, self.numParameters()), device=X.device)

        dfdt[..., 0, 0] = torch.nan_to_num(2*x*(x**4 + x**2 *(2*y**2 - 1) + y**2*(y**2 - 1)))/(r * (s*(2*x**2 + 2*y**2 -1) - 1)**2)
        dfdt[..., 1, 1] = torch.nan_to_num(2*y*(x**4 + x**2 *(2*y**2 - 1) + y**2*(y**2 - 1)))/(r * (t*(2*x**2 + 2*y**2 -1) - 1)**2)
        dfdt[..., 0, 1] = 0
        dfdt[..., 1, 0] = 0

        # apply phi
        x = (x*(s-1)) / (2*s*r - s - 1)
        y = (y*(t-1)) / (2*t*r - t - 1)
        # x = phi(PSI(x))

        # apply inverse PSI
        x, y = self.disk_to_square(x, y)
        # x = PSI_inv(phi(PSI(x))
        jac = self.inv_2x2(self.elliptic_jac(x, y))

        dfdt = dfdt @ jac

        dfdt[torch.isnan(dfdt)] = 0

        return dfdt

    def getAllDerivatives(self, X: Float[torch.Tensor, "*batch 2"], theta: Float[torch.Tensor, "n"]) -> tuple[Float[torch.Tensor, "*batch 2"], Float[torch.Tensor, "*batch 2 2"], Float[torch.Tensor, "*batch 2 n"]]:
        x = X[..., 0]
        y = X[..., 1]
        s = theta[0]
        t = theta[1]

        jac_right = self.elliptic_jac(x, y) # J_PSI(x)

        # apply PSI
        x, y = self.square_to_disk(x, y)
        # x = PSI(x)

        # compute J_phi(PSI(x))
        r = torch.sqrt(x**2 + y**2)
        r2 = r**2
        
        j11 = torch.nan_to_num((-2*s*x**2*(s - 1)*r2**(7/2) + (1 - s)*r2**4*(-2*s*r + s + 1))/(r2**4*(-2*s*r + s + 1)**2), nan=1.0, posinf=1.0, neginf=1.0)
        j22 = torch.nan_to_num((-2*t*y**2*(t - 1)*r2**(7/2) + (1 - t)*r2**4*(-2*t*r + t + 1))/(r2**4*(-2*t*r + t + 1)**2), nan=1.0, posinf=1.0, neginf=1.0)
        j12 = torch.nan_to_num(-2*s*x*y*(s - 1)/(r*(-2*s*r + s + 1)**2), nan=0.0, posinf=0.0, neginf=0.0)
        j21 = torch.nan_to_num(-2*t*x*y*(t - 1)/(r*(-2*t*r + t + 1)**2), nan=0.0, posinf=0.0, neginf=0.0)
        jac_middle = torch.stack((torch.stack((j11, j12), dim=-1), torch.stack((j21, j22), dim=-1)), dim=-2) # J_phi(PSI(x))

        # compute dfdt
        dfdt = torch.zeros((*X.shape, self.numParameters()), device=X.device)

        dfdt[..., 0, 0] = torch.nan_to_num(2*x*(x**4 + x**2 *(2*y**2 - 1) + y**2*(y**2 - 1)))/(r * (s*(2*x**2 + 2*y**2 -1) - 1)**2)
        dfdt[..., 1, 1] = torch.nan_to_num(2*y*(x**4 + x**2 *(2*y**2 - 1) + y**2*(y**2 - 1)))/(r * (t*(2*x**2 + 2*y**2 -1) - 1)**2)
        dfdt[..., 0, 1] = 0
        dfdt[..., 1, 0] = 0

        # apply phi
        x = (x*(s-1)) / (2*s*r - s - 1)
        y = (y*(t-1)) / (2*t*r - t - 1)
        # x = phi(PSI(x))

        # apply inverse PSI
        x, y = self.disk_to_square(x, y)
        # x = PSI_inv(phi(PSI(x))
        jac_left = self.inv_2x2(self.elliptic_jac(x, y))

        dfdt = dfdt @ jac_left
        jac = jac_left @ jac_middle @ jac_right

        dfdt[torch.isnan(dfdt)] = 0

        return torch.stack((x, y), dim=-1), jac, dfdt

    def numParameters(self) -> int:
        return 2

    def getNeutralParameter(self) -> Float[torch.Tensor, "n"]:
        return torch.Tensor([0.0, 0.0])

    def inverse(self, X: Float[torch.Tensor, "*batch 2"], theta: Float[torch.Tensor, "n"]) -> Float[torch.Tensor, "*batch 2"]:
        # custom implementation because we can speed up the inverse for 2x2 jacobians
        x = X.clone()

        for _ in range(5):
            curr, jac = self(x, theta)
            jac = self.inv_2x2(jac)
            x = x - torch.matmul(jac, curr.unsqueeze(-1) - X.unsqueeze(-1)).squeeze(-1)
        return x


class ComposedDeformation(Deformation):
    def __init__(self, *deforms: Deformation, freeze_indices: Optional[list[int]] = None, init_params: Optional[list[float]] = None) -> None:
        super().__init__()
        self.deforms = list(deforms)

        self.theta_bounds = [0]
        self.total_params = 0
        self.freeze_indices = set(freeze_indices) if freeze_indices else None
        for deform in self.deforms:
            num_params = deform.numParameters()
            self.total_params += num_params
            self.theta_bounds.append(self.theta_bounds[-1] + num_params)

        self.frozen_params = []
        if init_params:
            for i in range(len(self.deforms)):
                if self.freeze_indices and i in self.freeze_indices:
                    l, r = self.theta_bounds[i:i+2]
                    self.frozen_params = init_params[l:r]

    def __call__(self, X: Float[torch.Tensor, "*batch 2"], theta: Float[torch.Tensor, "n"]) -> tuple[Float[torch.Tensor, "*batch 2"], Float[torch.Tensor, "*batch 2 2"]]:
        jac = torch.zeros((*X.shape, 2), device=X.device)
        jac[..., 0, 0] = 1
        jac[..., 1, 1] = 1

        for i in range(len(self.deforms)):
            deform = self.deforms[i]
            l, r = self.theta_bounds[i:i+2]
            t = theta[l:r]
            if self.freeze_indices and self.frozen_params and i in self.freeze_indices:
                t = torch.Tensor(self.frozen_params).to(X.device)

            X, local_jac = deform(X, t)
            jac = local_jac @ jac

        return X, jac

    def jacobian(self, X: Float[torch.Tensor, "*batch 2"], theta: Float[torch.Tensor, "n"]) -> Float[torch.Tensor, "*batch 2 2"]:
        return self(X, theta)[1]

    def dfdt(self, X: Float[torch.Tensor, "*batch 2"], theta: Float[torch.Tensor, "n"]) -> Float[torch.Tensor, "*batch 2 n"]:

        dfdt = torch.zeros((*X.shape, self.numParameters()), device=X.device)

        for i in range(len(self.deforms)):
            deform = self.deforms[i]
            l, r = self.theta_bounds[i:i+2]
            t = theta[l:r]
            if self.freeze_indices and self.frozen_params and i in self.freeze_indices:
                t = torch.Tensor(self.frozen_params).to(X.device)

            local_dfdt = deform.dfdt(X, t)
            dfdt[..., l:r] = local_dfdt
            X, local_jac = deform(X, t)

            for j in range(i):
                ll, rr = self.theta_bounds[j:j+2]
                dfdt[..., ll:rr] = local_jac @ dfdt[..., ll:rr]

        return dfdt

    def getAllDerivatives(self, X: Float[torch.Tensor, "*batch 2"], theta: Float[torch.Tensor, "n"]) -> tuple[Float[torch.Tensor, "*batch 2"], Float[torch.Tensor, "*batch 2 2"], Float[torch.Tensor, "*batch 2 n"]]:
        dfdt = torch.zeros((*X.shape, self.numParameters()), device=X.device)
        
        total_jac = torch.zeros((*X.shape, 2), device=X.device)
        total_jac[..., 0, 0] = 1
        total_jac[..., 1, 1] = 1

        for i in range(len(self.deforms)):
            deform = self.deforms[i]
            l, r = self.theta_bounds[i:i+2]
            t = theta[l:r]
            if self.freeze_indices and self.frozen_params and i in self.freeze_indices:
                t = torch.Tensor(self.frozen_params).to(X.device)

            local_dfdt = deform.dfdt(X, t)
            X, local_jac, local_dfdt = deform.getAllDerivatives(X, t)
            dfdt[..., l:r] = local_dfdt
            total_jac = local_jac @ total_jac

            for j in range(i):
                ll, rr = self.theta_bounds[j:j+2]
                dfdt[..., ll:rr] = local_jac @ dfdt[..., ll:rr]

        return X, total_jac, dfdt

    def numParameters(self) -> int:
        return self.total_params

    def getNeutralParameter(self) -> Float[torch.Tensor, "n"]:
        param = torch.zeros(self.total_params)
        for i in range(len(self.deforms)):
            l, r = self.theta_bounds[i:i+2]
            param[l:r] = self.deforms[i].getNeutralParameter()
        return param

    def getGradMask(self, device=None) -> Optional[Float[torch.Tensor, "n"]]:
        if self.freeze_indices is None:
            return None

        mask = torch.ones(self.total_params, device=device, dtype=torch.int)
        for i in range(len(self.deforms)):
            l, r = self.theta_bounds[i:i+2]
            if i in self.freeze_indices:
                mask[l:r] = 0
        return mask

    def inverse(self, X: Float[torch.Tensor, "*batch 2"], theta: Float[torch.Tensor, "n"]) -> Float[torch.Tensor, "*batch 2"]:
        for i in reversed(range(len(self.deforms))):
            deform = self.deforms[i]
            l, r = self.theta_bounds[i:i+2]
            t = theta[l:r]
            if self.freeze_indices and self.frozen_params and i in self.freeze_indices:
                t = torch.Tensor(self.frozen_params).to(X.device)
            X = deform.inverse(X, t)

        return X
