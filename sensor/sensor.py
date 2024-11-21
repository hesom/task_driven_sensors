from typing import Optional
from jaxtyping import Float
import torch
import torch.nn as nn
import torch.nn.functional as F

import einops

from .deformations import RectangularDeformation, Deformation, IdentityDeformation

class Simulator(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
        img: Float[torch.Tensor, "batch 3 height width"],
        deformation: Deformation,
        t: Float[torch.Tensor, "n"],
        sensor_size: tuple[int, int],
        spp=16,
        ) -> Float[torch.Tensor, "batch 3 sensor_height sensor_width"]:
        """
        Simulate the sensor
        :param img: input image, shape (batch_size, 3, H, W)
        :param deformation: diffeomorphism from [-1, 1]^2 -> [-1, 1]^2
        :param t: parameter controlling the deformation
        :param sensor_size: size of the sensor in pixels
        :param spp: samples per pixel
        :return: sensor image, shape (batch_size, 3, sensor_size[0], sensor_size[1])
        """
        ctx.deformation = deformation   # deformation is a function
        ctx.sensor_size = sensor_size
        ctx.spp = spp
        b = img.shape[0]

        # we use arange because linspace is inclusive at the end point and we want to add subpixel jitter
        step_x = (1 + 1) / sensor_size[1]
        step_y = (1 + 1) / sensor_size[0]
        pixel_pos_x = torch.arange(-1, 1, step_x, device=img.device)
        pixel_pos_y = torch.arange(-1, 1, step_y, device=img.device)
        pixel_pos_x, pixel_pos_y = torch.meshgrid(pixel_pos_x, pixel_pos_y, indexing='xy')
        pixel_pos = torch.stack((pixel_pos_x, pixel_pos_y), dim=2)  # (sensor_size[0], sensor_size[1], 2)

        # add subpixel jitter
        jitter = torch.rand((spp, sensor_size[0], sensor_size[1], 2), device=img.device)
        jitter[..., 0] *= step_x
        jitter[..., 1] *= step_y
        pixel_pos = pixel_pos.unsqueeze(0) + jitter # (spp, sensor_size[0], sensor_size[1], 2)

        # deform the sampling positions
        # if the deformation was chosen correctly, all values in pixel_pos should still be between -1 and 1
        pixel_pos, jac = deformation(pixel_pos, t) # jac has shape (spp, sensor_size[0], sensor_size[1], 2, 2)
        det = jac[..., 0, 0]*jac[..., 1, 1] - jac[..., 0, 1]*jac[..., 1, 0]

        # tile the image and the sampling positions for batching
        # TODO the following operations might be very memory inefficient to the point where it justifies a custom cuda kernel
        pixel_pos = einops.repeat(pixel_pos, 'spp h w c -> (b spp) h w c', b=b)
        det = einops.repeat(det, 'spp h w -> (b spp) 1 h w', b=b)
        img_tiled = einops.repeat(img, 'b c h w -> (b spp) c h w', spp=spp)

        # sample the image at deformed positions
        samples = F.grid_sample(img_tiled, pixel_pos, mode='bilinear', padding_mode='border', align_corners=False)
        samples = samples * det
        samples = einops.reduce(samples, '(b spp) c h w -> b c h w', 'mean', b=b)

        # normalize by pixel area = integral of determinant
        area = einops.reduce(det, '(b spp) 1 h w -> b 1 h w', 'mean', b=b)
        samples /= area

        # area can be reused in backward
        ctx.save_for_backward(img, samples, t, area)
        ctx.save_for_forward(img, samples, t, area)

        return samples

    @staticmethod
    def grad(img, sensor_img, t, area, deformation, sensor_size, spp):
        b = img.shape[0]
        spp = spp // 4 * 4

        step_x = (1 + 1) / sensor_size[1]
        step_y = (1 + 1) / sensor_size[0]
        pixel_pos_x = torch.arange(-1, 1, step_x, device=img.device)
        pixel_pos_y = torch.arange(-1, 1, step_y, device=img.device)
        pixel_pos_x, pixel_pos_y = torch.meshgrid(pixel_pos_x, pixel_pos_y, indexing='xy')
        pixel_pos = torch.stack((pixel_pos_x, pixel_pos_y), dim=2)  # (sensor_size[0], sensor_size[1], 2)

        # random positions on the pixel boundaries
        jitter = torch.rand((spp, sensor_size[0], sensor_size[1]), device=img.device)
        boundary_pos = einops.repeat(pixel_pos, 'h w c -> spp h w c', spp=spp).contiguous()
        # top
        boundary_pos[:spp//4, :, :, 0] += jitter[:spp//4, ...] * step_x
        #left
        boundary_pos[spp//4:spp//2, :, :, 1] += jitter[spp//4:spp//2, ...] * step_y
        # bottom
        boundary_pos[spp//2:3*spp//4, :, :, 0] += jitter[spp//2:3*spp//4, ...] * step_x
        boundary_pos[spp//2:3*spp//4, :, :, 1] += step_y
        # right
        boundary_pos[3*spp//4:, :, :, 1] += jitter[3*spp//4:, ...] * step_y
        boundary_pos[3*spp//4:, :, :, 0] += step_x

        # compute forward unit vector on the boundary in clockwise circular order
        # origin is top left corner and y points down
        # ----------------->
        # ^                |
        # |                |
        # |                |
        # |                |
        # |                |
        # |                v
        # <-----------------
        forward_vector = torch.zeros_like(boundary_pos)
        # top
        forward_vector[:spp//4, :, :, 0] = 4 / step_x
        # left
        forward_vector[spp//4:spp//2, :, :, 1] = -4 / step_y
        # bottom
        forward_vector[spp//2:3*spp//4, :, :, 0] = -4 / step_x
        # right
        forward_vector[3*spp//4:, :, :, 1] = 4 / step_y

        boundary_pos, boundary_jac, drdt = deformation.getAllDerivatives(boundary_pos, t)

        # transform forward vector with jacobian
        forward_vector = (boundary_jac @ forward_vector.unsqueeze(-1)).squeeze(-1)

        velocity = torch.norm(forward_vector, dim=-1)

        # outwards pointing normal on the boundary
        normal = torch.stack((forward_vector[..., 1], -forward_vector[..., 0]), dim=-1)
        normal = normal / torch.norm(normal, dim=-1, keepdim=True)

        if deformation.numParameters() == 1 and drdt.shape[-1] != 1:
            boundary_weight = torch.einsum('shwc,shwc->shw', normal, drdt).unsqueeze(-1) # just dot product
        else:
            boundary_weight = torch.einsum('shwc,shwcq->shwq', normal, drdt)

        # Keep in mind that this function should compute the derivative of each pixel value with respect to t
        # Through the product rule this can be separated into two parts
        # 1. The derivative of the pixel area with respect to t
        # 2. The derivative of the captured radiance with respect to t

        # --------------------------------------------------
        # 1. Derivative of the pixel area with respect to t
        # The pixel area is the integral of the determinant of the deformation
        # The derivative is the integral of the determinant over the pixel boundary

        area_boundary_change = boundary_weight * velocity.unsqueeze(-1)
        area_boundary_change = einops.reduce(area_boundary_change, 'spp h w q-> 1 1 h w q', 'mean')
        dAreaReciprocal = -area_boundary_change / area.unsqueeze(-1)

        # --------------------------------------------------

        # 2. Derivative of the captured radiance with respect to t
        # The captured radiance is the integral of the radiance field over the pixel area
        # The derivative is the integral of the radiance field over the pixel boundary

        tiled_boundary_pos = einops.repeat(boundary_pos, 'spp h w c -> (b spp) h w c', b=b)
        tiled_boundary_weight = einops.repeat(boundary_weight, 'spp h w q-> (b spp) 1 h w q', b=b)
        tiled_img = einops.repeat(img, 'b c h w -> (b spp) c h w', spp=spp)
        tiled_velocity = einops.repeat(velocity, 'spp h w -> (b spp) 1 h w 1', b=b)
        radiance = torch.nn.functional.grid_sample(tiled_img, tiled_boundary_pos, mode='bilinear', padding_mode='border', align_corners=False)
        radiance = radiance.unsqueeze(-1) * tiled_boundary_weight * tiled_velocity
        dRadiance = einops.reduce(radiance, '(b spp) c h w q-> b c h w q', 'mean', spp=spp)

        # --------------------------------------------------

        # Combine everything according to the product rule
        dPixel = dRadiance / area.unsqueeze(-1) + dAreaReciprocal * sensor_img.unsqueeze(-1)

        return dPixel

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output_pixel):
        img, sensor_img, t, area = ctx.saved_tensors
        deformation: Deformation = ctx.deformation
        sensor_size = ctx.sensor_size
        spp = ctx.spp
        
        grad_t = None
        if ctx.needs_input_grad[2]:
            dPixel = Simulator.grad(img, sensor_img, t, area, deformation, sensor_size, spp)
            grad_t = grad_output_pixel.unsqueeze(-1) * dPixel
            grad_mask = deformation.getGradMask(device=grad_t.device)
            if grad_mask is not None:
                grad_t[grad_mask.expand_as(grad_t) == 0] = 0

        return None, None, grad_t, None, None, None

    @staticmethod
    def jvp(ctx, *gradInputs):
        img, sensor_img, t, area = ctx.saved_tensors
        deformation = ctx.deformation
        sensor_size = ctx.sensor_size
        spp = ctx.spp

        dPixel = Simulator.grad(img, sensor_img, t, area, deformation, sensor_size, spp)
        grad_t = torch.einsum('...q,...q->...', gradInputs[2], dPixel)

        return grad_t

simulate = Simulator.apply

class FixedFoveatedSensor(nn.Module):
    """Sensor module. Simulates the sensor by approximating the radiance field with an image"""
    def __init__(self, sensor_size: tuple[int, int], t_init: Optional[list[float]] = None,
                 spp: int = 16, deform: Deformation = RectangularDeformation(1.0)):
        super().__init__()
        if (not isinstance(sensor_size, tuple)
            and not isinstance(sensor_size, list)):
            sensor_size = (sensor_size, sensor_size)
        if len(sensor_size) != 2:
            raise ValueError(f"sensor_size must be a tuple of length 2, got {sensor_size}")

        self.sensor_size = sensor_size
        self.deform = deform
        self.numParameters = deform.numParameters()
        if t_init is None:
            raise NotImplementedError
        self.register_buffer('t', torch.Tensor([t_init]).squeeze())
        self.spp = spp
        self.t_transform = lambda x: x

    def forwardDeform(self, positions: Float[torch.Tensor, "*batch 2"]) -> Float[torch.Tensor, "*batch 2"]:
        assert(isinstance(self.t, torch.Tensor))
        return self.deform(positions, self.t)[0]
    
    def backwardDeform(self, positions: Float[torch.Tensor, "*batch 2"]) -> Float[torch.Tensor, "batch 2"]:
        assert(isinstance(self.t, torch.Tensor))
        return self.deform.inverse(positions, self.t)

    def thetaToString(self):
        with torch.no_grad():
            t = self.t.cpu().numpy().tolist()
            return f"{t}"

    def forward(self, img: Float[torch.Tensor, "batch 3 height width"]) -> Float[torch.Tensor, "batch 3 sensor_height sensor_width"]:
        """
        Simulate the sensor
        :param img: high-res input image, shape (batch_size, 3, H, W)
        :return: simulated sensor image, shape (batch_size, 3, sensor_size[0], sensor_size[1])
        """
        radiance = simulate(img, self.deform, self.t, self.sensor_size, self.spp)
        assert(isinstance(radiance, torch.Tensor))
        return radiance

class FoveatedSensor(nn.Module):
    """Sensor module. Simulates the sensor by approximating the radiance field with an image"""
    def __init__(self, sensor_size: tuple[int, int], t_init: Optional[list[float]] = None,
                 spp: int = 16, deform: Deformation = RectangularDeformation(1.0),
                 constrain_t: bool = True, constrain_factor: float = 0.6):
        super().__init__()
        if (not isinstance(sensor_size, tuple)
            and not isinstance(sensor_size, list)):
            sensor_size = (sensor_size, sensor_size)
        if len(sensor_size) != 2:
            raise ValueError(f"sensor_size must be a tuple of length 2, got {sensor_size}")

        self.sensor_size = sensor_size
        self.deform = deform
        self.numParameters = deform.numParameters()
        if t_init is None:
            t = deform.getNeutralParameter()
            self.t = nn.Parameter(t)
        else:
            self.t = nn.Parameter(torch.Tensor([t_init]).squeeze())
        self.spp = spp
        self.constrain_t = constrain_t
        self.constrain_factor = constrain_factor
        self.t_transform = lambda x: x

        if self.constrain_t:
            self.t_transform = lambda x: constrain_factor*torch.tanh(x)


    def forwardDeform(self, positions: Float[torch.Tensor, "*batch 2"]) -> Float[torch.Tensor, "*batch 2"]:
        return self.deform(positions, self.t_transform(self.t))[0]
    
    def backwardDeform(self, positions: Float[torch.Tensor, "*batch 2"]) -> Float[torch.Tensor, "*batch 2"]:
        return self.deform.inverse(positions, self.t_transform(self.t))

    def thetaToString(self):
        with torch.no_grad():
            t = self.t_transform(self.t) 
            t = t.cpu().numpy().tolist()
            return f"{t}"

    def forward(self, img: Float[torch.Tensor, "batch 3 height width"]) -> Float[torch.Tensor, "batch 3 sensor_height sensor_width"]:
        """
        Simulate the sensor
        :param img: high-res input image, shape (batch_size, 3, H, W)
        :return: simulated sensor image, shape (batch_size, 3, sensor_size[0], sensor_size[1])
        """
        radiance = simulate(img, self.deform, self.t_transform(self.t), self.sensor_size, self.spp)
        assert(isinstance(radiance, torch.Tensor))
        return radiance

class UniformSensor(nn.Module):
    """Sensor module. Simulates the sensor by approximating the radiance field with an image"""
    def __init__(self, sensor_size: tuple[int, int], 
                 spp: int = 16
                 ):
        super().__init__()
        if (not isinstance(sensor_size, tuple)
            and not isinstance(sensor_size, list)):
            sensor_size = (sensor_size, sensor_size)
        if len(sensor_size) != 2:
            raise ValueError(f"sensor_size must be a tuple of length 2, got {sensor_size}")

        self.sensor_size = sensor_size
        self.numParameters = 0 
        self.spp = spp

        self.deform = IdentityDeformation()

    def forwardDeform(self, positions: Float[torch.Tensor, "*batch 2"]) -> Float[torch.Tensor, "*batch 2"]:
        return positions
    
    def backwardDeform(self, positions: Float[torch.Tensor, "*batch 2"]) -> Float[torch.Tensor, "*batch 2"]:
        return positions

    def thetaToString(self):
            return "Uniform"

    def forward(self, img: Float[torch.Tensor, "batch 3 height width"]) -> Float[torch.Tensor, "batch 3 sensor_height sensor_width"]:
        """
        Simulate the sensor
        :param img: high-res input image, shape (batch_size, 3, H, W)
        :return: simulated sensor image, shape (batch_size, 3, sensor_size[0], sensor_size[1])
        """
        radiance = simulate(img, self.deform, torch.Tensor([]), self.sensor_size, self.spp)
        assert(isinstance(radiance, torch.Tensor))
        return radiance
