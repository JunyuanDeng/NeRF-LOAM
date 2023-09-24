
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy


class OptimizablePose(nn.Module):
    def __init__(self, init_pose):
        super().__init__()
        assert (isinstance(init_pose, torch.FloatTensor))
        self.register_parameter('data', nn.Parameter(init_pose))
        self.data.required_grad_ = True

    def copy_from(self, pose):
        self.data = deepcopy(pose.data)

    def matrix(self):
        Rt = torch.eye(4)
        Rt[:3, :3] = self.rotation()
        Rt[:3, 3] = self.translation()
        return Rt

    def rotation(self):
        w = self.data[3:]
        wx = self.skew_symmetric(w)
        theta = w.norm(dim=-1)[..., None, None]
        I = torch.eye(3, device=w.device, dtype=torch.float32)
        A = self.taylor_A(theta)
        B = self.taylor_B(theta)
        R = I+A*wx+B*wx@wx
        return R

    def translation(self,):
        return self.data[:3]

    @classmethod
    def log(cls, R, eps=1e-7):  # [...,3,3]
        trace = R[..., 0, 0]+R[..., 1, 1]+R[..., 2, 2]
        # ln(R) will explode if theta==pi
        theta = ((trace-1)/2).clamp(-1+eps, 1-eps).acos_()[..., None, None] % np.pi
        lnR = 1/(2*cls.taylor_A(theta)+1e-8) * (R-R.transpose(-2, -1))  # FIXME: wei-chiu finds it weird
        w0, w1, w2 = lnR[..., 2, 1], lnR[..., 0, 2], lnR[..., 1, 0]
        w = torch.stack([w0, w1, w2], dim=-1)
        return w

    @classmethod
    def from_matrix(cls, Rt, eps=1e-8):  # [...,3,4]
        R, u = Rt[:3, :3], Rt[:3, 3]
        w = cls.log(R)
        return OptimizablePose(torch.cat([u, w], dim=-1))

    @classmethod
    def skew_symmetric(cls, w):
        w0, w1, w2 = w.unbind(dim=-1)
        O = torch.zeros_like(w0)
        wx = torch.stack([
            torch.stack([O, -w2, w1], dim=-1),
            torch.stack([w2, O, -w0], dim=-1),
            torch.stack([-w1, w0, O], dim=-1)], dim=-2)
        return wx

    @classmethod
    def taylor_A(cls, x, nth=10):
        # Taylor expansion of sin(x)/x
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth+1):
            if i > 0:
                denom *= (2*i)*(2*i+1)
            ans = ans+(-1)**i*x**(2*i)/denom
        return ans

    @classmethod
    def taylor_B(cls, x, nth=10):
        # Taylor expansion of (1-cos(x))/x**2
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth+1):
            denom *= (2*i+1)*(2*i+2)
            ans = ans+(-1)**i*x**(2*i)/denom
        return ans

    @classmethod
    def taylor_C(cls, x, nth=10):
        # Taylor expansion of (x-sin(x))/x**3
        ans = torch.zeros_like(x)
        denom = 1.
        for i in range(nth+1):
            denom *= (2*i+2)*(2*i+3)
            ans = ans+(-1)**i*x**(2*i)/denom
        return ans


if __name__ == '__main__':
    before = torch.tensor([[-0.955421, 0.119616, - 0.269932, 2.655830],
                           [0.295248, 0.388339, - 0.872939, 2.981598],
                           [0.000408, - 0.913720, - 0.406343, 1.368648],
                           [0.000000, 0.000000, 0.000000, 1.000000]])
    pose = OptimizablePose.from_matrix(before)
    print(pose.rotation())
    print(pose.translation())
    after = pose.matrix()
    print(after)
    print(torch.abs((before-after)[:3, 3]))
