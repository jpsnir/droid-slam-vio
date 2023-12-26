import cv2
import torch
import lietorch

from collections import OrderedDict
from droid_net import DroidNet

import geom.projective_ops as pops
from modules.corr import CorrBlock

# from torchvision.utils import flow_to_image
from matplotlib import pyplot as plt


class MotionFilter:
    """This class is used to filter incoming frames and extract features"""

    def __init__(self, net, video, thresh=2.5, device="cuda:0"):
        # split net modules
        self.cnet = net.cnet
        self.fnet = net.fnet
        self.update = net.update

        self.video = video
        self.thresh = thresh
        self.device = device

        self.count = 0

        # mean, std for image normalization
        self.MEAN = torch.as_tensor([0.485, 0.456, 0.406], device=self.device)[
            :, None, None
        ]
        self.STDV = torch.as_tensor([0.229, 0.224, 0.225], device=self.device)[
            :, None, None
        ]

    @torch.cuda.amp.autocast(enabled=True)
    def __context_encoder(self, image):
        """context features"""
        net, inp = self.cnet(image).split([128, 128], dim=2)
        return net.tanh().squeeze(0), inp.relu().squeeze(0)

    @torch.cuda.amp.autocast(enabled=True)
    def __feature_encoder(self, image):
        """features for correlation volume"""
        return self.fnet(image).squeeze(0)

    @torch.cuda.amp.autocast(enabled=True)
    @torch.no_grad()
    def track(self, tstamp, image, depth=None, intrinsics=None):
        """main update operation - run on every frame in video"""

        Id = lietorch.SE3.Identity(
            1,
        ).data.squeeze()

        # The image shape has been downsampled by 8 times for
        # computing features using the RAFT network
        # The depth map will be of same dimensions.
        # This is the size that is being sent for update in DBA layer.
        ht = image.shape[-2] // 8
        wd = image.shape[-1] // 8

        # normalize images
        # The shape of the input tensor is [1, 3, 392, 488]
        # during normalization of the image, the shape is changed to [1, 1, 3, 392, 488]
        # which is the input to train.
        inputs = image[None, :, [2, 1, 0]].to(self.device) / 255.0
        inputs = inputs.sub_(self.MEAN).div_(self.STDV)

        # extract features
        gmap = self.__feature_encoder(inputs)

        print(f"Motion filter: Image Timestamp: {tstamp}")
        ### always add first frame to the depth video ###
        if self.video.counter.value == 0:
            net, inp = self.__context_encoder(inputs[:, [0]])
            self.net, self.inp, self.fmap = net, inp, gmap
            self.video.append(
                tstamp,
                image[0],
                Id,
                1.0,
                depth,
                intrinsics / 8.0,
                gmap,
                net[0, 0],
                inp[0, 0],
            )

        ### only add new frame if there is enough motion ###
        else:
            # index correlation volume
            coords0 = pops.coords_grid(ht, wd, device=self.device)[None, None]
            corr = CorrBlock(self.fmap[None, [0]], gmap[None, [0]])(coords0)

            # approximate flow magnitude using 1 update iteration
            _, delta, weight = self.update(self.net[None], self.inp[None], corr)

            # check motion magnitue / add new frame to video
            if delta.norm(dim=-1).mean().item() > self.thresh:
                print(
                    f"Motion filter: Adding frame timstamp - {tstamp} to video buffer"
                )
                self.count = 0
                net, inp = self.__context_encoder(inputs[:, [0]])
                self.net, self.inp, self.fmap = net, inp, gmap
                self.video.append(
                    tstamp,
                    image[0],
                    None,
                    None,
                    depth,
                    intrinsics / 8.0,
                    gmap,
                    net[0],
                    inp[0],
                )

            else:
                self.count += 1


# class MotionFilter:
#     """ This class is used to filter incoming frames and extract features """

#     def __init__(self, net, video, thresh=2.5, device="cuda:0"):

#         # split net modules
#         self.cnet = net.cnet
#         self.fnet = net.fnet
#         self.update = net.update

#         self.video = video
#         self.thresh = thresh
#         self.device = device

#         self.count = 0

#         # mean, std for image normalization
#         self.MEAN = torch.as_tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
#         self.STDV = torch.as_tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]

#     @torch.cuda.amp.autocast(enabled=True)
#     def __context_encoder(self, image):
#         """ context features """
#         x = self.cnet(image)
#         net, inp = self.cnet(image).split([128,128], dim=2)
#         return net.tanh().squeeze(0), inp.relu().squeeze(0)

#     @torch.cuda.amp.autocast(enabled=True)
#     def __feature_encoder(self, image):
#         """ features for correlation volume """
#         return self.fnet(image).squeeze(0)

#     @torch.cuda.amp.autocast(enabled=True)
#     @torch.no_grad()
#     def track(self, tstamp, image, depth=None, intrinsics=None):
#         """ main update operation - run on every frame in video """

#         Id = lietorch.SE3.Identity(1,).data.squeeze()
#         ht = image.shape[-2] // 8
#         wd = image.shape[-1] // 8

#         # normalize images
#         inputs = image[None, None, [2,1,0]].to(self.device) / 255.0
#         inputs = inputs.sub_(self.MEAN).div_(self.STDV)

#         # extract features
#         gmap = self.__feature_encoder(inputs)

#         ### always add first frame to the depth video ###
#         if self.video.counter.value == 0:
#             net, inp = self.__context_encoder(inputs)
#             self.net, self.inp, self.fmap = net, inp, gmap
#             self.video.append(tstamp, image, Id, 1.0, intrinsics / 8.0, gmap[0], net[0], inp[0])

#         ### only add new frame if there is enough motion ###
#         else:
#             # index correlation volume
#             coords0 = pops.coords_grid(ht, wd, device=self.device)[None,None]
#             corr = CorrBlock(self.fmap[None], gmap[None])(coords0)

#             # approximate flow magnitude using 1 update iteration
#             _, delta, weight = self.update(self.net[None], self.inp[None], corr)

#             # check motion magnitue / add new frame to video
#             if delta.norm(dim=-1).mean().item() > self.thresh:
#                 self.count = 0
#                 net, inp = self.__context_encoder(inputs)
#                 self.net, self.inp, self.fmap = net, inp, gmap
#                 self.video.append(tstamp, image, None, None, intrinsics / 8.0, gmap[0], net[0], inp[0])

#             else:
#                 self.count += 1
