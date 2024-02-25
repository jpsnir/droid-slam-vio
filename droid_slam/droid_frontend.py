import torch
import lietorch
import numpy as np

from lietorch import SE3
from factor_graph import FactorGraph
from pathlib import Path
from datetime import datetime
import pickle


class FactorGraphContainer(torch.nn.Module):
    def __init__(self, values_dict):
        super().__init__()
        for key in values_dict:
            setattr(self, key, values_dict[key])


class DroidFrontend:
    def __init__(self, net, video, args):
        self.video = video
        self.update_op = net.update
        self.graph = FactorGraph(
            video, net.update, max_factors=48, upsample=args.upsample
        )

        # local optimization window
        self.t0 = 0
        self.t1 = 0

        # frontent variables
        self.is_initialized = False
        self.count = 0

        self.max_age = 25
        self.iters1 = 4
        self.iters2 = 2

        self.warmup = args.warmup
        self.beta = args.beta
        self.frontend_nms = args.frontend_nms
        self.keyframe_thresh = args.keyframe_thresh
        self.frontend_window = args.frontend_window
        self.frontend_thresh = args.frontend_thresh
        self.frontend_radius = args.frontend_radius
        
        tag_stereo = ""
        tag_upsample = ""
        tag_basename = "factorgraph_data_"
        
        if args.stereo:
            tag_stereo = "_stereo_"

        if args.upsample:
            tag_upsample = "_upsample_"
        
        self.save_fg_path = Path(args.reconstruction_path).joinpath(
                     tag_basename + tag_stereo + tag_upsample + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            )


        self.save_fg_path.mkdir(exist_ok=True,parents=True)
        print(f"Factor graph save format - {args.factor_graph_save_format}")
        
        self.save_fg_fmt = args.factor_graph_save_format

    def __update(self):
        """add edges, perform update"""

        self.count += 1
        self.t1 += 1
        print(f"DROID FRONT END update: count = {self.count}, t1= {self.t1}")
        print(f"Video Buffer counter : {self.video.counter.value} ")

        if self.graph.corr is not None:
            self.graph.rm_factors(self.graph.age > self.max_age, store=True)

        self.graph.add_proximity_factors(
            self.t1 - 5,
            max(self.t1 - self.frontend_window, 0),
            rad=self.frontend_radius,
            nms=self.frontend_nms,
            thresh=self.frontend_thresh,
            beta=self.beta,
            remove=True,
        )

        self.video.disps[self.t1 - 1] = torch.where(
            self.video.disps_sens[self.t1 - 1] > 0,
            self.video.disps_sens[self.t1 - 1],
            self.video.disps[self.t1 - 1],
        )

        for itr in range(self.iters1):
            self.graph.update(None, None, use_inactive=True)

        # set initial pose for next frame
        poses = SE3(self.video.poses)
        d = self.video.distance(
            [self.t1 - 3], [self.t1 - 2], beta=self.beta, bidirectional=True
        )

        if d.item() < self.keyframe_thresh:
            self.graph.rm_keyframe(self.t1 - 2)
            print(
                f"DROID FRONTEND - removing keyframe {self.t1 -2}, motion - {d.item()} less \
                    than threshold {self.keyframe_thresh}"
            )
            with self.video.get_lock():
                self.video.counter.value -= 1
                self.t1 -= 1

        else:
            for itr in range(self.iters2):
                print(f"DROID FRONTEND update iteration - phase2 - {itr}")
                self.graph.update(None, None, use_inactive=True)

        self.log_factor_graphs(format=self.save_fg_fmt)
        # set pose for next itration
        self.video.poses[self.t1] = self.video.poses[self.t1 - 1]
        self.video.disps[self.t1] = self.video.disps[self.t1 - 1].mean()

        # update visualization
        self.video.dirty[self.graph.ii.min() : self.t1] = True
    
    def log_factor_graphs_stereo(self):
        '''
        logs the factor graph of stereo data.
        '''
        

    def log_factor_graphs(self, format: str = "pkl"):
        """
        log the factor graph data in binary or ascii format
        """
        # t1 - 1 is the last frame
        print(
            f"Tstamp of last frame {self.t1 - 1} : {self.video.tstamp[self.t1 -1].item()}"
        )

        max_idx = self.graph.ii_total.max().item()
        if max_idx == self.t1:
            filename = f"fg_{self.count:04}_id_{int(self.video.tstamp[self.t1].item()):04}.{format}"
            t = self.t1  
        else:
            filename = f"fg_{self.count:04}_id_{int(self.video.tstamp[self.t1 - 1].item()):04}.{format}"
            t = self.t1 - 1
        

        factor_graph_data_dict = {
                "max_id": False,
                "id": self.video.tstamp[t].cpu(),
                "intrinsics": self.video.intrinsics[0].cpu(),
                "graph_data": {
                    "ii": self.graph.ii_total.cpu(),
                    "jj": self.graph.jj_total.cpu(),
                },
                "disps": self.video.disps[: t + 1].cpu(),
                "disps_sens": self.video.disps_sens[: t  + 1].cpu(),
                "disps_up": self.video.disps_up[: t + 1].cpu(),
                "c_map": self.graph.weight_total.cpu(),
                "predicted": self.graph.target_total.cpu(),
                "poses": self.video.poses[: t + 1].cpu(),
                "tstamp": self.video.tstamp[: t + 1].cpu(),
            }

        if format == "pt":
            data_container = torch.jit.script(
                FactorGraphContainer(factor_graph_data_dict)
            )
            data_container.save(self.save_fg_path.joinpath(filename))
        if format == "pkl":
            with open(self.save_fg_path.joinpath(filename), "wb") as f:
                pickle.dump(factor_graph_data_dict, f)
            f.close()

    def __initialize(self):
        """initialize the SLAM system"""

        self.t0 = 0
        self.t1 = self.video.counter.value

        self.graph.add_neighborhood_factors(self.t0, self.t1, r=3)

        for itr in range(8):
            self.graph.update(1, use_inactive=True)

        self.graph.add_proximity_factors(
            0, 0, rad=2, nms=2, thresh=self.frontend_thresh, remove=False
        )

        for itr in range(8):
            self.graph.update(1, use_inactive=True)

        # self.video.normalize()
        self.video.poses[self.t1] = self.video.poses[self.t1 - 1].clone()
        self.video.disps[self.t1] = self.video.disps[self.t1 - 4 : self.t1].mean()

        # initialization complete
        self.is_initialized = True
        self.last_pose = self.video.poses[self.t1 - 1].clone()
        self.last_disp = self.video.disps[self.t1 - 1].clone()
        self.last_time = self.video.tstamp[self.t1 - 1].clone()

        with self.video.get_lock():
            self.video.ready.value = 1
            self.video.dirty[: self.t1] = True

        self.graph.rm_factors(self.graph.ii < self.warmup - 4, store=True)

    def __call__(self):
        """main update"""

        # do initialization
        if not self.is_initialized and self.video.counter.value == self.warmup:
            self.__initialize()

        # do update
        elif self.is_initialized and self.t1 < self.video.counter.value:
            self.__update()
