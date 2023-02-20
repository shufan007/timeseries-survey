# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import torch
import glob
import shutil
from datetime import datetime


def _create_if_not_exist(path):
    basedir = os.path.dirname(path)
    if not os.path.exists(basedir):
        os.makedirs(basedir)


def save_model(output_path,
               model,
               steps,
               opt,
               lr_scheduler,
               max_ckpt=2
               ):
    output_dir = os.path.join(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    state = {'model': model.state_dict(), 'optim': opt.state_dict(), 'lr_scheduler': lr_scheduler.state_dict(),'epoch':steps}
    torch.save(state,
               os.path.join(output_dir, "best_model.pth" ) )
    torch.save(model,
               os.path.join(output_dir, "best_model.pkl" ) )    
    print("save model %s" % output_dir)

    ckpt_paths = glob.glob(os.path.join(output_path, "model_*"))
    if len(ckpt_paths) > max_ckpt:
        def version(x):
            x = int(x.split("_")[-1])
            return x

        rm_ckpt_paths = sorted(
            ckpt_paths, key=version, reverse=True)[max_ckpt:]
        for ckpt_dir in rm_ckpt_paths:
            if os.path.exists(ckpt_dir):
                shutil.rmtree(ckpt_dir)


def load_model(output_path, model=None, opt=None, lr_scheduler=None):
    def version(x):
        x = int(x.split("_")[-1])
        return x

    ckpt_paths = glob.glob(os.path.join(output_path))

    if model==None:   
        #model = torch.load(os.path.join(output_path, "best_model.pkl"), map_location=torch.device('cpu'))
        model = torch.load(os.path.join(output_path, "best_model.pkl"))
        return model
    
    steps = 0
    if len(ckpt_paths) > 0:
        # output_dir = sorted(ckpt_paths, key=version, reverse=True)[0]
            
        model_state_dict = torch.load(
            os.path.join(output_path, "best_model.pth"))
        model.load_state_dict({k.replace('module.',''):v for k,v in model_state_dict['model'].items()})
        if opt:
            opt.load_state_dict(model_state_dict['optim'])
        if lr_scheduler:
            lr_scheduler.load_state_dict(model_state_dict['lr_scheduler'])
        print("load model from  %s" % output_path)

    return steps


if __name__ == '__main__':
    model_state_dict=torch.load(os.path.join('output/baseline/model_6', "best_model_6.pth"))
    from wpf_model import GruAttModel
    import argparse, yaml
    from addict import Dict
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument("--conf", type=str, default="./config.yaml")
    args = parser.parse_args()
    config = Dict(yaml.load(open(args.conf), Loader=yaml.FullLoader))
    model = GruAttModel(config)

    print(model_state_dict['optim'])
    model.load_state_dict({k.replace('module.',''):v for k,v in model_state_dict['model'].items()})
    model.eval()