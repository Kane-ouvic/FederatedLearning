# Copyright (c) 2021, NVIDIA CORPORATION.
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

import os.path

import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import *

'''import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize'''

from nvflare.apis.dxo import from_shareable, DataKind, DXO
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from simple_network import SimpleNetwork

imgpath = './images/train/images/'
maskpath = './images/train/label/'

class Cifar10Validator(Executor):
    
    def __init__(self, validate_task_name=AppConstants.TASK_VALIDATION, in_channels=1, out_channels=1):
        super(Cifar10Validator, self).__init__()

        self._validate_task_name = validate_task_name

        # Setup the model
        self.model = SimpleNetwork(in_channels, out_channels)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(self.device)

        # Preparing the dataset for testing.

        self.loss = nn.BCELoss()
        _, self.test_loader = split_Train_Val_Data(imgpath, maskpath, batch_size=4)

        '''transforms = Compose([
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.test_data = CIFAR10(root='~/data', train=False, transform=transforms)
        self.test_loader = DataLoader(self.test_data, batch_size=4, shuffle=False)'''

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        if task_name == self._validate_task_name:
            model_owner = "?"
            try:
                try:
                    dxo = from_shareable(shareable)
                except:
                    self.log_error(fl_ctx, "Error in extracting dxo from shareable.")
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Ensure data_kind is weights.
                if not dxo.data_kind == DataKind.WEIGHTS:
                    self.log_exception(fl_ctx, f"DXO is of type {dxo.data_kind} but expected type WEIGHTS.")
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Extract weights and ensure they are tensor.
                model_owner = shareable.get_header(AppConstants.MODEL_OWNER, "?")
                weights = {k: torch.as_tensor(v, device=self.device) for k, v in dxo.data.items()}

                # Get validation accuracy
                val_accuracy = self.do_validation(weights, abort_signal)
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)

                self.log_info(fl_ctx, f"Accuracy when validating {model_owner}'s model on"
                                      f" {fl_ctx.get_identity_name()}"f's data: {val_accuracy}')

                dxo = DXO(data_kind=DataKind.METRICS, data={'val_acc': val_accuracy})
                return dxo.to_shareable()
            except:
                self.log_exception(fl_ctx, f"Exception in validating model from {model_owner}")
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)
        else:
            return make_reply(ReturnCode.TASK_UNKNOWN)

    def do_validation(self, weights, abort_signal):
        self.model.load_state_dict(weights)

        self.model.eval()

        val_loss_C = 0
        val_iter = 0

        with torch.no_grad():
            for i, (x, label) in enumerate(self.test_loader):
                if abort_signal.triggered:
                    return 0

                x, label = x.to(self.device), label.to(self.device)
                val_outputs = self.model(x.float())

                val_loss = self.loss(val_outputs, label.float())
                val_loss_C += val_loss.item()
                val_iter += 1

        print('Validation loss_C: %.3f' % (val_loss_C / val_iter))

        return val_loss_C / val_iter
