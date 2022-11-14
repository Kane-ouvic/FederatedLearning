# Unet Trainer V.0.1

import os.path

import torch
import torch.nn as nn
import torch.optim as optim
from loss import *
from dataloader import *

from nvflare.apis.dxo import from_shareable, DataKind, DXO
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from unetwork import Unet

os.environ['CUDA_VISIBLE_DEVICES']='0'
batch_size = 4

imgpath = './images/train/images/'
maskpath = './images/train/label/'

class UnetValidator(Executor):

    def __init__(self, validate_task_name=AppConstants.TASK_VALIDATION):
        super(UnetValidator, self).__init__()

        self._validate_task_name = validate_task_name

        self.model = Unet(1, 1)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(self.device)
        self.criteron = nn.BCELoss()

        self._, self.val_dataloader = split_Train_Val_Data(imgpath, maskpath, batch_size)
        self._n_iterations = len(self.val_dataloader)
    
    def do_validation(self, weights, abort_signal):
        self.model.load_state_dict(weights)

        self.model.eval()
        
        val_iter = 0
        val_loss_C = 0.0

        with torch.no_grad():
            for i, (x, label) in enumerate(self.val_dataloader):   
                with torch.no_grad(): 
                    x, label = x.to(device), label.to(device)
                    val_outputs = C(x.float())
                    
                    val_loss = criteron(val_outputs, label.float())
                    val_loss_C += val_loss.item()
                    val_iter += 1
        
        print('Validation epoch: %d / loss_C: %.3f' % (epoch + 1, val_loss_C / val_iter))

        return val_loss_C / val_iter

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

    