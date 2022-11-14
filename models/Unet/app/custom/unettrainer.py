# Unet Trainer V.0.1

import os.path

import torch
import torch.nn as nn
import torch.optim as optim
from loss import *
from dataloader import *

from nvflare.apis.dxo import from_shareable, DXO, DataKind, MetaKey
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.model import make_model_learnable, model_learnable_to_dxo
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.pt.pt_fed_utils import PTModelPersistenceFormatManager
from pt_constants import PTConstants
from unetwork import Unet

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
batch_size = 4

imgpath = './images/train/images'
maskpath = './images/train/label'


class UnetTrainer(Executor):

    def __init__(self, lr=1e-4, epochs=15, train_task_name=AppConstants.TASK_TRAIN,
                 submit_model_task_name=AppConstants.TASK_SUBMIT_MODEL, exclude_vars=None):

        super(UnetTrainer, self).__init__()

        self._lr = lr
        self._epochs = epochs
        self._train_model_task_name = submit_model_task_name
        self._exclude_vars = exclude_vars

        self.model = Unet(1, 1)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.criteron = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self._train_loader, self._ = split_Train_Val_Data(
            imgpath, maskpath, batch_size)
        self._n_iterations = len(self._train_loader)

        self._default_train_conf = {
            "train": {"model": type(self.model).__name__}}
        self.persistence_manager = PTModelPersistenceFormatManager(
            data=self.model.state_dict(), default_train_conf=self._default_train_conf)

    def local_train(self, fl_ctx, weights, abort_signal):

        self.model.load_state_dict(state_dict=weights)

        self.model.train()
        for epoch in range(self._epochs):
            iter = 0
            train_loss_C = 0.0
            print('epcoh: ' + str(epoch + 1) + ' / ' + str(epochs))
            print('***************************',
                  len(self._train_loader), '***************************')
            for i, (x, labels) in enumerate(self._train_loader):
                if abort_signal.triggered:
                    return
                x, labels = x.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(x.float())

                loss = self.criteron(outputs, label.float())
                loss.backward()
                self.optimizer.step()

                train_loss_C += loss
                iter += 1

            print('Training epoch: %d / loss_C: %.3f' %
                  (epoch + 1, train_loss_C / iter))

    # Maybe need modify ??
    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        try:
            if task_name == self._train_task_name:
                # Get model weights
                try:
                    dxo = from_shareable(shareable)
                except:
                    self.log_error(
                        fl_ctx, "Unable to extract dxo from shareable.")
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Ensure data kind is weights.
                if not dxo.data_kind == DataKind.WEIGHTS:
                    self.log_error(
                        fl_ctx, f"data_kind expected WEIGHTS but got {dxo.data_kind} instead.")
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Convert weights to tensor. Run training
                torch_weights = {k: torch.as_tensor(
                    v) for k, v in dxo.data.items()}
                self.local_train(fl_ctx, torch_weights, abort_signal)

                # Check the abort_signal after training.
                # local_train returns early if abort_signal is triggered.
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)

                # Save the local model after training.
                self.save_local_model(fl_ctx)

                # Get the new state dict and send as weights
                new_weights = self.model.state_dict()
                new_weights = {k: v.cpu().numpy()
                               for k, v in new_weights.items()}

                outgoing_dxo = DXO(data_kind=DataKind.WEIGHTS, data=new_weights,
                                   meta={MetaKey.NUM_STEPS_CURRENT_ROUND: self._n_iterations})
                return outgoing_dxo.to_shareable()
            elif task_name == self._submit_model_task_name:
                # Load local model
                ml = self.load_local_model(fl_ctx)

                # Get the model parameters and create dxo from it
                dxo = model_learnable_to_dxo(ml)
                return dxo.to_shareable()
            else:
                return make_reply(ReturnCode.TASK_UNKNOWN)
        except:
            self.log_exception(fl_ctx, f"Exception in simple trainer.")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def save_local_model(self, fl_ctx: FLContext):
        run_dir = fl_ctx.get_engine().get_workspace().get_run_dir(
            fl_ctx.get_prop(ReservedKey.RUN_NUM))
        models_dir = os.path.join(run_dir, PTConstants.PTModelsDir)
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        model_path = os.path.join(models_dir, PTConstants.PTLocalModelName)

        ml = make_model_learnable(self.model.state_dict(), {})
        self.persistence_manager.update(ml)
        torch.save(self.persistence_manager.to_persistence_dict(), model_path)

    def load_local_model(self, fl_ctx: FLContext):
        run_dir = fl_ctx.get_engine().get_workspace().get_run_dir(
            fl_ctx.get_prop(ReservedKey.RUN_NUM))
        models_dir = os.path.join(run_dir, PTConstants.PTModelsDir)
        if not os.path.exists(models_dir):
            return None
        model_path = os.path.join(models_dir, PTConstants.PTLocalModelName)

        self.persistence_manager = PTModelPersistenceFormatManager(data=torch.load(model_path),
                                                                   default_train_conf=self._default_train_conf)
        ml = self.persistence_manager.to_model_learnable(
            exclude_vars=self._exclude_vars)
        return ml
