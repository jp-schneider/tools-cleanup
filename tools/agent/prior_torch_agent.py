from collections import OrderedDict
from datetime import datetime
import inspect
import io
import logging
import os.path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, Union

import numpy as np
from tools.agent.torch_agent_checkpoint import TorchAgentCheckpoint
from tools.dataset.prior_dataset import PriorDataset, PriorManager, create_prior_collate_fn
from tools.dataset.torch_datasource import TorchDataSource
from tools.error.stop_training import StopTraining
from tools.event.agent_save_event_args import SaveStage
from tools.event.torch_agent_save_event_args import TorchAgentSaveEventArgs
from tools.event.torch_training_started_event_args import TorchTrainingStartedEventArgs
from tools.event.training_finished_event_args import TrainingFinishedEventArgs
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.autonotebook import tqdm

from tools.agent.agent import Agent
from tools.agent.util import (DataTracker, LearningMode, LearningScope,
                                MetricMode, Tracker)
from tools.metric.metric_scope import MetricScope
from tools.metric.metric_summary import MetricSummary
from tools.event import Event, TorchModelStepEventArgs
from tools.event.torch_optimizer_created_event_args import \
    TorchOptimizerCreatedEventArgs
from tools.measures.tracker_loss import TrackerLoss
from tools.model.pretrainable_module import PretrainableModule
from tools.util.format import strfdelta
from tools.util.timer import Timer
from tools.util.torch import TensorUtil
from tools.agent.torch_agent import TorchAgent

class PriorTorchAgent(TorchAgent):

    def __init__(self,
                 use_prior_collate_fn: bool = False,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.use_prior_collate_fn = use_prior_collate_fn

    def _perform_epoch(self,
                       epoch: int,
                       num_epochs: int,
                       model: Optional[torch.nn.Module],
                       optimizer: Optional[torch.optim.Optimizer],
                       criterion: torch.nn.Module,
                       training_dataset: Dataset,
                       validation_dataset: Dataset,
                       training_batch_size: int,
                       training_data_shuffle: bool,
                       validation_batch_size: int,
                       validation_data_shuffle: bool,
                       shuffle_seed: int,
                       tracker: Tracker,
                       remaining_iterations: int,
                       use_progress_bar: bool,
                       dataset_config: Dict[str, Any],
                       index_in_item: bool = False,
                       batch_progress_bar: Optional[tqdm] = None,
                       epoch_progress_bar: Optional[tqdm] = None,
                       keep_device: bool = False,
                       **kwargs) -> Tuple[torch.nn.Module, torch.optim.Optimizer, Optional[tqdm]]:
        is_training_finished = False
        try:

            epoch_output_event_args = None
            with Timer() as epoch_timer:
                if epoch_progress_bar:
                    epoch_progress_bar.set_description(
                        'Epoch {}/{}'.format(epoch + 1, num_epochs), refresh=True)

                # If model was removed because of saving
                if model is None:
                    model = self._get_prepared_model()
                    # init optimizer again
                    optimizer = self._get_optimizer(model)

                def worker_init_fn(x):
                    np.random.seed(shuffle_seed)
                    torch.random.manual_seed(shuffle_seed)

                dataloaders = OrderedDict()

                collate_fn = None
                if self.use_prior_collate_fn:
                    use_prior = isinstance(training_dataset, PriorDataset) and training_dataset.has_prior
                    collate_fn = create_prior_collate_fn(has_prior=use_prior)

                # Setting up the data loaders
                if len(training_dataset) > 0:
                    dataloader_train = DataLoader(training_dataset, batch_size=training_batch_size,
                                                  shuffle=training_data_shuffle,
                                                  worker_init_fn=worker_init_fn, collate_fn=collate_fn)
                    dataloaders[LearningMode.TRAINING] = dataloader_train

                if len(validation_dataset) > 0:
                    if self.should_validate_on_epoch is not None and self.should_validate_on_epoch(epoch):
                        dataloader_val = DataLoader(validation_dataset, batch_size=validation_batch_size,
                                                    shuffle=validation_data_shuffle,
                                                    worker_init_fn=worker_init_fn, collate_fn=collate_fn)
                        dataloaders[LearningMode.VALIDATION] = dataloader_val

                # Each epoch has a training and validation phase
                for phase, loader in dataloaders.items():
                    if phase == LearningMode.TRAINING:
                        model.train()  # Set model to training mode
                    else:
                        model.eval()  # Set model to evaluate mode

                    # Enlarge batch metric tracker
                    tracker.extend_batch_metrics(
                        len(loader), LearningMode.to_metric_mode(phase))

                    if use_progress_bar and self.batch_progress_bar:
                        desc = f'Model-Epoch: {tracker.get_epoch(phase == LearningMode.TRAINING)} / {phase} - Batches'
                        if batch_progress_bar is None:
                            batch_progress_bar = tqdm(
                                total=len(loader),
                                desc=desc,
                                leave=True)
                        else:
                            batch_progress_bar.reset(total=len(loader))
                            batch_progress_bar.set_description(
                                desc
                            )

                    epoch_data_tracker = DataTracker(
                        track_input=self.track_input,
                        track_label=self.track_label,
                        track_loss=self.track_loss,
                        track_prediction=self.track_prediction,
                        track_indices=index_in_item and self.track_indices
                    )
                    try:
                        for idx, item in enumerate(loader):
                            self._perform_step(
                                index=idx,
                                item=item,
                                phase=phase,
                                optimizer=optimizer,
                                model=model,
                                criterion=criterion,
                                tracker=tracker,
                                epoch_data_tracker=epoch_data_tracker,
                                dataset_config=dataset_config,
                                remaining_iterations=remaining_iterations,
                                batch_progress_bar=batch_progress_bar,
                                epoch_progress_bar=epoch_progress_bar,
                                index_in_item=index_in_item,
                                **kwargs
                            )
                    except StopTraining as err:
                        is_training_finished = True
                        raise err
                    finally:
                        tracker.epoch(
                                in_training=(phase == LearningMode.TRAINING))

                        epoch_loss = epoch_data_tracker.running_loss
                        # Track epoch main metric
                        tracker.epoch_metric(Tracker.get_metric_name(
                            self.loss), value=epoch_loss,
                            in_training=(phase == LearningMode.TRAINING),
                            is_primary=True)

                        epoch_output_event_args = TorchModelStepEventArgs(
                            model=model,
                            model_args=self.model_args,
                            optimizer=optimizer,
                            mode=phase,
                            label=epoch_data_tracker.combined_labels(),
                            output=epoch_data_tracker.combined_predictions(),
                            input=epoch_data_tracker.combined_inputs(),
                            indices=epoch_data_tracker.combined_indices(),
                            loss=epoch_loss,
                            loss_name=self.get_loss_name(),
                            tracker=tracker,
                            remaining_iterations=remaining_iterations if not is_training_finished else 0,
                            dataset_config=dataset_config,
                            scope=LearningScope.EPOCH
                        )
                        self.epoch_processed.notify(
                            epoch_output_event_args)
                        
                        if use_progress_bar:
                            epoch_progress_bar.set_postfix(
                                loss=epoch_loss, refresh=False)
                        
        except StopTraining as err:
            raise
        finally:
            # Ending epoch timing
            logging.info(
                f'Epoch {epoch + 1} / {num_epochs} time {strfdelta(epoch_timer.duration, "%H:%M:%S")}')

            # Compare validation result with best and save model if current is better.
            if (epoch_output_event_args is not None
                and (epoch_output_event_args.mode == LearningMode.VALIDATION or LearningMode.VALIDATION not in dataloaders.keys())
                    and tracker.is_current_state_best_model()):
                best = tracker.get_recent_performance()

                if not keep_device:
                    model.to('cpu')

                self.save(execution_context=kwargs,
                                 keep_device=keep_device,
                                 stage=SaveStage.BEST,
                                 )
                logging.info(f'Accuracy: {best.value}')

                if not keep_device and self.device != "cpu":
                    # Remove reference model optimizer, because its invalid now
                    model = None
                    self._free_optimizer()
                    optimizer = None
        return model, optimizer, batch_progress_bar

    def _decompose_training_item(self, item: Any) -> Tuple[Any, Any, torch.Tensor, Optional[Any]]:
        """Unpacks the item from the training dataset.

        Parameters
        ----------
        item : Any
            The item from the training dataset.

        Returns
        -------
        Tuple[Any, Any, torch.Tensor, Optional[Any]]
            The unpacked item.
            1. The inputs
            2. The labels
            3. The indices
            4. The prior state
        """
        return type(self).decompose_training_item(item, training_dataset=self.training_dataset, use_prior_collate_fn=self.use_prior_collate_fn)

    @classmethod
    def decompose_training_item(cls, item: Any, 
                                training_dataset: TorchDataSource,
                                use_prior_collate_fn: bool = False
                                ) -> Tuple[Any, Any, torch.Tensor, Optional[Any]]:
        """Unpacks the item from the training dataset.

        Parameters
        ----------
        item : Any
            The item from the training dataset.
        training_dataset : TorchDataSource
            The training dataset.

        Returns
        -------
        Tuple[Any, Any, torch.Tensor, Optional[Any]]
            The unpacked item.
            1. The inputs
            2. The labels
            3. The indices
            4. The prior state
        """
        prior_state = None
        # Extraxct prior of attached
        if isinstance(training_dataset, PriorDataset) and training_dataset.has_prior:
            # If its a prior dataset the return will be a tuple (prior, usual_return) of the dataset
            prior_state = item[0]
            key, state = prior_state
            key = key.item()  # Converting it to int again

            if not use_prior_collate_fn:
                # In older versions we using single batch for our priors
                # Removing batch dimension for each entry in state
                new_state = TensorUtil.apply_deep(state, lambda x: x[0] if isinstance(
                    x, torch.Tensor) and x.shape[0] == 1 else x)
            else:
                # In newer versions we using the prior collate function which is already a list of states.
                new_state = state
            
            prior_state = (key, new_state)
            
            item = item[1]

        inputs, labels = item[0], item[1]
        indices = item[2] if training_dataset.returns_index else None

        return inputs, labels, indices, prior_state

    def _perform_step(self,
                      item: Tuple[torch.Tensor, ...],
                      phase: LearningMode,
                      optimizer: torch.optim.Optimizer,
                      model: torch.nn.Module,
                      criterion: torch.nn.Module,
                      tracker: Tracker,
                      epoch_data_tracker: DataTracker,
                      dataset_config: Dict[str, Any],
                      remaining_iterations: int,
                      batch_progress_bar: Optional[tqdm] = None,
                      epoch_progress_bar: Optional[tqdm] = None,
                      index_in_item: bool = False, **kwargs):
        # Getting the inputs and labels unpacked from what training dataset returns
        inputs, labels, indices, prior_state = self._decompose_training_item(
            item)

        device_inputs: torch.Tensor = TensorUtil.to(inputs, device=self.device)
        device_labels: torch.Tensor = TensorUtil.to(labels, device=self.device)

        higher_opt = kwargs.get("higher_optimization", False)

        if not higher_opt:
            # zero the parameter gradients
            optimizer.zero_grad()

        # Optional statistics of batch
        stats = None

        # forward
        # track history if only in train
        with (torch.set_grad_enabled(phase == LearningMode.TRAINING),
              PriorManager(model,
                           prior_state,
                           getattr(self.training_dataset, "__prior_cache__", None), 
                            model_device=self.device, 
                            training=phase == LearningMode.TRAINING) as prior_manager):

            model_kwargs = {}
            if self.model_gets_targets:
                model_kwargs['targets'] = device_labels

            if isinstance(device_inputs, list):
                # Unpacking as list as dataloader wraps multiple args within a list
                device_outputs: torch.Tensor = model(*device_inputs, **model_kwargs)
            else:
                device_outputs: torch.Tensor = model(
                    device_inputs, **model_kwargs)

            # If the loss function accepts additional arguments, pass them
            if self.forward_additional_loss_args:
                loss: torch.Tensor = criterion(
                    device_outputs, device_labels, _input=device_inputs)
            else:
                loss: torch.Tensor = criterion(device_outputs, device_labels)

            if torch.isnan(loss):
                logging.warning("Loss is NaN!")
                breakpoint()
                raise StopTraining()
            # backward + optimize only if in training phase
            if phase == LearningMode.TRAINING:
                if not higher_opt:
                    loss.backward()
                    optimizer.step()
                else:
                    optimizer.step(loss)

        # Increase step
        tracker.step(
            in_training=(phase == LearningMode.TRAINING))

        # Getting the loss
        loss_detached = loss.item()
        # detaching tensors
        outputs_detached = TensorUtil.apply_deep(
            device_outputs, fnc=lambda x: x.detach().cpu())

        # Tracking data
        epoch_data_tracker.push(
            loss=loss_detached,
            prediction=outputs_detached,
            label=labels,
            input=inputs,
            indices=indices
        )
        # Track main loss
        tracker.step_metric(Tracker.get_metric_name(
            self.loss),
            value=loss_detached,
            in_training=(phase == LearningMode.TRAINING),
            is_primary=True)

        try:
            # Batch notify
            self.batch_processed.notify(TorchModelStepEventArgs(
                model=model,
                model_args=self.model_args,
                optimizer=optimizer,
                mode=phase,
                label=labels,
                output=outputs_detached,
                indices=indices,
                input=inputs,
                loss=loss_detached,
                loss_name=self.get_loss_name(),
                tracker=tracker,
                remaining_iterations=remaining_iterations,
                dataset_config=dataset_config,
                scope=LearningScope.BATCH
            ))
        finally:
            if batch_progress_bar is not None:
                batch_progress_bar.update()
                batch_progress_bar.set_postfix(
                    loss_avg=epoch_data_tracker.running_loss,
                    loss=loss_detached,
                    refresh=False)

            elif epoch_progress_bar is not None:
                epoch_progress_bar.set_postfix(
                    loss_avg=epoch_data_tracker.running_loss,
                    loss=loss_detached,
                    refresh=False)