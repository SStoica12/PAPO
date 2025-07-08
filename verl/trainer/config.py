# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
PPO config
"""

import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Optional, Tuple, Dict, Any

from ..workers.config import WorkerConfig


def recursive_post_init(dataclass_obj):
    if hasattr(dataclass_obj, "post_init"):
        dataclass_obj.post_init()

    for attr in fields(dataclass_obj):
        if is_dataclass(getattr(dataclass_obj, attr.name)):
            recursive_post_init(getattr(dataclass_obj, attr.name))


@dataclass
class DataConfig:
    train_files: str = ""
    val_files: str = ""
    prompt_key: str = "prompt"
    answer_key: str = "answer"
    image_key: str = "images"
    aug_image_key: str = "images_aug"
    max_prompt_length: int = 512
    max_response_length: int = 512
    rollout_batch_size: int = 512
    val_batch_size: int = -1
    format_prompt: Optional[str] = None
    format_prompt_stage_2: Optional[str] = None
    override_chat_template: Optional[str] = None
    shuffle: bool = True
    seed: int = 1
    max_pixels: int = 4194304
    min_pixels: int = 262144
    filter_overlong_prompts: bool = True

    def post_init(self):
        if self.format_prompt is not None:
            if os.path.exists(self.format_prompt):
                self.format_prompt = os.path.abspath(self.format_prompt)
            else:
                self.format_prompt = None

        if self.format_prompt_stage_2 is not None:
            if os.path.exists(self.format_prompt_stage_2):
                self.format_prompt_stage_2 = os.path.abspath(self.format_prompt_stage_2)
            else:
                self.format_prompt_stage_2 = None

@dataclass
class AlgorithmConfig:
    gamma: float = 1.0
    lam: float = 1.0
    adv_estimator: str = "grpo"
    disable_kl: bool = False
    use_kl_loss: bool = False
    kl_penalty: str = "kl"
    kl_coef: float = 1e-3
    kl_type: str = "fixed"
    kl_horizon: float = 0.0
    kl_target: float = 0.0
    use_contrastive_kl: bool = False
    contrastive_type: str = "augmented"
    contrastive_kl_schedule: str = "fixed"
    contrastive_kl_schedule_args: Dict[str, Any] = field(default_factory=dict)
    contrastive_kl_penalty: str = "kl"
    contrastive_kl_coef: float = 1e-3
    contrastive_kl_apply_mode: str = "all"
    aug_config: Dict[str, Any] = field(default_factory=dict)
    incorrect_weighting: float = 0.1

    use_aug_entropy_loss: bool = False
    aug_entropy_loss_coef: float = 1e-2

    use_ori_entropy_loss: bool = False
    ori_entropy_loss_coef: float = 1e-2
    
    use_contrastive_kl_clipping: bool = False
    contrastive_kl_clipping: float = 0.2

    use_contrastive_kl_token_level_mask: bool = False
    contrastive_kl_token_level_mask_top_p: float = 0.2

    # custom for use sft loss
    use_sft_loss: bool = False
    sft_loss_coef: float = 1e-3

@dataclass
class TrainerConfig:
    total_epochs: int = 10
    max_steps: Optional[int] = None
    project_name: str = "papo"
    experiment_name: str = "papo_exp"
    logger: Tuple[str] = ("console", "wandb")
    nnodes: int = 1
    n_gpus_per_node: int = 8
    critic_warmup: int = 0
    val_freq: int = -1
    val_before_train: bool = True
    val_only: bool = False
    val_generations_to_log: int = 0
    save_freq: int = -1
    save_limit: int = -1
    save_checkpoint_path: Optional[str] = None
    load_checkpoint_path: Optional[str] = None
    save_best_checkpoint: bool = False

    def post_init(self):
        if self.save_checkpoint_path is None:
            self.save_checkpoint_path = os.path.join("checkpoints", self.project_name, self.experiment_name)

        self.save_checkpoint_path = os.path.abspath(self.save_checkpoint_path)
        if self.load_checkpoint_path is not None:
            self.load_checkpoint_path = os.path.abspath(self.load_checkpoint_path)


@dataclass
class PPOConfig:
    data: DataConfig = field(default_factory=DataConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    def post_init(self):
        self.worker.rollout.prompt_length = self.data.max_prompt_length
        self.worker.rollout.response_length = self.data.max_response_length
        self.worker.rollout.trust_remote_code = self.worker.actor.model.trust_remote_code
        self.worker.actor.disable_kl = self.algorithm.disable_kl
        self.worker.actor.use_kl_loss = self.algorithm.use_kl_loss
        self.worker.actor.kl_penalty = self.algorithm.kl_penalty
        self.worker.actor.kl_coef = self.algorithm.kl_coef
        self.worker.actor.use_contrastive_kl = self.algorithm.use_contrastive_kl
        self.worker.actor.contrastive_kl_penalty = self.algorithm.contrastive_kl_penalty
        self.worker.actor.contrastive_kl_coef = self.algorithm.contrastive_kl_coef
        self.worker.actor.contrastive_kl_apply_mode = self.algorithm.contrastive_kl_apply_mode
        self.worker.actor.use_aug_entropy_loss = self.algorithm.use_aug_entropy_loss
        self.worker.actor.aug_entropy_loss_coef = self.algorithm.aug_entropy_loss_coef
        self.worker.actor.use_ori_entropy_loss = self.algorithm.use_ori_entropy_loss
        self.worker.actor.ori_entropy_loss_coef = self.algorithm.ori_entropy_loss_coef
        self.worker.actor.use_contrastive_kl_clipping = self.algorithm.use_contrastive_kl_clipping
        self.worker.actor.contrastive_kl_clipping = self.algorithm.contrastive_kl_clipping
        self.worker.actor.use_contrastive_kl_token_level_mask = self.algorithm.use_contrastive_kl_token_level_mask
        self.worker.actor.contrastive_kl_token_level_mask_top_p = self.algorithm.contrastive_kl_token_level_mask_top_p

        # auto keys for sft loss
        self.worker.actor.use_sft_loss = self.algorithm.use_sft_loss
        self.worker.actor.sft_loss_coef = self.algorithm.sft_loss_coef


    def deep_post_init(self):
        recursive_post_init(self)

    def to_dict(self):
        return asdict(self)
