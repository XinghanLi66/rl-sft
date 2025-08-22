set -x

CHECKPOINTS_DIR=/local1/lxh/save

export VLLM_ATTENTION_BACKEND=XFORMERS

model_name="Qwen2.5-Math-7B"
data_name="pi1_r128"
codebase_name="offline"
ts=$(TZ=America/Los_Angeles date +"%m%d-%H%M")
exp_name=${model_name}_${data_name}_${codebase_name}_${ts}
log_file="${model_name}_${data_name}_${codebase_name}.log"

python3 -m verl.trainer.main_ppo_sft \
 algorithm.adv_estimator=grpo_offline \
 data.train_files=data/train/one_shot_rlvr/${data_name}.parquet \
 data.val_files=data/test/math_minerva_aime25x8.parquet \
 data.train_batch_size=128 \
 data.val_batch_size=530 \
 data.max_prompt_length=1024 \
 data.max_response_length=3072 \
 reward_model.reward_manager='naive' \
 actor_rollout_ref.model.path=/homes/gws/lxh22/models/${model_name} \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.model.use_remove_padding=True \
 actor_rollout_ref.actor.ppo_mini_batch_size=128 \
 actor_rollout_ref.actor.use_dynamic_bsz=True \
 actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
 actor_rollout_ref.actor.use_kl_loss=True \
 actor_rollout_ref.actor.kl_loss_coef=0.001 \
 actor_rollout_ref.actor.kl_loss_type=low_var_kl \
 actor_rollout_ref.model.enable_gradient_checkpointing=True \
 actor_rollout_ref.actor.fsdp_config.param_offload=False \
 +actor_rollout_ref.actor.fsdp_config.grad_offload=False \
 actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
 actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
 actor_rollout_ref.rollout.name=vllm \
 actor_rollout_ref.rollout.temperature=0.6 \
 +actor_rollout_ref.rollout.val_temperature=0.0 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
 actor_rollout_ref.rollout.n=8 \
 +actor_rollout_ref.rollout.n_val=1 \
 actor_rollout_ref.ref.fsdp_config.param_offload=True \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.critic_warmup=0 \
 trainer.logger=['console','wandb'] \
 trainer.project_name='offline_grpo'\
 trainer.experiment_name=$exp_name \
 trainer.checkpoints_dir=$CHECKPOINTS_DIR \
 +trainer.val_before_train=True \
 trainer.n_gpus_per_node=8 \
 trainer.nnodes=1 \
 trainer.save_freq=20 \
 trainer.test_freq=20 \
 trainer.default_hdfs_dir=null \
 trainer.total_training_steps=120 \
 trainer.total_epochs=2000 2>&1 | tee $log_file