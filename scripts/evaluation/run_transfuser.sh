TRAIN_TEST_SPLIT=warmup_test_e2e
CHECKPOINT=exp/training_transfuser_agent/2025.04.09.08.36.13/lightning_logs/version_0/checkpoints/epoch99-step266000.ckpt

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=transfuser_agent \
worker=single_machine_thread_pool \
agent.checkpoint_path=$CHECKPOINT \
experiment_name=transfuser_agent_eval
