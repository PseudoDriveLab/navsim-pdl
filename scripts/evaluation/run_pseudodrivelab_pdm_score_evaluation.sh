TRAIN_TEST_SPLIT=warmup_two_stage
CHECKPOINT=$NAVSIM_DEVKIT_ROOT/exp/training_pseudodrivelab_agent/***.ckpt

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
train_test_split=$TRAIN_TEST_SPLIT \
experiment_name=pseudodrivelab_agent \
worker=single_machine_thread_pool \
agent=pseudodrivelab_agent \
agent.checkpoint_path=$CHECKPOINT \
agent.config.latent=true



# If you want to run one stage only simulation, plesae uncomment and keep only the following lines:
# TRAIN_TEST_SPLIT=navtest
# CHECKPOINT=/path/to/pseudolab.ckpt

# python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
# train_test_split=$TRAIN_TEST_SPLIT \
# agent=pseudolab_agent \
# worker=single_machine_thread_pool \
# agent.checkpoint_path=$CHECKPOINT \
# experiment_name=pseudolab_agent \
# traffic_agents_policy=non_reactive \
