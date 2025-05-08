export HYDRA_FULL_ERROR=1

TRAIN_TEST_SPLIT=navtrain

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
agent=pseudodrivelab_agent \
experiment_name=training_pseudodrivelab_agent \
train_test_split=$TRAIN_TEST_SPLIT \
use_cache_without_dataset=true \            
force_cache_computation=false \             
trainer.params.max_epochs=20 \
dataloader.params.batch_size=32 \

# use_cache_without_dataset
# false: caching        true: not caching
# force_cache_computation
# true: caching        false: not caching

