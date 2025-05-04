TRAIN_TEST_SPLIT=navtrain

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
agent=transfuser_agent \
experiment_name=training_transfuser_agent \
train_test_split=$TRAIN_TEST_SPLIT \
dataloader.params.batch_size=2 \
dataloader.params.num_workers=8 \
trainer.params.strategy=ddp \
trainer.params.accumulate_grad_batches=2 \
