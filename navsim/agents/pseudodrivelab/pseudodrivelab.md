# BEV Backbone : Advanced Latent TransFuser

기존 TransFuser 코드에서 수정 및 업데이트 된 부분은 'EDITTED: ' Comment를 통해 확인 하실 수 있습니다.

## Training
```
bash scripts/training/run_pseudodrivelab_training.sh
```

## Evaluation
```
bash scripts/evaluation/run_pseudodrivelab_pdm_score_evaluation.sh
```

## Submission
```
bash (Not yet updated)
```


## Updated 08 May 2025
- Extended camera input : 3 to 7
- Extended BEV Semantic map : Forward only (-90 deg ~ 90 deg) to Forward and Backward (-130 deg ~ 130 deg)
- Updated loss function : CE Loss for extened BEV Semantic map

