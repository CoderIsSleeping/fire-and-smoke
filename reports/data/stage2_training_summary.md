# Training summary

- data: `/kaggle/input/datasets/coderissleeping/fire-and-smoke/fire_smoke_yolo/fire_smoke_yolo/data.yaml`
- backbone: `vit_small_patch16_dinov3.lvd1689m` (unfrozen blocks: 2)
- image size: 640, batch 8 x accum 1
- epochs run: 15
- best epoch: 15 (this is what weights/best.pt contains)
- best val mAP@0.5: 0.7406

## Validation detection metrics (epoch 15)

```
   class     #GT     AP50     AP75   AP50-95
--------------------------------------------
   smoke    1889   0.8255   0.4406    0.4470
    fire    2154   0.6556   0.2426    0.3030
--------------------------------------------
    mean           0.7406   0.3416    0.3750
```

## Alarm sweep on the validation split (epoch 15)

```
  conf  FP imgs      FPR  rec fire  rec smoke  rec any
------------------------------------------------------
  0.05      781   0.4851    0.9977     0.9976   0.9978
  0.10      548   0.3404    0.9955     0.9964   0.9962
  0.15      414   0.2571    0.9932     0.9952   0.9956
  0.20      323   0.2006    0.9910     0.9940   0.9946
  0.25      270   0.1677    0.9887     0.9916   0.9918
  0.30      229   0.1422    0.9865     0.9856   0.9880
  0.35      194   0.1205    0.9854     0.9814   0.9853
  0.40      173   0.1075    0.9854     0.9790   0.9842
  0.45      145   0.0901    0.9820     0.9760   0.9815
  0.50      121   0.0752    0.9764     0.9742   0.9798
  0.55       99   0.0615    0.9673     0.9724   0.9771
  0.60       76   0.0472    0.9617     0.9706   0.9749
  0.65       58   0.0360    0.9505     0.9640   0.9728
  0.70       45   0.0280    0.9403     0.9544   0.9651
  0.75       40   0.0248    0.9279     0.9484   0.9602
  0.80       25   0.0155    0.9065     0.9298   0.9499
  0.85       19   0.0118    0.8761     0.8974   0.9264
  0.90       10   0.0062    0.7827     0.8529   0.8888
  0.95        2   0.0012    0.5709     0.7431   0.7798
```

Recommended operating threshold for FPR <= 0.010: conf >= 0.90 (FPR 0.0062, fire recall 0.7827, smoke recall 0.8529)

`FPR` is the fraction of verified-negative images that produced at least one
box at that confidence. Multiply by the frame rate to get false boxes per
second before temporal confirmation.
