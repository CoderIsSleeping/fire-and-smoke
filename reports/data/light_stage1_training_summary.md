# Training summary

- data: `/kaggle/input/datasets/coderissleeping/fire-and-smoke/fire_smoke_yolo/fire_smoke_yolo/data.yaml`
- backbone: `hf_hub:timm/vit_tiny_patch16_dinov3_qkvb.eupe_lvd1689m` (unfrozen blocks: 0)
- image size: 640, batch 8 x accum 1
- epochs run: 45
- best epoch: 44 (this is what weights/best.pt contains)
- best val mAP@0.5: 0.6220

## Validation detection metrics (epoch 44)

```
   class     #GT     AP50     AP75   AP50-95
--------------------------------------------
   smoke    1889   0.7277   0.3073    0.3666
    fire    2154   0.5163   0.1633    0.2215
--------------------------------------------
    mean           0.6220   0.2353    0.2941
```

## Alarm sweep on the validation split (epoch 44)

```
  conf  FP imgs      FPR  rec fire  rec smoke  rec any
------------------------------------------------------
  0.05     1610   1.0000    0.9989     1.0000   1.0000
  0.10     1610   1.0000    0.9989     1.0000   1.0000
  0.15     1607   0.9981    0.9989     1.0000   1.0000
  0.20     1475   0.9161    0.9989     1.0000   1.0000
  0.25     1074   0.6671    0.9944     0.9958   0.9956
  0.30      676   0.4199    0.9887     0.9928   0.9924
  0.35      329   0.2043    0.9831     0.9814   0.9831
  0.40      164   0.1019    0.9628     0.9562   0.9646
  0.45       73   0.0453    0.9324     0.9040   0.9351
  0.50       21   0.0130    0.8401     0.8217   0.8807
  0.55        9   0.0056    0.6475     0.6429   0.7422
  0.60        1   0.0006    0.3829     0.4442   0.5341
  0.65        0   0.0000    0.1745     0.2563   0.3095
  0.70        0   0.0000    0.0755     0.1212   0.1466
  0.75        0   0.0000    0.0135     0.0468   0.0490
  0.80        0   0.0000    0.0000     0.0096   0.0087
  0.85        0   0.0000    0.0000     0.0000   0.0000
  0.90        0   0.0000    0.0000     0.0000   0.0000
  0.95        0   0.0000    0.0000     0.0000   0.0000
```

Recommended operating threshold for FPR <= 0.010: conf >= 0.55 (FPR 0.0056, fire recall 0.6475, smoke recall 0.6429)

`FPR` is the fraction of verified-negative images that produced at least one
box at that confidence. Multiply by the frame rate to get false boxes per
second before temporal confirmation.
