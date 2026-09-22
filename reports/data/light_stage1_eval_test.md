# Evaluation report - test split

- weights: `/kaggle/working/fire_smoke/light_stage1/weights/best.pt`
- images: 4306 (2005 verified negatives)
- image size: 640

## Detection metrics

```
   class     #GT     AP50     AP75   AP50-95
--------------------------------------------
   smoke    2311   0.7158   0.3006    0.3583
    fire    2878   0.5092   0.1404    0.2125
--------------------------------------------
    mean           0.6125   0.2205    0.2854
```

## Alarm sweep

```
  conf  FP imgs      FPR  rec fire  rec smoke  rec any
------------------------------------------------------
  0.05     2005   1.0000    0.9973     1.0000   1.0000
  0.10     2005   1.0000    0.9973     1.0000   1.0000
  0.15     2003   0.9990    0.9955     1.0000   1.0000
  0.20     1857   0.9262    0.9928     0.9990   0.9987
  0.25     1366   0.6813    0.9848     0.9976   0.9970
  0.30      845   0.4214    0.9776     0.9918   0.9909
  0.35      436   0.2175    0.9614     0.9817   0.9813
  0.40      197   0.0983    0.9444     0.9500   0.9609
  0.45       86   0.0429    0.9049     0.8962   0.9309
  0.50       23   0.0115    0.8126     0.7914   0.8648
  0.55       11   0.0055    0.6206     0.6266   0.7292
  0.60        7   0.0035    0.3973     0.4262   0.5289
  0.65        2   0.0010    0.1830     0.2331   0.2907
  0.70        1   0.0005    0.0789     0.1182   0.1460
  0.75        0   0.0000    0.0251     0.0481   0.0561
  0.80        0   0.0000    0.0000     0.0096   0.0087
  0.85        0   0.0000    0.0000     0.0000   0.0000
  0.90        0   0.0000    0.0000     0.0000   0.0000
  0.95        0   0.0000    0.0000     0.0000   0.0000
```

## Recommended operating point

- confidence threshold: **0.55** (budget: FPR <= 0.010)
- false-alarm rate: 0.0055 (11/2005 negative images)
- recall at that threshold: fire 0.6206, smoke 0.6266

At 5 fps that is roughly 99 false boxes per hour **before** temporal confirmation. The N-of-M confirmation in `predict_video_dinov3.py` is what turns this into a usable alarm rate; requiring 6 linked hits inside 15 frames removes uncorrelated flicker.
