# Evaluation report - test split

- weights: `/kaggle/working/fire_smoke/stage2/weights/best.pt`
- images: 4306 (2005 verified negatives)
- image size: 640

## Detection metrics

```
   class     #GT     AP50     AP75   AP50-95
--------------------------------------------
   smoke    2311   0.8146   0.4053    0.4309
    fire    2878   0.6377   0.2363    0.2980
--------------------------------------------
    mean           0.7261   0.3208    0.3644
```

## Alarm sweep

```
  conf  FP imgs      FPR  rec fire  rec smoke  rec any
------------------------------------------------------
  0.05     1085   0.5411    0.9928     0.9981   0.9974
  0.10      743   0.3706    0.9901     0.9966   0.9965
  0.15      582   0.2903    0.9839     0.9957   0.9961
  0.20      474   0.2364    0.9785     0.9928   0.9935
  0.25      381   0.1900    0.9776     0.9904   0.9909
  0.30      319   0.1591    0.9758     0.9885   0.9896
  0.35      271   0.1352    0.9740     0.9861   0.9878
  0.40      227   0.1132    0.9695     0.9841   0.9861
  0.45      192   0.0958    0.9668     0.9817   0.9848
  0.50      169   0.0843    0.9614     0.9779   0.9826
  0.55      142   0.0708    0.9578     0.9745   0.9809
  0.60      113   0.0564    0.9534     0.9702   0.9774
  0.65       82   0.0409    0.9498     0.9644   0.9744
  0.70       70   0.0349    0.9372     0.9553   0.9674
  0.75       50   0.0249    0.9193     0.9438   0.9591
  0.80       35   0.0175    0.8969     0.9241   0.9426
  0.85       26   0.0130    0.8574     0.8991   0.9235
  0.90       14   0.0070    0.7803     0.8385   0.8748
  0.95        7   0.0035    0.5982     0.7227   0.7662
```

## Recommended operating point

- confidence threshold: **0.90** (budget: FPR <= 0.010)
- false-alarm rate: 0.0070 (14/2005 negative images)
- recall at that threshold: fire 0.7803, smoke 0.8385

At 5 fps that is roughly 126 false boxes per hour **before** temporal confirmation. The N-of-M confirmation in `predict_video_dinov3.py` is what turns this into a usable alarm rate; requiring 6 linked hits inside 15 frames removes uncorrelated flicker.
