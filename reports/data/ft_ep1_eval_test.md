# Evaluation report - test split (merged + D-Fire fine-tune, epoch 1, best.pt)

Transcribed from the Kaggle console output.

```
   class     #GT     AP50     AP75   AP50-95
--------------------------------------------
   smoke    2311   0.8058   0.4067    0.4314
    fire    2878   0.6197   0.2321    0.2898
--------------------------------------------
    mean           0.7128   0.3194    0.3606
```

```
  conf  FP imgs      FPR  rec fire  rec smoke  rec any
------------------------------------------------------
  0.85       21   0.0105    0.8502     0.8861   0.9100
  0.88       16   0.0080    0.8090     0.8669   0.8931
  0.90       15   0.0075    0.7578     0.8323   0.8661
```
