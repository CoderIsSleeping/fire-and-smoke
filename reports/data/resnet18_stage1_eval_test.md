# Evaluation report - test split (ResNet-18 stage 1)

Transcribed from the Kaggle console output (the run's eval_test/ folder was not
included in the downloaded zip).

- weights: `/kaggle/working/fire_smoke/resnet18_stage1/weights/best.pt` (epoch 39, best val mAP50 0.6567)
- images: 4306 (2005 verified negatives), image size 640, device cuda

```
   class     #GT     AP50     AP75   AP50-95
--------------------------------------------
   smoke    2311   0.7177   0.2980    0.3517
    fire    2878   0.5778   0.1872    0.2574
--------------------------------------------
    mean           0.6477   0.2426    0.3046
```

Selected alarm-sweep rows:

```
  conf  FP imgs      FPR  rec fire  rec smoke  rec any
  0.85       23   0.0115    0.7462     0.7689   0.8110
  0.88       17   0.0085    0.6933     0.7314   0.7757
  0.90       13   0.0065    0.6143     0.6804   0.7288
```

Recommended operating threshold for FPR <= 0.010: conf >= 0.88
(FPR 0.0085, recall fire 0.6933, smoke 0.7314, any 0.7757).

Scene classifier head @ 0.5: smoke precision 0.8898 recall 0.8616 fpr 0.0998;
fire precision 0.9328 recall 0.8834 fpr 0.0223.
