# Evaluation report - test split

- weights: `/kaggle/input/datasets/coderissleeping/dinov3-stage1/best.pt`
- images: 4306 (2005 verified negatives)
- image size: 640

## Detection metrics

```
   class     #GT     AP50     AP75   AP50-95
--------------------------------------------
   smoke    2311   0.8050   0.4172    0.4290
    fire    2878   0.6342   0.2333    0.2964
--------------------------------------------
    mean           0.7196   0.3252    0.3627
```

## Alarm sweep

```
  conf  FP imgs      FPR  rec fire  rec smoke  rec any
------------------------------------------------------
  0.05     1188   0.5925    0.9901     0.9986   0.9978
  0.10      846   0.4219    0.9892     0.9976   0.9970
  0.15      669   0.3337    0.9865     0.9952   0.9952
  0.20      534   0.2663    0.9830     0.9933   0.9935
  0.25      440   0.2195    0.9794     0.9909   0.9913
  0.30      384   0.1915    0.9749     0.9880   0.9887
  0.35      324   0.1616    0.9695     0.9865   0.9878
  0.40      265   0.1322    0.9641     0.9851   0.9870
  0.45      231   0.1152    0.9632     0.9817   0.9848
  0.50      196   0.0978    0.9596     0.9774   0.9813
  0.55      166   0.0828    0.9552     0.9736   0.9783
  0.60      133   0.0663    0.9507     0.9673   0.9744
  0.65      113   0.0564    0.9417     0.9592   0.9691
  0.70       86   0.0429    0.9336     0.9505   0.9631
  0.75       64   0.0319    0.9193     0.9346   0.9496
  0.80       40   0.0200    0.8969     0.9106   0.9322
  0.85       25   0.0125    0.8395     0.8775   0.9035
  0.90       14   0.0070    0.7587     0.8121   0.8479
  0.95        7   0.0035    0.5543     0.6843   0.7288
```

## Recommended operating point

- confidence threshold: **0.90** (budget: FPR <= 0.010)
- false-alarm rate: 0.0070 (14/2005 negative images)
- recall at that threshold: fire 0.7587, smoke 0.8121

At 5 fps that is roughly 126 false boxes per hour **before** temporal confirmation. The N-of-M confirmation in `predict_video_dinov3.py` is what turns this into a usable alarm rate; requiring 6 linked hits inside 15 frames removes uncorrelated flicker.
