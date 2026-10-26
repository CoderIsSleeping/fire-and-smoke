# Evaluation report - test split - Model 7 (hard negatives + small-fire oversampling)
- run `dinov3_896_negs_smallfire`: init from Model 6, imgsz 896, mosaic 0.3, unfreeze 2, lr 2e-5, 6 epochs,
  train = D-Fire + 1,008 DFS "other" + 1,500 hard-hat images (all negatives), small fires (<=32 px) drawn x3
- best val mAP50 0.7401 at epoch 4; images: 4306 (2005 verified negatives), image size 896
```
   class     #GT     AP50     AP75   AP50-95
--------------------------------------------
   smoke    2311   0.8031   0.4080    0.4254
    fire    2878   0.6570   0.2180    0.2974
--------------------------------------------
    mean           0.7301   0.3130    0.3614
```
```
 class  size (px@640)    #GT    AP50  rec@0.5  rec@0.8  rec@0.9  med conf
 smoke      tiny 0-16     66  0.3164    0.379    0.273    0.136     0.777
 smoke    small 16-32    232  0.7320    0.806    0.716    0.565     0.931
 smoke   medium 32-96    490  0.7235    0.800    0.702    0.598     0.951
 smoke      large >96   1523  0.8575    0.907    0.845    0.779     0.980
  fire      tiny 0-16    534  0.3765    0.575    0.358    0.182     0.816
  fire    small 16-32    784  0.6752    0.832    0.624    0.421     0.888
  fire   medium 32-96   1137  0.7373    0.850    0.705    0.544     0.928
  fire      large >96    423  0.7356    0.832    0.730    0.596     0.953
```
```
  conf  FP imgs      FPR  rec fire  rec smoke  rec any
  0.80       57   0.0284    0.9121     0.9202   0.9361
  0.85       42   0.0209    0.8735     0.8880   0.9153
  0.88       35   0.0175    0.8475     0.8558   0.8927
  0.90       26   0.0130    0.7946     0.8198   0.8666
  0.93       18   0.0090    0.7247     0.7684   0.8201
  0.95        7   0.0035    0.6081     0.6877   0.7458
```
Scene head @0.5: smoke precision 0.9503 recall 0.9380 fpr 0.0458; fire precision 0.9247 recall 0.9587 fpr 0.0273.
