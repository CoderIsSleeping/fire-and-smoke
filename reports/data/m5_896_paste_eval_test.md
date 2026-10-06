# Evaluation report - test split (896 px + mosaic + flame paste)

- run `dinov3_smallfire_896` (from stage 2; imgsz 896, mosaic 0.3, paste 0.3, 6 epochs, best val 0.7237)
- images: 4306 (2005 verified negatives), image size 896
- a second identical run scored 0.7166 mAP / 44 FP images at 0.90 (run-to-run variation)

```
   class     #GT     AP50     AP75   AP50-95
--------------------------------------------
   smoke    2311   0.8021   0.4033    0.4255
    fire    2878   0.6360   0.2235    0.2871
--------------------------------------------
    mean           0.7190   0.3134    0.3563
```

```
 class  size (px@640)    #GT    AP50  rec@0.5  rec@0.8  rec@0.9  med conf
 smoke      tiny 0-16     66  0.3459    0.379    0.333    0.212     0.831
 smoke    small 16-32    232  0.7320    0.832    0.724    0.603     0.938
 smoke   medium 32-96    490  0.7308    0.798    0.692    0.592     0.958
 smoke      large >96   1523  0.8526    0.907    0.842    0.775     0.979
  fire      tiny 0-16    534  0.3109    0.532    0.300    0.165     0.780
  fire    small 16-32    784  0.6322    0.815    0.611    0.434     0.889
  fire   medium 32-96   1137  0.7317    0.843    0.697    0.541     0.927
  fire      large >96    423  0.7505    0.837    0.747    0.612     0.951
```

```
  conf  FP imgs      FPR  rec fire  rec smoke  rec any
  0.70      185   0.0923    0.9596     0.9568   0.9687
  0.75      152   0.0758    0.9453     0.9433   0.9578
  0.80      116   0.0579    0.9220     0.9241   0.9431
  0.85       76   0.0379    0.8834     0.8904   0.9222
  0.88       61   0.0304    0.8511     0.8602   0.8983
  0.90       46   0.0229    0.8117     0.8246   0.8709
  0.93       34   0.0170    0.7291     0.7761   0.8266
  0.95       16   0.0080    0.5973     0.6881   0.7432
  0.97        3   0.0015    0.3794     0.5123   0.5654
```

Scene head @0.5: smoke precision 0.9438 recall 0.9447 fpr 0.0526; fire precision 0.8396 recall 0.9812 fpr 0.0655.
