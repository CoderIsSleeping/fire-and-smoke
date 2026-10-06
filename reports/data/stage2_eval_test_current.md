# Evaluation report - test split (stage 2, 640 px, current test-time settings)

Re-run on 2026-10-01 with the test-time settings in use since 2026-09-09
(300 RPN proposals, at most 20 detections per image). `stage2_eval_test.md`
is the original run with the earlier settings (1000 proposals, 50 detections),
which scored 0.7261; every model evaluated after 2026-09-09 uses these.

- weights: `dinov3_stage2_best.pt` (epoch 15, best val mAP50 0.7406)
- images: 4306 (2005 verified negatives), image size 640

```
   class     #GT     AP50     AP75   AP50-95
--------------------------------------------
   smoke    2311   0.8137   0.4068    0.4314
    fire    2878   0.6278   0.2354    0.2944
--------------------------------------------
    mean           0.7207   0.3211    0.3629
```

```
 class  size (px@640)    #GT    AP50  rec@0.5  rec@0.8  rec@0.9  med conf
 smoke      tiny 0-16     66  0.3130    0.333    0.227    0.136     0.637
 smoke    small 16-32    232  0.7260    0.819    0.681    0.478     0.916
 smoke   medium 32-96    490  0.7525    0.802    0.696    0.618     0.959
 smoke      large >96   1523  0.8670    0.908    0.850    0.794     0.986
  fire      tiny 0-16    534  0.2511    0.371    0.213    0.097     0.740
  fire    small 16-32    784  0.6312    0.735    0.543    0.398     0.878
  fire   medium 32-96   1137  0.7369    0.841    0.688    0.545     0.930
  fire      large >96    423  0.7689    0.868    0.775    0.664     0.957
```

```
  conf  FP imgs      FPR  rec fire  rec smoke  rec any
  0.50      163   0.0813    0.9614     0.9769   0.9817
  0.60      108   0.0539    0.9525     0.9697   0.9770
  0.65       79   0.0394    0.9489     0.9640   0.9739
  0.70       68   0.0339    0.9363     0.9553   0.9670
  0.75       48   0.0239    0.9184     0.9428   0.9587
  0.78       39   0.0195    0.9076     0.9356   0.9526
  0.80       33   0.0165    0.8951     0.9217   0.9418
  0.82       27   0.0135    0.8771     0.9125   0.9344
  0.85       24   0.0120    0.8556     0.8976   0.9231
  0.88       17   0.0085    0.8287     0.8712   0.9009
  0.90       14   0.0070    0.7767     0.8366   0.8744
  0.93       11   0.0055    0.7022     0.7948   0.8344
  0.95        7   0.0035    0.5973     0.7208   0.7658
  0.97        1   0.0005    0.3740     0.5666   0.6089
```

Scene head @0.5: smoke precision 0.9424 recall 0.9428 fpr 0.0539; fire precision 0.9576 recall 0.9309 fpr 0.0144.
