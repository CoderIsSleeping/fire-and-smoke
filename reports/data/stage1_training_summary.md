# Training summary

- data: `/kaggle/input/datasets/coderissleeping/fire-and-smoke/fire_smoke_yolo/fire_smoke_yolo/data.yaml`
- backbone: `vit_small_patch16_dinov3.lvd1689m` (unfrozen blocks: 0)
- image size: 640, batch 8 x accum 1
- epochs run: 40
- best val mAP@0.5: 0.7312

## Validation detection metrics (best epoch)

```
   class     #GT     AP50     AP75   AP50-95
--------------------------------------------
   smoke    1889   0.8216   0.4361    0.4479
    fire    2154   0.6377   0.2394    0.2997
--------------------------------------------
    mean           0.7297   0.3378    0.3738
```

## Alarm sweep on the validation split

```
  conf  FP imgs      FPR  rec fire  rec smoke  rec any
------------------------------------------------------
  0.05      824   0.5118    0.9977     0.9976   0.9978
  0.10      584   0.3627    0.9955     0.9964   0.9973
  0.15      445   0.2764    0.9932     0.9946   0.9956
  0.20      361   0.2242    0.9910     0.9922   0.9935
  0.25      284   0.1764    0.9899     0.9910   0.9918
  0.30      245   0.1522    0.9865     0.9856   0.9896
  0.35      209   0.1298    0.9831     0.9832   0.9869
  0.40      169   0.1050    0.9809     0.9808   0.9847
  0.45      148   0.0919    0.9775     0.9772   0.9820
  0.50      123   0.0764    0.9764     0.9736   0.9793
  0.55      102   0.0634    0.9730     0.9700   0.9766
  0.60       83   0.0516    0.9696     0.9664   0.9733
  0.65       69   0.0429    0.9516     0.9604   0.9706
  0.70       56   0.0348    0.9448     0.9520   0.9651
  0.75       36   0.0224    0.9212     0.9364   0.9537
  0.80       24   0.0149    0.8975     0.9148   0.9373
  0.85       15   0.0093    0.8615     0.8830   0.9150
  0.90        5   0.0031    0.7579     0.8295   0.8698
  0.95        1   0.0006    0.5327     0.7119   0.7499
```

`FPR` is the fraction of verified-negative images that produced at least one
box at that confidence. Multiply by the frame rate to get false boxes per
second before temporal confirmation.
