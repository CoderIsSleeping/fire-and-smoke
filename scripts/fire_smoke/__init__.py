"""Fire & smoke detection package (DINOv3 backbone + detection/scene heads)."""

CLASS_NAMES = ["__background__", "smoke", "fire"]
# YOLO-format class ids in the D-Fire label files.
YOLO_CLASS_NAMES = ["smoke", "fire"]
# Detector label id = yolo class id + 1 (0 is reserved for background).
