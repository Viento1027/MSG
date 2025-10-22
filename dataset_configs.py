DATASETS = {
    # Salinas (512, 217, 224)
    "Salinas": {
        "hsi_path": "./data/Salinas/Salinas.mat",
        "gt_path": "./data/Salinas/Salinas_gt.mat",
        "hsi_key": "salinas",
        "gt_key": "salinas_gt",
        "num_classes": 16
    },
    # Berlin (1723, 476, 244)
    "Berlin": {
        "hsi_path": "./data/Berlin/berlin_hsi.mat",
        "gt_path": "./data/Berlin/berlin_gt.mat",
        "hsi_key": "berlin_hsi",
        "gt_key": "berlin_gt",
        "num_classes": 8
    },
    # WHU_Hi_LongKou (550, 400, 270)
    "WHU_Hi_LongKou": {
        "hsi_path": "./data/WHU_Hi_LongKou/WHU_Hi_LongKou.mat",
        "gt_path": "./data/WHU_Hi_LongKou/WHU_Hi_LongKou_gt.mat",
        "hsi_key": "WHU_Hi_LongKou",
        "gt_key": "WHU_Hi_LongKou_gt",
        "num_classes": 9
    }
}
