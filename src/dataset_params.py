from pathlib import Path

SMALL_SET = {
    'base_path': Path('data')/'KITTI_SMALL',
    'date': '2011_09_26',
    'drives': [5],
}

TRAIN_SET = {
    'base_path': Path('data')/'KITTI',
    'date': '2011_09_26',
    'drives': [1, 2, 9, 11, 13, 14, 15,
               17, 18, 19, 20, 22, 23,
               27, 28, 29, 32, 35, 36, 39,
               46, 48, 51, 52, 56, 57,
               59, 60, 61, 64, 79,
               84, 86, 87, 91, 93, 95,
               96, 101, 104, 106, 113,
               117, 119],
}

VAL_SET = {
    'base_path': Path('data')/'KITTI',
    'date': '2011_09_26',
    'drives': [5, 70],
}

TEST_SET = {
    'base_path': Path('data')/'KITTI',
    'date': '2011_09_30',
    'drives': [28],
}