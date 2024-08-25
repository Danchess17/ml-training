import numpy as np

# Загрузить маску
mask = np.load('gt_masks/000001.npy')

# Создать маски для уровней иерархии
mask_level0 = mask  # Маска для уровня 0 (body)
mask_level1 = np.zeros_like(mask)
mask_level1[(mask == 1) | (mask == 6)] = 1  # Маска для уровня 1 (upper_body)
mask_level1[(mask == 3) | (mask == 5)] = 2  # Маска для уровня 1 (lower_body)
mask_level2 = mask  # Маски для уровня 2 (исходные маски)