from keras.metrics import MeanIoU

# Создание метрики mIoU
iou_metric = MeanIoU(num_classes=6)  # 6 классов без background

# Вычисление mIoU на тестовой выборке
iou_metric.update_state(y_true, y_pred)
iou_value = iou_metric.result()

# Вывод значений mIoU
print("mIoU for body:", iou_value)