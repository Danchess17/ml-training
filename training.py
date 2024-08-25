from keras.optimizers import *
from keras.losses import *
from model import MyUnet
from preproccesing import dataset_padding, without_padding
from keras import backend as be
import os 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

image_dir = 'Pascal-part/JPEGImages'
mask_dir = 'Pascal-part/gt_masks'
X_train, y_train = dataset_padding(image_dir, mask_dir, mode='train')
X_val, y_val = dataset_padding(image_dir, mask_dir, mode='val')
# X_train, y_train = without_padding(image_dir, mask_dir, mode='train')
# X_val, y_val = without_padding(image_dir, mask_dir, mode='val')


model = MyUnet()
# model.summary()
be.clear_session()

model.compile(optimizer=Adam(learning_rate=1e-4),
              loss=CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
print('compiled')

history = model.fit(X_train[:2000], y_train[:2000], epochs=7, batch_size=1, validation_data=(X_val, y_val))
