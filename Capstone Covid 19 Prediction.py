# Capstone Project

from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten,Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

from kerastuner import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters


# re-size all the images to this
IMAGE_SIZE = [224, 224]

train_path = "Dataset\Chest X-Ray and CT Scan Images\Train"
test_path = "Dataset\Chest X-Ray and CT Scan Images\Test"


  
def build_model(hp):

    model=Sequential()
    model.add(Conv2D(filters=hp.Int('filter1',min_value=16, max_value=164, step=16),kernel_size=hp.Choice('kernel 1',values=[2,3,4,5]),padding="same",activation="relu",input_shape=(224,224,3)))
    model.add(MaxPooling2D(pool_size=hp.Choice('pool 1',values=[2,3,4,5])))
    model.add(Conv2D(filters=hp.Int('filter2',min_value=16, max_value=164, step=16),kernel_size=hp.Choice('kernel 2',values=[2,3,4,5]),padding="same",activation ="relu"))
    model.add(MaxPooling2D(pool_size=hp.Choice('pool 2',values=[2,3,4,5])))
    model.add(Conv2D(filters=hp.Int('filter3',min_value=16, max_value=164, step=16),kernel_size=hp.Choice('kernel 3',values=[2,3,4,5]),padding="same",activation="relu"))
    model.add(MaxPooling2D(pool_size=hp.Choice('pool 3',values=[2,3,4,5])))
    model.add(Flatten())
    model.add(Dense(hp.Int('dense1',min_value=300, max_value=800, step=50),activation="relu"))
    model.add(Dense(2,activation="softmax"))
    model.summary()
    
    model.compile(
      loss='categorical_crossentropy',
      optimizer='adam',
      metrics=['accuracy']
    )
    
    return model

tuner_search=RandomSearch(build_model,objective='accuracy',max_trials=5,directory=r"C:\Users\khares\Work\output13")

tuner_search.search(training_set, epochs=5)

model=tuner_search.get_best_models(num_models=1)[0]

model.summary()

best_hyperparameters = tuner_search.get_best_hyperparameters(1)[0].values
best_hyperparameters


model=Sequential()
model.add(Conv2D(filters=48,kernel_size=5,padding="same",activation="relu",input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=4))
model.add(Conv2D(filters=160,kernel_size=3,padding="same",activation ="relu"))
model.add(MaxPooling2D(pool_size=4))
model.add(Conv2D(filters=144,kernel_size=5,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(550,activation="relu"))
model.add(Dense(2,activation="softmax"))
model.summary()

model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)



train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(train_path,
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(test_path,
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')

# fit the model
r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=50,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)

# plot the loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# plot the accuracy
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')


model.save('model_covid19GPU100e.h5')


