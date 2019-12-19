# save the final model to file
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
# define cnn model
def define_model():
	# load model
	model = VGG19(include_top=False, input_shape=(32, 32, 3))
	# mark loaded layers as not trainable
	for layer in model.layers:
		layer.trainable = False
	# add new classifier layers
	flat1 = Flatten()(model.layers[-1].output)
	class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
    class2 = Dense(64, activation='relu', kernel_initializer='he_uniform')(class1)
    class3 = Dense(32, activation='relu', kernel_initializer='he_uniform')(class2)
    
	output = Dense(1, activation='sigmoid')(class3)
	# define new model
	model = Model(inputs=model.inputs, outputs=output)
	# compile model
	opt = SGD(lr=0.001, momentum=0.9, nesterov = True)
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	return model

# run the test harness for evaluating a model
def run_test_harness():
    model = define_model()
    datagen = ImageDataGenerator(featurewise_center=True)
    train_it = datagen.flow_from_directory('train_set',class_mode='binary', batch_size=64, target_size=(32, 32))
    val_it = datagen.flow_from_directory('validation_set',class_mode='binary', batch_size=64, target_size=(32, 32))
    """model = load_model('final_model.h5')"""
    model.fit_generator(train_it, steps_per_epoch=len(train_it),validation_data = val_it, epochs=35, verbose=1,nb_val_samples = 200)
    model.save('final_model.h5')
    
run_test_harness()
    
    