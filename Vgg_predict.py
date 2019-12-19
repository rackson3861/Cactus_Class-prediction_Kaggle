# make a prediction for a new image.
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import os
# load and prepare the image
present = []
absent = []
pr_prob = []
ab_prob = []
def load_image(filename):
	# load the image
	img = load_img(filename, target_size=(32, 32))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 3 channels
	img = img.reshape(1, 32, 32, 3)
	# center pixel data
	img = img.astype('float32')
	return img

# load an image and predict the class
def run_example():
	# load the image
    model = load_model('final_model.h5')
    i=0
    for file_name in os.listdir("test"):
        i=i+1
        s = "test/"+file_name
        img = load_image(s)
        result = model.predict(img)
        if result[0] <0.5:
            absent.append(file_name)
            ab_prob.append(result[0])
            
        else:
            present.append(file_name)
            pr_prob.append(result[0])
            
        print(i)

# entry point, run the example
run_example()

absent = pd.DataFrame(absent)
present = pd.DataFrame(present)
ab_prob = pd.DataFrame(ab_prob)
pr_prob = pd.DataFrame(pr_prob)

absent = pd.concat([absent,ab_prob],axis=1)
present = pd.concat([present,pr_prob],axis=1)

absent.columns = ['image','probabilty']
present.columns = ['image','probabilty']

absent.to_csv('absent_test.csv',index = False)
present.to_csv('present_test.csv', index = False)