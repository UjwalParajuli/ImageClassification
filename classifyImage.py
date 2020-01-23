from keras.models import model_from_json
from pathlib import Path
from keras.preprocessing import image
import numpy as np
from tkinter import *
from tkinter import messagebox
from PIL import ImageTk, Image
from tkinter import filedialog
from keras.preprocessing import image
from keras.applications import vgg16
import os

# These are the CIFAR10 class labels from the training data (in order from 0 to 9)
class_labels = [
    "Plane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck"
]

root = Tk()
root.title("Image Classification")
root.geometry('900x800')
root.resizable(False, False)
root.config(bg="skyblue", padx=20, pady=20)

frame = LabelFrame(root, borderwidth = 0, highlightthickness = 0, padx = 20, pady = 20)
frame.grid(row = 0, column = 0)
frame.config(bg="skyblue")

frameRight = LabelFrame(root, borderwidth = 0, highlightthickness = 0, padx = 20, pady = 20)
frameRight.grid(row = 0, column = 1)
frameRight.config(bg="skyblue")

frame2 = LabelFrame(frameRight, borderwidth = 0, highlightthickness = 0, padx = 20, pady = 75)
frame2.grid(row = 0, column = 0, sticky = NS)
frame2.config(bg="skyblue")

frame3 = LabelFrame(frameRight, borderwidth = 0, highlightthickness = 0, padx = 20, pady = 20)
frame3.grid(row = 1, column = 0)
frame3.config(bg="skyblue")


def save():
    try:
        global myImage, myImage2, myImageLabel, img
        root.filename = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select an Image", filetypes=[("Image Files", ("*.png", "*.jpg", "*.jpeg"))])
        img = Image.open(root.filename)
        myImage = img.resize((512, 512), Image.ANTIALIAS)
        myImage2 = ImageTk.PhotoImage(myImage)
        myImageLabel = Label(frame, image = myImage2)
        myImageLabel.grid(row=1, column=0, pady=20)

        noteLabel = Label(frame, justify=LEFT,  text = "Note: Custom trained model can detect only 10 types of objects;" + "\n" + "Plane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck" + "\n"
                          + "If other than those above objects appears on the image it shows incorrect result.")
        noteLabel.config(fg="red")
        noteLabel.grid(row = 2, column = 0, pady = 20)

        # Load the json file that contains the model's structure
        f = Path("model_structure.json")
        model_structure = f.read_text()

        # Recreate the Keras model object from the json data
        model = model_from_json(model_structure)

        # Re-load the model's trained weights
        model.load_weights("model_weights.h5")

        # Load an image file to test, resizing it to 32x32 pixels (as required by this model)
        img = image.load_img(root.filename, target_size=(32, 32))

        # Convert the image to a numpy array
        image_to_test = image.img_to_array(img)

        # Add a fourth dimension to the image (since Keras expects a list of images, not a single image)
        list_of_images = np.expand_dims(image_to_test, axis=0)

        # Make a prediction using the model
        results = model.predict(list_of_images)

        # Since we are only testing one image, we only need to check the first result
        single_result = results[0]

        # We will get a likelihood score for all 10 possible classes. Find out which class had the highest score.
        most_likely_class_index = int(np.argmax(single_result))
        class_likelihood = single_result[most_likely_class_index] * 100


        # Get the name of the most likely class
        class_label = class_labels[most_likely_class_index]

        # Print the result of Custom Model
        textLabel = Label(frame2, justify = LEFT, padx = 10, pady = 10, text = "Using Custom Trained Model:" + "\n" + "\n" + "This image is a " + class_label + "\n" + "\n" + "Accuracy: " + str((class_likelihood).round(2)) + "%")
        textLabel.grid(row = 0, column = 0)

        # Load Keras' VGG16 model that was pre-trained against the ImageNet database
        model = vgg16.VGG16()

        # Load the image file, resizing it to 224x224 pixels (required by this model)
        img = image.load_img(root.filename, target_size=(224, 224))

        # Convert the image to a numpy array
        x = image.img_to_array(img)

        # Add a fourth dimension (since Keras expects a list of images)
        x = np.expand_dims(x, axis=0)

        # Normalize the input image's pixel values to the range used when training the neural network
        x = vgg16.preprocess_input(x)

        # Run the image through the deep neural network to make a prediction
        predictions = model.predict(x)

        # Look up the names of the predicted classes. Index zero is the results for the first image.
        predicted_classes = vgg16.decode_predictions(predictions)
        List = []
        for imagenet_id, name, likelihood in predicted_classes[0]:
            List.append(name)
            List.append(likelihood)

        textLabel2 = Label(frame3, justify=LEFT, padx=20, pady=10,
                          text="Using Pre-Trained Model:" + "\n" + "\n" + "This image is a "  + List[0] + "\n" + "\n" + "Accuracy: " + " " + str((List[1] * 100).round(2)) + "%")
        textLabel2.grid(row=0, column=0)

    except Exception as e:
        messagebox.showerror("File Read Error", "No Image Selected")

openButton = Button(frame, text="Select Image", command=save)
openButton.grid(row=0, column=0, sticky=W)

root.mainloop()
