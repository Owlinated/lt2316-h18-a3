# This file is the primary way that your assignment will be tested.
# It will be loaded as a Jupyter notebook, and the "predictive_search"
# function will be called in the notebook line.
#
# You can add whatever functions you want, as long as the PredictiveSearch
# class is present in this file.  You can import whatever you need,
# both your own modules, mycoco.py, and third party modules available on mltgpu
# (other than what literally solves the assignment, if any such thing
# exists).

# Do not use GPU for testing (it is probably busy training)
from os import environ
from pathlib import Path

from keras import Model
from keras.engine.saving import load_model
from numpy import argmax, array

print("Disabling GPU")
environ['CUDA_VISIBLE_DEVICES'] = '-1'

from keras_preprocessing.sequence import pad_sequences
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

import mycoco

class PredictiveSearch:
    def __init__(self, modelfile=None):
        """
        Load the model however you want, do whatever initialization you
        need here.  You can change the method signature to require
        more parameters if you think you need them, as long as you document
        this.
        """
        mycoco.setmode('test')
        mode_root = Path(__file__).parent.joinpath("model")

        # Load autoencoder from assignment 2
        auto_encoder_file = str(mode_root.joinpath("autoencoder.h5"))
        auto_encoder = load_model(auto_encoder_file)
        encoder = Model(inputs=auto_encoder.input, outputs=auto_encoder.get_layer("encoder").output)
        encoder._make_predict_function()

        # Load model from train.py
        if modelfile is None:
            modelfile = str(mode_root.joinpath("model.h5"))
        self.model = load_model(modelfile)

        # Initialize word encoder
        self.tokenizer = mycoco.create_tokenizer()
        self.tokenizer_length = len(self.tokenizer.word_index)
        self.max_caption_size = 249
        self.word_count = 5

        # Generate encoded representations for all images
        print("Loading images...")
        self.image_ids, self.images = mycoco.get_all_encoded_images(encoder)
        self.image_ids = array(self.image_ids)
        self.images = array(self.images)
        self.images = self.images.reshape((self.images.shape[0], -1))
        print("Initializing nearest neighbor search...")
        self.nearest_neighbors = NearestNeighbors(n_neighbors=3).fit(self.images)

    def predictive_search(self, words: str):
        """
        Based on the loaded model data, do the following:

        0. Process the text so that it is compatible with your model input.
           (I.e., do whatever tokenization you did for training, etc.)
        1. Strip out all the out-of-vocabulary words (ie, words not
           represented in the model). Print them in a message/list of
           messages.
        2. If there are no words left, raise an error.
        3. From the remaining words, predict and print out the top five
           most likely words according to the model to be the next word.
        4. Predict and display (using e.g. plt.show(), so that it appears
           on the Jupyter console) the top three corresponding images
           that the model predicts.
        """
        # 0. - 2. Tokenizer does that automatically, print unmapped words:
        words = words.lower()
        for word in words.split():
            if word not in self.tokenizer.word_index:
                print(f"Word not in training set: {word}")

        # 3. Predict five additional words, keep track of latest predicted image
        image = None
        for _ in range(self.word_count):
            tokens = self.tokenizer.texts_to_sequences([words])[0]
            tokens = pad_sequences([tokens], maxlen=self.max_caption_size, padding='pre')
            predicted = self.model.predict(tokens, verbose=0)
            predicted_class = argmax(predicted[0])
            predicted_word = self.tokenizer.index_word[predicted_class]
            words += " " + predicted_word
            image = predicted[1]
        print(f"Predicted sentence: {words}")

        # 4. Find three nearest neighbors of all images
        _, indices = self.nearest_neighbors.kneighbors(image.reshape((1, -1)))
        image_ids = self.image_ids[indices[0]]
        for image in mycoco.get_images(image_ids):
            plt.imshow(image)


if __name__ == "__main__":
    ps = PredictiveSearch()
    ps.predictive_search("A toilet standing")
    ps.predictive_search("the Man is")
