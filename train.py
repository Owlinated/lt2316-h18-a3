# You can put whatever you want in here for training the model,
# provided you document it well enouth for us to understand. Use of
# the argument parser is recommended but not required.

# Limit GPU usage
from os import environ
print("Limiting gpu usage")
environ['CUDA_VISIBLE_DEVICES'] = '2'

from argparse import ArgumentParser

from keras import Model, Input
from keras.callbacks import ModelCheckpoint
from keras.engine.saving import load_model
from keras.layers import Dense, Reshape
from keras.layers import Embedding
from keras.layers import LSTM
from keras_preprocessing.text import Tokenizer

import mycoco


def create_model(tokenizer: Tokenizer, embedding_dims = 10) -> Model:
    """
    Create model which takes some words of a caption
    and predicts both the next word and the encoded representation of the matching image
    :param tokenizer: Tokenizer used to encode input
    :return: Compiled model
    """
    tokenizer_length = len(tokenizer.word_index)
    max_caption_size = mycoco.get_max_caption_size()

    input_layer = Input(shape=(max_caption_size - 1,))

    temp_layer = Embedding(tokenizer_length, embedding_dims, input_length=mycoco.get_max_caption_size(), name="input_layer")(input_layer)
    temp_layer = LSTM(50, name="lstm_layer")(temp_layer)

    word_prediction = Dense(tokenizer_length, activation="softmax", name="word_prediction")(temp_layer)

    image_dense = Dense(10 * 10 * 8, activation="relu", name="image_dense_layer")(temp_layer)
    image_prediction = Reshape((10, 10, 8), name="image_prediction")(image_dense)

    model = Model(input_layer, [word_prediction, image_prediction])
    losses = {
        "word_prediction": "categorical_crossentropy",
        "image_prediction": "mean_absolute_error",
    }
    model.compile("adam", loss=losses)

    print(model.summary())
    return model


def train_save():
    # Load autoencoder model
    auto_encoder = load_model(args.autoencodermodelfile)
    encoder = Model(inputs=auto_encoder.input, outputs=auto_encoder.get_layer("encoder").output)
    encoder._make_predict_function()

    # Load tokenizer for captions
    tokenizer = mycoco.create_tokenizer()

    # Create generator
    batch_size = 32
    generator = mycoco.iter_all_captions_images(tokenizer, encoder, batch=batch_size)

    # Create and train model
    model = create_model(tokenizer)
    model.fit_generator(
        generator,
        epochs=100,
        steps_per_epoch=mycoco.count_all_images() / batch_size,
        callbacks=[
            ModelCheckpoint(args.checkpointdir + '/checkpoint.{epoch:04d}.h5', monitor='loss', mode='auto'),
        ])

    model.save(args.modelfile)


if __name__ == "__main__":
    parser = ArgumentParser("Train a model.")
    parser.add_argument("autoencodermodelfile", type=str, help="autoencoder model file from assignment 2")
    parser.add_argument('checkpointdir', type=str, help="directory for storing checkpointed models")
    parser.add_argument('modelfile', type=str, help="output model file")
    args = parser.parse_args()

    mycoco.setmode('train')
    train_save()