# This is a helper module that contains conveniences to access the MS COCO
# dataset. You can modify at will.  In fact, you will almost certainly have
# to, or implement otherwise.

import sys

# This is evil, forgive me, but practical under the circumstances.
# It's a hardcoded access to the COCO API.
from sre_parse import Tokenizer

from keras import Model
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences

COCOAPI_PATH='/scratch/lt2316-h18-resources/cocoapi/PythonAPI/'
TRAIN_ANNOT_FILE='/scratch/lt2316-h18-resources/coco/annotations/instances_train2017.json'
VAL_ANNOT_FILE='/scratch/lt2316-h18-resources/coco/annotations/instances_val2017.json'
TRAIN_CAP_FILE='/scratch/lt2316-h18-resources/coco/annotations/captions_train2017.json'
VAL_CAP_FILE='/scratch/lt2316-h18-resources/coco/annotations/captions_val2017.json'
TRAIN_IMG_DIR='/scratch/lt2316-h18-resources/coco/train2017/'
VAL_IMG_DIR='/scratch/lt2316-h18-resources/coco/val2017/'
annotfile = TRAIN_ANNOT_FILE
capfile = TRAIN_CAP_FILE
imgdir = TRAIN_IMG_DIR

sys.path.append(COCOAPI_PATH)
from pycocotools.coco import COCO

annotcoco = None
capcoco = None
catdict = {}

# OK back to normal.
import random
import skimage.io as io
import skimage.transform as tform
import numpy as np
import keras

def setmode(mode):
    '''
    Set entire module's mode as 'train' or 'test' for the purpose of data extraction.
    '''
    global annotfile
    global capfile
    global imgdir
    global annotcoco, capcoco
    global catdict
    if mode == "train":
        annotfile = TRAIN_ANNOT_FILE
        capfile = TRAIN_CAP_FILE
        imgdir = TRAIN_IMG_DIR
    elif mode == "test":
        annotfile = VAL_ANNOT_FILE
        capfile = VAL_CAP_FILE
        imgdir = VAL_IMG_DIR
    else:
        raise ValueError

    annotcoco = COCO(annotfile)
    capcoco = COCO(capfile)

    # To facilitate category lookup.
    cats = annotcoco.getCatIds()
    catdict = {x:(annotcoco.loadCats(ids=[x])[0]['name']) for x in cats}
    
def query(queries, exclusive=True):
    '''
    Collects mutually-exclusive lists of COCO ids by queries, so returns 
    a parallel list of lists.
    (Setting 'exclusive' to False makes the lists non-exclusive.)  
    e.g., exclusive_query([['toilet', 'boat'], ['umbrella', 'bench']])
    to find two mutually exclusive lists of images, one with toilets and
    boats, and the other with umbrellas and benches in the same image.
    '''
    if not annotcoco:
        raise ValueError
    imgsets = [set(annotcoco.getImgIds(catIds=annotcoco.getCatIds(catNms=x))) for x in queries]
    if len(queries) > 1:
        if exclusive:
            common = set.intersection(*imgsets)
            return [[x for x in y if x not in common] for y in imgsets]
        else:
            return [list(y) for y in imgsets]
    else:
        return [list(imgsets[0])]
    
def get_captions_for_ids(idlist):
    annids =  capcoco.getAnnIds(imgIds=idlist)
    anns = capcoco.loadAnns(annids)
    return [ann['caption'] for ann in anns]    

def get_cats_for_img(imgid):
    '''
    Takes an image id and gets a category list for it.
    '''
    if not annotcoco:
        raise ValueError
        
    imgannids = annotcoco.getAnnIds(imgIds=imgid)
    imganns = annotcoco.loadAnns(imgannids)
    return list(set([catdict[x['category_id']] for x in imganns]))
    
    
def iter_captions(idlists, cats, batch=1):
    '''
    Obtains the corresponding captions from multiple COCO id lists.
    Randomizes the order.  
    Returns an infinite iterator (do not convert to list!) that returns tuples (captions, categories)
    as parallel lists at size of batch.
    '''
    if not capcoco:
        raise ValueError
    if batch < 1:
        raise ValueError

    full = []
    for z in zip(idlists, cats):
        for x in z[0]:
            full.append((x, z[1]))
        
    while True:
        randomlist = random.sample(full, k=len(full))
        captions = []
        labels = []

        for p in randomlist:
            annids =  capcoco.getAnnIds(imgIds=[p[0]])
            anns = capcoco.loadAnns(annids)
            for ann in anns:
                captions.append(ann['caption'])
                # For LSTM you may want to do more with the captions
                # or otherwise distribute the data.
                labels.append(p[1])
                if len(captions) % batch == 0:
                    yield (captions, labels)
                    captions = []
                    labels = []

def iter_captions_cats(idlists, cats, batch=1):
    '''
    Obtains the corresponding captions from multiple COCO id lists alongside all associated image captions per image.
    Randomizes the order.  
    Returns an infinite iterator (do not convert to list!) that returns tuples (captions, categories)
    as parallel lists at size of batch.
    '''
    if not capcoco:
        raise ValueError
    if batch < 1:
        raise ValueError

    full = []
    for z in zip(idlists, cats):
        for x in z[0]:
            full.append((x, z[1]))
        
    while True:
        randomlist = random.sample(full, k=len(full))
        captions = []
        labels = []

        for p in randomlist:
            annids =  capcoco.getAnnIds(imgIds=[p[0]])
            anns = capcoco.loadAnns(annids)
            for ann in anns:
                imgid = ann['image_id']
                cats = get_cats_for_img(imgid)
                captions.append((ann['caption'], cats))
                # For LSTM you may want to do more with the captions
                # or otherwise distribute the data.
                labels.append(p[1])
                if len(captions) % batch == 0:
                    yield (captions, labels)
                    captions = []
                    labels = []
                    
def iter_images(idlists, cats, size=(200,200), batch=1):
    '''
    Obtains the corresponding image data as numpy array from multiple COCO id lists.
    Returns an infinite iterator (do not convert to list!) that returns tuples (imagess, categories)
    as parallel lists at size of batch.
    By default, randomizes the order and resizes the image.
    '''
    if not annotcoco:
        raise ValueError
    if batch < 1:
        raise ValueError
    if not size:
        raise ValueError # size is mandatory

    full = []
    for z in zip(idlists, cats):
        for x in z[0]:
            full.append((x, z[1]))

    while True:
        randomlist = random.sample(full, k=len(full))

        images = []
        labels = []
        for r in randomlist:
            imgfile = annotcoco.loadImgs([r[0]])[0]['file_name']
            img = io.imread(imgdir + imgfile)
            imgscaled = tform.resize(img, size)
            # Colour images only.
            if imgscaled.shape == (size[0], size[1], 3):
                images.append(imgscaled)
                labels.append(r[1])
                if len(images) % batch == 0:
                    yield (np.array(images), np.array(labels))
                    images = []
                    labels = []


def create_tokenizer() -> keras.preprocessing.text.Tokenizer:
    """
    Creates a tokenizer and fits it to the set of captions
    :return:
    """
    # Find all captions
    image_ids = annotcoco.getImgIds()
    annotation_ids = capcoco.getAnnIds(imgIds=image_ids)
    annotations = capcoco.loadAnns(annotation_ids)
    captions = [annotation['caption'] for annotation in annotations]

    # Create and fit tokenizer
    tokenizer = keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(captions)
    return tokenizer


def get_max_caption_size() -> int:
    """
    Gets the length of the longest caption
    """
    image_ids = annotcoco.getImgIds()
    annotation_ids = capcoco.getAnnIds(imgIds=image_ids)
    annotations = capcoco.loadAnns(annotation_ids)
    captions = [annotation['caption'] for annotation in annotations]
    return max(map(lambda caption: len(caption), captions))


def count_all_images():
    """
    Gets the total count of images available
    """
    return len(annotcoco.getImgIds())


def iter_all_captions_images(tokenizer: keras.preprocessing.text.Tokenizer, encoder: Model, batch=32, size=(200,200)):
    """
    Creates an iterator for all images and captions.
    Resizes and encodes images using the provided autoencoder.
    Splits caption into all prefixes, followed by a single predicted word.
    Batches generated output for both word and image predictions based on an array of word encodings.
    :param tokenizer: Tokenizer to encode words
    :param encoder: Autoencoder to encode images
    :param batch: Size of produced batches
    :param size: Required image size for autoencoder
    :return: Generator for encoded captions and encoded images
    """
    # Find all ids
    id_list = annotcoco.getImgIds()
    max_caption_size = get_max_caption_size()
    tokenizer_length = len(tokenizer.word_index)
    while True:
        # Determine random id order and reset batch
        random_ids = random.sample(id_list, len(id_list))
        captions = []
        images = []
        for image_id in random_ids:
            # Parse image, make sure it is has 3 components
            image = io.imread(imgdir + annotcoco.loadImgs(image_id)[0]['file_name'])
            if not image.shape == (image.shape[0], image.shape[1], 3):
                continue

            # Encode the image
            image = tform.resize(image, size)
            image = encoder.predict(image[np.newaxis, :])[0]

            # Encode the caption
            annotation_id = capcoco.getAnnIds(imgIds=[image_id])[0]
            annotation = capcoco.loadAnns(annotation_id)[0]
            caption = tokenizer.texts_to_sequences([annotation['caption']])
            caption = caption[0]

            # Yield results for all caption prefixes
            for i in range(1, len(caption)):
                caption_prefix = caption[:i + 1]
                images.append(image)
                captions.append(caption_prefix)

                # Yield a batch and reset lists
                if len(images) % batch == 0:
                    captions = np.array(pad_sequences(captions, maxlen=max_caption_size, padding='pre'))
                    word_prefix = captions[:, :-1]
                    word_prediction = to_categorical(captions[:, -1], num_classes=tokenizer_length)
                    image_prediction = np.array(images)
                    yield (word_prefix, {'word_prediction': word_prediction, 'image_prediction': image_prediction})
                    captions = []
                    images = []


def get_all_encoded_images(encoder: Model, size=(200,200)):
    """
    Gets all image ids and encoded images
    """
    image_ids = []
    images = []
    for image_id in annotcoco.getImgIds():
        # Parse image, make sure it is has 3 components
        image = io.imread(imgdir + annotcoco.loadImgs(image_id)[0]['file_name'])
        if not image.shape == (image.shape[0], image.shape[1], 3):
            continue

        # Encode the image
        image = tform.resize(image, size)
        image = encoder.predict(image[np.newaxis, :])[0]
        image_ids.append(image_id)
        images.append(image)

    return image_ids, images


def get_images(image_ids):
    """
    Load images with ids
    """
    for image_path in annotcoco.loadImgs(image_ids):
        full_path = imgdir + image_path['file_name']
        print(full_path)
        yield io.imread(full_path)