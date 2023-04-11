# %%
print('Loading models...')

import av
import os
import face_recognition
from PIL import Image
from prettytable import PrettyTable

import numpy as np
import shutil
import math
import random
import pickle
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import tensorflow_model_optimization as tfmot

import keras
from keras import layers
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, Dropout, LeakyReLU
from keras.optimizers import Adam
from keras.utils import img_to_array

import warnings
warnings.filterwarnings("ignore")

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# %%
INPUT_DIR = 'input_files/'
INTERMEDIARY_DIR = 'intermediary_files/'
OUTPUT_DIR = 'output_files/'

TOP_DIRS = [INPUT_DIR, INTERMEDIARY_DIR, OUTPUT_DIR]

IFRAME_DIR = 'iframes/'
FACECROP_DIR = 'faces/'
RESIDUAL_DIR = 'residual/'

PREPROCESS_DIRS = [IFRAME_DIR, FACECROP_DIR, RESIDUAL_DIR]

MESONET_PATH  = 'saved_models/mesonet/'
SRM_PATH      = 'saved_models/srm/'
TEMPORAL_PATH = 'saved_models/temporal/'
SVM_PATH = 'saved_models/svm/'

MODEL_DIRS = [MESONET_PATH, SRM_PATH, TEMPORAL_PATH, SVM_PATH]

# %%
for dir in TOP_DIRS:
    try:
        os.makedirs(dir)
    except Exception:
        pass

for dir in PREPROCESS_DIRS:
    try:
        os.makedirs(INTERMEDIARY_DIR + dir)
    except Exception:
        pass

for dir in MODEL_DIRS:
    try:
        os.makedirs(dir)
    except Exception:
        pass

# %% [markdown]
# # Flow
# 
# - Read all videos present in input_files folder
# - For each video in the input directory
#     - Extract I-Frames and crop faces
#     - Extract Extract residuals
#     - save face-cropped video and residuals video
#     - In Frame-level stream
#         - Extract all frames in face-cropped video
#         - Take average of prediction results as video score
#     - In SRM stream
#         - Extract snippets from face-cropped video
#         - Take average of prediction results as video score
#     - In Temporal stream
#         - Extract all residuals from residual video
#         - Take average of prediction per segment
#         - Select the most extreme value as video score (closest to 0 or 1)
#     - In score aggregation
#         - Take average of three scores
#         - Use voting to determine class (Use extreme value of major class as video score)
#         - Use trained svm model to predict class probabilities

# %% [markdown]
# # Functions

# %% [markdown]
# ## Pre-Processing

# %%
def extract_iframes(fp):
    input_vid = av.open(fp)
    output_vid = av.open(INTERMEDIARY_DIR + IFRAME_DIR + os.path.split(fp)[1], 'w')

    in_stream = input_vid.streams.video[0]
    in_stream.codec_context.skip_frame = "NONKEY"

    out_stream = output_vid.add_stream(template=in_stream)

    for packet in input_vid.demux(in_stream):
        if packet.dts is None:
            continue

        if packet.is_keyframe:
            packet.stream = out_stream
            output_vid.mux(packet)

    input_vid.close()
    output_vid.close()

# %%
# MesoNet works best with images having 256x256 dimension
# If face location borders span a smaller distance, extend the borders
# on either side equally to ensure 256x256 image

def normalize_face_borders(low, high, max_val, req_dim):
    diff = high - low
    if diff >= 256:
        return

    offset = float((req_dim - diff)) / 2
    low = max(0, low - offset)
    high = min(max_val, high + offset)

    return low, high

# %%
# Face Location: (left, top, right, bottom)
def modify_crop_window(face_location, height, width, req_dim):
    left, right = normalize_face_borders(face_location[0], face_location[2], width, req_dim)
    top, bot = normalize_face_borders(face_location[1], face_location[3], height, req_dim)

    face_location = (left, top, right, bot)

    return face_location

# %%
def save_cropped_faces_to_video(fp, req_dim):
    input = av.open(fp)
    output = av.open(INTERMEDIARY_DIR + FACECROP_DIR + os.path.split(fp)[1], 'w')

    in_stream = input.streams.video[0]
    codec_name = in_stream.codec_context.name

    # output video dimension should be 256x256
    out_stream = output.add_stream(codec_name, rate=8)
    out_stream.width = 256
    out_stream.height = 256
    out_stream.pix_fmt = in_stream.codec_context.pix_fmt

    for frame in input.decode(in_stream):
        img_frame = frame.to_image()
        nd_frame = frame.to_ndarray()

        # Face location returned by face_recognition api: [(top, right, bottom, left)]
        # Origin considered at top left corner of image => right margin > left margin, bottom > top
        face_location = face_recognition.api.face_locations(nd_frame)

        # if can't find a face, then skip that frame
        # TODO : sync frame skipping with temporality stream
        if len(face_location) == 0:
            continue

        # Face location required by PIL.Image: (left, top, right, bottom)
        face_location = (face_location[0][3], face_location[0][0], 
                         face_location[0][1], face_location[0][2])
            
        # Modify crop window size only if positive value given.
        if (req_dim > 0):    
            face_location = modify_crop_window(face_location, img_frame.height, img_frame.width, req_dim)
            
        img_frame = img_frame.crop(face_location)
        
        out_frame = av.VideoFrame.from_image(img_frame)
        out_packet = out_stream.encode(out_frame)
        output.mux(out_packet)

    out_packet = out_stream.encode(None)
    output.mux(out_packet)

    input.close()
    output.close()

# %%
def compute_residual(a, b):
    return Image.fromarray(np.asarray(a) - np.asarray(b))

# %%
def extract_residuals(fp):
    input_vid = av.open(fp)
    output_vid = av.open(INTERMEDIARY_DIR + RESIDUAL_DIR + os.path.split(fp)[1], 'w')

    in_stream = input_vid.streams.video[0]
    codec_name = in_stream.codec_context.name

    # output video dimension should be 256x256
    out_stream = output_vid.add_stream(codec_name, rate=8)
    out_stream.width = 224
    out_stream.height = 224
    out_stream.pix_fmt = in_stream.codec_context.pix_fmt

    # Extract residuals
    frame_list = [frame for frame in input_vid.decode()]
    
    input_vid.seek(0)
    iframe_index = [i for i, packet in enumerate(input_vid.demux()) if packet.is_keyframe]

    residuals = []
    gop_start_index = 0
    for index in iframe_index:
        if index == 0:
            continue

        residual = compute_residual(frame_list[index - 1].to_image(), frame_list[gop_start_index].to_image())
        out_frame = av.VideoFrame.from_image(residual)
        out_packet = out_stream.encode(out_frame)
        output_vid.mux(out_packet)

        gop_start_index = index

    residual = compute_residual(frame_list[-1].to_image(), frame_list[gop_start_index].to_image())
    out_frame = av.VideoFrame.from_image(residual)
    out_packet = out_stream.encode(out_frame)
    output_vid.mux(out_packet)

    out_packet = out_stream.encode(None)
    output_vid.mux(out_packet)

    input_vid.close()
    output_vid.close()

# %% [markdown]
# ## Models

# %%
def extract_frames_from_video(fp):
    vid_container = av.open(fp)

    frames = []
    for frame in vid_container.decode():
        frames.append(frame.to_image())

    return frames

# %%
# Returns the index of frames that begin a new segment (except the first segment)
def get_segment_dividers(frame_count, num_segments):
    segments_per_frame = math.floor(frame_count / num_segments)

    return [(segments_per_frame * i) for i in range(1, num_segments) ]

# %%
# Returns the indices of the frames that will be randomly selected from each segment
# Multiple snippets indices per segment can be returned by setting the num_snippets arg 
def get_snippet_indices(segment_dividers, num_snippets):
    start_index = 0
    num_snippets = 1 if num_snippets <= 0 else num_snippets

    snippet_indices = []
    for end_index in segment_dividers:

        # Extracting multiple snippets per segment (if needed)
        for _ in range(num_snippets):
            snippet_indices.append(random.randint(start_index, end_index - 1))

        start_index = end_index
        
    return snippet_indices

# %%
# Returns an array of randomly selected snippets(PIL.Image) from each segment of the input video
def extract_snippets(fp, num_segments, num_snippets):
    vid_container = av.open(fp)
    vid_stream = vid_container.streams.video[0]
    frame_count = vid_stream.frames

    snippets = []

    # If number of frames in video is less than the number of frames that need to sampled
    # then take all frames in the video
    if frame_count < num_segments * num_snippets:
        for frame in vid_container.decode():
            snippets.append(frame.to_image())

    else:
        segment_dividers = get_segment_dividers(frame_count, num_segments)
        segment_dividers = segment_dividers + [frame_count]

        snippet_indices = get_snippet_indices(segment_dividers, num_snippets)

        frame_index = 0
        for frame in vid_container.decode():
            if frame_index > max(snippet_indices):
                break

            if frame_index in snippet_indices:
                snippets.append(frame.to_image())

            frame_index += 1

        
    vid_container.close()
    return snippets

# %% [markdown]
# ### Mesonet Stream

# %%
def create_model(input_size):
    model = keras.Sequential(name='Mesonet')
    model.add(layers.Conv2D(input_shape=input_size, filters=8, kernel_size=3, activation='relu', padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPool2D(2, 2, padding="same"))

    model.add(layers.Conv2D(input_shape=(128, 128, 8), filters=8, kernel_size=5, activation='relu', padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPool2D(2, 2, padding="same"))

  
    model.add(layers.Conv2D(input_shape=(64, 64, 8), filters=16, kernel_size=5, activation='relu', padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPool2D(4, 4, padding="same"))

  
    model.add(layers.Conv2D(input_shape=(16, 16, 16), filters=16, kernel_size=5, activation='relu', padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPool2D(4, 4, padding="same"))
    model.add(Flatten())

    model.add(Dropout(0.5))
    model.add(layers.Dense(16))
    model.add(layers.LeakyReLU())

    model.add(Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
  
    return model

# %%
input_size = (256, 256, 3)
mesonet_model = create_model(input_size)
mesonet_model.compile(optimizer=Adam(learning_rate=0.0001), 
              loss=keras.losses.BinaryCrossentropy(), 
              metrics = [keras.metrics.BinaryAccuracy(), 
                         keras.metrics.Precision(), 
                         keras.metrics.Recall(),
                         keras.metrics.AUC(),
                         keras.metrics.FalseNegatives(),
                         keras.metrics.FalsePositives(),
                         keras.metrics.TrueNegatives(),
                         keras.metrics.TruePositives()])

# %%
# Pruning only dense layers
# Helper function uses `prune_low_magnitude` to make only the 
# Dense layers train with pruning.
def apply_pruning_to_dense(layer):
    
    if isinstance(layer, keras.layers.Dense):
        return tfmot.sparsity.keras.prune_low_magnitude(layer)
    
    return layer

# %%
# Use `tf.keras.models.clone_model` to apply `apply_pruning_to_dense` 
# to the layers of the model.
mesonet_model_dense_prune = keras.models.clone_model(
    mesonet_model,
    clone_function=apply_pruning_to_dense,
)

mesonet_model_dense_prune.compile(
    loss=keras.losses.BinaryCrossentropy(),
    optimizer=Adam(learning_rate=0.0001),
    metrics = [keras.metrics.BinaryAccuracy(), 
                         keras.metrics.Precision(), 
                         keras.metrics.Recall(),
                         keras.metrics.AUC(),
                         keras.metrics.FalseNegatives(),
                         keras.metrics.FalsePositives(),
                         keras.metrics.TrueNegatives(),
                         keras.metrics.TruePositives()]
)

# %%
mesonet_model_dense_prune.load_weights(MESONET_PATH + 'model_pruned').expect_partial()

# %%
def calculate_mesonet_score(model, fp):
    frames = extract_frames_from_video(fp)

    tf_frames = []
    for frame in frames:
        tf_frames.append(tf.convert_to_tensor(frame))

    tf_frames = tf.convert_to_tensor(tf_frames)
    results = model.predict(tf_frames, verbose=0)

    return np.average(results)

# %% [markdown]
# ### SRM Stream

# %%
class SRMLayer(keras.layers.Layer):
    def __init__(self, strides=[1,1,1,1], padding='SAME'):
        super(SRMLayer, self).__init__()
        self.strides = strides
        self.padding = padding

        # Set of 3 fixed SRM Filters used to extract noise & semantic features
        self.filter_small = tf.constant([[0, 0,  0, 0, 0],
                                         [0, 0,  0, 0, 0],
                                         [0, 1, -2, 1, 0],
                                         [0, 0,  0, 0, 0],
                                         [0, 0,  0, 0, 0]], dtype=tf.float32)
        
        self.filter_med = tf.constant([[0,  0,  0,  0, 0],
                                       [0, -1,  2, -1, 0],
                                       [0,  2, -4,  2, 0],
                                       [0, -1,  2, -1, 0],
                                       [0,  0,  0,  0, 0]], dtype=tf.float32)
        
        self.filter_large = tf.constant([[-1,  2,  -2,  2, -1],
                                         [ 2, -6,   8, -6,  2],
                                         [-2,  8, -12,  8, -2],
                                         [ 2, -6,   8, -6,  2],
                                         [-1,  2,  -2,  2, -1]], dtype=tf.float32)

        # Learnability in SRM filters introduced by 'q' values
        # SRM filters are divided by their respective 'q' values before convolution
        # 'q' values are updated during backpropagation using gradient descent
        self.q_small = self.add_weight(name='q_small',
                                       shape=(5, 5, 3, 1),
                                       initializer=keras.initializers.Constant(value=2.0),
                                       trainable=True)
        
        self.q_med = self.add_weight(name='q_med',
                                     shape=(5, 5, 3, 1),
                                     initializer=keras.initializers.Constant(value=4.0),
                                     trainable=True)
        
        self.q_large = self.add_weight(name='q_large',
                                       shape=(5, 5, 3, 1),
                                       initializer=keras.initializers.Constant(value=12.0),
                                       trainable=True)
        
        # 3rd dimension of filters => number of input channels (Three channels)
        self.filter_small = tf.stack([self.filter_small, self.filter_small, self.filter_small], axis=2)
        self.filter_med   = tf.stack([self.filter_med, self.filter_med, self.filter_med], axis=2)
        self.filter_large = tf.stack([self.filter_large, self.filter_large, self.filter_large], axis=2)

        # 4th dimension of filters => number of output feature maps (One feature map)
        # Each filter gives a single output feature map
        self.filter_small = tf.expand_dims(self.filter_small, axis=-1)
        self.filter_med   = tf.expand_dims(self.filter_med, axis=-1)
        self.filter_large = tf.expand_dims(self.filter_large, axis=-1)
        
    def call(self, inputs):
        filter_small = tf.math.divide(self.filter_small, self.q_small)
        filter_med   = tf.math.divide(self.filter_med, self.q_med)
        filter_large = tf.math.divide(self.filter_large, self.q_large)

        output_small = tf.nn.conv2d(inputs, filter_small, strides=self.strides, padding=self.padding)
        output_med   = tf.nn.conv2d(inputs, filter_med,   strides=self.strides, padding=self.padding)
        output_large = tf.nn.conv2d(inputs, filter_large, strides=self.strides, padding=self.padding)

        return tf.concat([output_small, output_med, output_large], axis=3)

    def get_config(self):
        config = super(SRMLayer, self).get_config()
        config.update({'strides': self.strides,
                       'padding': self.padding,
                       'filter_small': self.filter_small,
                       'filter_med': self.filter_med,
                       'filter_large': self.filter_large,
                       'q_small': self.q_small,
                       'q_med': self.q_med,
                       'q_large': self.q_large})
        
        return config


# %%
XceptionNetwork = keras.applications.Xception(
    include_top = False,
    weights = 'imagenet',
    input_shape = (256, 256, 3),
    pooling = max,
    classes = 2
)

# %%
def create_SRM_model(xception_training):
    inputs = keras.layers.Input(shape=(256, 256, 3))
    SRM_noise_maps = SRMLayer()(inputs)
    
    feature_maps = tf.keras.applications.xception.preprocess_input(SRM_noise_maps)
    feature_maps = XceptionNetwork(feature_maps, training=xception_training)

    features = keras.layers.Flatten()(feature_maps)
    features = keras.layers.Dropout(0.8)(features)
    features = keras.layers.Dense(units=130, activation=keras.layers.LeakyReLU())(features)
    outputs = keras.layers.Dense(units=1, activation='sigmoid')(features)

    return keras.Model(inputs, outputs, name='SRM_Model')

# %%
SRM_Model = create_SRM_model(xception_training=False)

# %%
optimizer = keras.optimizers.Adam(beta_1=0.9, 
                                  beta_2=0.999, 
                                  epsilon=1e-6, 
                                  learning_rate=0.00002)

SRM_Model.compile(optimizer=optimizer,
                      loss=keras.losses.BinaryCrossentropy(),
                      metrics=[keras.metrics.BinaryAccuracy(), 
                               keras.metrics.Precision(), 
                               keras.metrics.Recall(),
                               keras.metrics.AUC(),
                               keras.metrics.FalseNegatives(),
                               keras.metrics.FalsePositives(),
                               keras.metrics.TrueNegatives(),
                               keras.metrics.TruePositives()])

# %%
SRM_Model.load_weights(SRM_PATH + 'model').expect_partial()

# %%
def calculate_srm_score(model, fp):
    frames = extract_snippets(fp, num_segments=8, num_snippets=1)

    tf_frames = []
    for frame in frames:
        tf_frames.append(tf.convert_to_tensor(frame))

    tf_frames = tf.convert_to_tensor(tf_frames)
    results = model.predict(tf_frames, verbose=0)

    return np.average(results)

# %% [markdown]
# ### Temporal Stream

# %%
resnet50v2 = tf.keras.applications.ResNet50V2(include_top=False)

# %%
inputs = keras.layers.Input((224, 224, 3))
x = tf.keras.applications.resnet_v2.preprocess_input(inputs)
x = resnet50v2(x)
x = Flatten()(x)
x = Dropout(0.8)(x)
x = Dense(100, activation=LeakyReLU())(x)
x = Dropout(0.8)(x)
out = Dense(1, activation='sigmoid')(x)

temporal_model = keras.Model(inputs, out, name="temporal_stream")

# %%
temporal_model.compile(optimizer = Adam(learning_rate = 0.00001), 
              loss = keras.losses.BinaryCrossentropy(), 
              metrics = [keras.metrics.BinaryAccuracy(), 
                         keras.metrics.Precision(), 
                         keras.metrics.Recall(),
                         keras.metrics.AUC(),
                         keras.metrics.FalseNegatives(),
                         keras.metrics.FalsePositives(),
                         keras.metrics.TrueNegatives(),
                         keras.metrics.TruePositives()],
             )

# %%
temporal_model.load_weights(TEMPORAL_PATH + 'checkpoint_final_sig').expect_partial()

# %%
def get_residuals(fp, num_segments):
    vid_container = av.open(fp)
    vid_stream = vid_container.streams.video[0]
    frame_count = vid_stream.frames

    segment_dividers = get_segment_dividers(frame_count, num_segments)

    vid_container.seek(0)
    frame_list = [frame.to_image() for frame in vid_container.decode()]

    residuals = []
    start_index = 0
    for sd in segment_dividers:
        residuals.append(frame_list[start_index:sd])
        start_index = sd
    
    residuals.append(frame_list[start_index:])

    vid_container.close()
    return residuals

# %%
def calculate_temporal_score(model, fp):
    residuals = get_residuals(fp, num_segments=3)

    results = []
    for residual_set in residuals:
        tf_frames = []

        for frame in residual_set:
            tf_frames.append(img_to_array(tf.image.resize(frame, size = [224, 224])))

        tf_frames = np.asarray(tf_frames)
        result = model.predict(tf_frames, verbose=0)
        results.append(np.average(result))

    max_val = np.max(results)
    min_val = np.min(results)

    return max_val if 1 - max_val < min_val else min_val

# %% [markdown]
# ## Score Aggregation

# %%
with open(SVM_PATH + 'svm.pkl', 'rb') as f:
    svm_model = pickle.load(f)

# %%
def get_class(score):
    return 'Fake' if score < 0.5 else 'Real'

# %%
def weighted_avg_score(m, s, t, w1=0.33, w2=0.33, w3=0.33):
    avg = (m * w1) + (s * w2) + (t * w3)

    return get_class(avg) 

# %%
def majority_voting_score(m, s, t):
    fake_pred_count = sum(1 if score < 0.5 else 0 for score in [m, s, t])

    score = max([m, s, t]) if fake_pred_count < 2 else min([m, s, t])

    return get_class(score)

# %%
def svm_classifer_score(m, s, t):
    prediction = svm_model.predict_proba([[m, s, t]])

    return get_class(prediction[0][1])

# %% [markdown]
# # Output Logging

# %%
def generate_result_report(results):
    table = PrettyTable()
    table.field_names = ["File Name", "Weighted Avg Result", "Majority Voting Result", "SVM Classfier Result"]

    table.add_rows(results)
    print(table)


# %% [markdown]
# # Execution

# %%
def process_video(fp):
    filename = os.path.split(fp)[1]

    if not os.path.exists(INTERMEDIARY_DIR + IFRAME_DIR + filename):
        extract_iframes(fp)

    if not os.path.exists(INTERMEDIARY_DIR + FACECROP_DIR + filename):
        save_cropped_faces_to_video(INTERMEDIARY_DIR + IFRAME_DIR + filename, -1)

    if not os.path.exists(INTERMEDIARY_DIR + RESIDUAL_DIR + filename):
        extract_residuals(fp)

    m_score = calculate_mesonet_score(mesonet_model_dense_prune, INTERMEDIARY_DIR + FACECROP_DIR + filename)
    s_score = calculate_srm_score(SRM_Model, INTERMEDIARY_DIR + FACECROP_DIR + filename)
    t_score = calculate_temporal_score(temporal_model, INTERMEDIARY_DIR + RESIDUAL_DIR + filename)

    wa_result = weighted_avg_score(m_score, s_score, t_score)
    mv_result = majority_voting_score(m_score, s_score, t_score)
    svm_result = svm_classifer_score(m_score, s_score, t_score)

    return (filename, wa_result, mv_result, svm_result)

# %%
if len(sys.argv) > 1:
    input_files = sys.argv
    for file in input_files:
        assert os.path.exists(file)

else:
    input_files = os.listdir(INPUT_DIR)

print(f'{len(input_files)} files given as input!')

results = []

for video in input_files:
    result = process_video(INPUT_DIR + video)
    results.append(result)

    print(f'Video processed: {video}')

generate_result_report(results)

# %%



