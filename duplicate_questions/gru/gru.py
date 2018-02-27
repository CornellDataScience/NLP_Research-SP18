import keras

VECTOR_DIM_IN = 0       # size of input vector
VECTOR_DIM_OUT = 0      # size of output vector
TIMESTEPS = 0           # length of unrolled network

# input tensor is (batch_size, timesteps, input_dim)
model = keras.models.Sequential()
model.add(keras.layers.GRU(VECTOR_DIM_OUT,
                           input_shape = (TIMESTEPS, VECTOR_DIM_IN)
                           activation='tanh',
                           dropout=0.0, 
                           recurrent_dropout=0.0))
