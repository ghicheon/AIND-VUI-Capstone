from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM)

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))

    simp_rnn = GRU(units, activation=activation, return_sequences=True, implementation=2, name='rnn')(input_data)

    bn_rnn =  BatchNormalization(name='bn_gru')(simp_rnn)

    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)

    y_pred = Activation('softmax', name='softmax')(time_dense)

    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    input_data = Input(name='the_input', shape=(None, input_dim))

    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)

    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)

    simp_rnn = SimpleRNN(units, activation='relu', return_sequences=True, implementation=2, name='rnn')(bn_cnn)

    bn_rnn =  BatchNormalization(name='bn_rnn')(simp_rnn)

    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)

    y_pred = Activation('softmax', name='softmax')(time_dense)


    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    input_data = Input(name='the_input', shape=(None, input_dim))

    layerB = input_data
    for _ in range(recur_layers):
        layerA = GRU(units, activation='relu', return_sequences=True, implementation=2)(layerB)
        layerB =  BatchNormalization()(layerA)

    time_dense = TimeDistributed(Dense(output_dim))( layerB)

    y_pred = Activation('softmax', name='softmax')(time_dense)


    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    input_data = Input(name='the_input', shape=(None, input_dim))

    bidir_rnn =   Bidirectional( GRU(units, activation='relu', return_sequences=True, implementation=2, name='bi_dir'))(input_data)

    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn)

    y_pred = Activation('softmax', name='softmax')(time_dense)


    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def model_number5():
    input_dim = 161
    units=300
    activation='relu'
    output_dim=29
    input_data = Input(name='the_input', shape=(None, input_dim))

    input_data = Input(name='the_input', shape=(None, input_dim))

    bidir_rnn1 =   Bidirectional( GRU(units, activation='relu', return_sequences=True, implementation=2, name='bi_dir1'))(input_data)
    bidir_rnn2 =   Bidirectional( GRU(units, activation='relu', return_sequences=True, implementation=2, name='bi_dir2'))(bidir_rnn1)

    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn2)

    y_pred = Activation('softmax', name='softmax')(time_dense)


    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())

    return model



def model_number6():
    input_dim = 13
    units=300
    activation='relu'
    output_dim=29

    input_data = Input(name='the_input', shape=(None, input_dim))

    L1 =   Bidirectional( GRU(units, activation='relu', return_sequences=True, implementation=2, dropout=0.2, recurrent_dropout=0.2, name='L1'))(input_data)
    L2 =   Bidirectional( GRU(units, activation='relu', return_sequences=True, implementation=2, dropout=0.2, recurrent_dropout=0.2, name='L2'))(L1)

    time_dense = TimeDistributed(Dense(output_dim))(L2)

    y_pred = Activation('softmax', name='softmax')(time_dense)


    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())

    return model

def final_model():
    return model_number6()


#
# I wish I could train it fast. But it took so long on aws.. 
# I'm sure It will be better than others.
#
# => Finally, I ran it on aws p3 instance!
def final_model2():
    input_dim = 13
    units=500
    activation='relu'
    output_dim=29

    input_data = Input(name='the_input', shape=(None, input_dim))

    L1 =   Bidirectional( GRU(units, activation='relu', return_sequences=True, implementation=2, dropout=0.2, recurrent_dropout=0.2, name='L1'))(input_data)
    L2 =   Bidirectional( GRU(units, activation='relu', return_sequences=True, implementation=2, dropout=0.2, recurrent_dropout=0.2, name='L2'))(L1)
    L3 =   Bidirectional( GRU(units, activation='relu', return_sequences=True, implementation=2, dropout=0.2, recurrent_dropout=0.2, name='L2'))(L2)

    time_dense = TimeDistributed(Dense(output_dim))(L3)

    y_pred = Activation('softmax', name='softmax')(time_dense)


    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())

    return model
