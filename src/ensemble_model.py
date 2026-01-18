import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Dropout, Concatenate, BatchNormalization

def build_ensemble_model(input_shape, output_steps=24):
    # Hyrja: (168 orë histori, numri i tipareve)
    input_layer = Input(shape=input_shape)
    
    # Dega 1: LSTM (fokus te sekuencat e gjata)
    lstm_branch = LSTM(128, return_sequences=True)(input_layer)
    lstm_branch = Dropout(0.2)(lstm_branch)
    lstm_branch = LSTM(64)(lstm_branch)
    
    # Dega 2: GRU (fokus te dinamika e shpejtë)
    gru_branch = GRU(128, return_sequences=True)(input_layer)
    gru_branch = Dropout(0.2)(gru_branch)
    gru_branch = GRU(64)(gru_branch)
    
    # Bashkimi i njohurive nga të dyja degët (Ensemble)
    combined = Concatenate()([lstm_branch, gru_branch])
    
    # Shtresat Dense për interpretimin e bashkimit
    dense = Dense(128, activation='relu')(combined)
    dense = BatchNormalization()(dense)
    dense = Dropout(0.2)(dense)
    
    # Outputi: 24 orë rresht
    output_layer = Dense(output_steps)(dense)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mae', metrics=['mse'])
    
    return model