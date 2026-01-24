from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Dropout, Concatenate, Bidirectional

def build_ensemble_model(input_shape, forecast_steps):
    input_layer = Input(shape=input_shape)
    
    # Dega 1: Bidirectional LSTM
    lstm_branch = Bidirectional(LSTM(128, return_sequences=True))(input_layer)
    lstm_branch = Dropout(0.2)(lstm_branch)
    lstm_branch = Bidirectional(LSTM(64))(lstm_branch)
    
    # Dega 2: Bidirectional GRU
    gru_branch = Bidirectional(GRU(128, return_sequences=True))(input_layer)
    gru_branch = Dropout(0.2)(gru_branch)
    gru_branch = Bidirectional(GRU(64))(gru_branch)
    
    # Bashkimi i degëve (Concatenate)
    combined = Concatenate()([lstm_branch, gru_branch])
    
    # Shtresat finale për parashikimin
    dense = Dense(128, activation='relu')(combined)
    dense = Dropout(0.2)(dense)
    output = Dense(forecast_steps)(dense)
    
    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer='adam', loss='mae')
    return model