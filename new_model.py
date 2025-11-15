import yfinance as yf
import pandas as pd
import numpy as np
import pywt # For Wavelets
from hurst import compute_Hc # For Hurst Exponent
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report

# --- Step 1: Get Data (NIFTY 50) ---
# We still get data from 1996 to use as a "warm-up" period
# for our 100-day rolling features.
ticker = yf.Ticker("^NSEI")
data = ticker.history(start="1996-01-01", end="2024-12-31")
data = data[['Close']]
data.dropna(inplace=True)

data.index = data.index.tz_localize(None)

# --- Step 2: Feature Engineering (Sharma-Inspired) ---
print("Starting Feature Engineering...")

# a) Baseline Features
data['Returns'] = data['Close'].pct_change()
data['Volatility'] = data['Returns'].rolling(window=21).std()

# b) Multi-scale Analysis (Wavelet Decomposition)
print("Applying Wavelet Features...")
def get_wavelet_cA_mean(x):
    coeffs = pywt.wavedec(x, 'db4', level=1)
    return coeffs[0].mean()

def get_wavelet_cA_std(x):
    coeffs = pywt.wavedec(x, 'db4', level=1)
    return coeffs[0].std()

def get_wavelet_cD_mean(x):
    coeffs = pywt.wavedec(x, 'db4', level=1)
    return coeffs[1].mean()

def get_wavelet_cD_std(x):
    coeffs = pywt.wavedec(x, 'db4', level=1)
    return coeffs[1].std()

data['Wavelet_cA_mean'] = data['Close'].rolling(window=30).apply(get_wavelet_cA_mean, raw=True)
data['Wavelet_cA_std'] = data['Close'].rolling(window=30).apply(get_wavelet_cA_std, raw=True)
data['Wavelet_cD_mean'] = data['Close'].rolling(window=30).apply(get_wavelet_cD_mean, raw=True)
data['Wavelet_cD_std'] = data['Close'].rolling(window=30).apply(get_wavelet_cD_std, raw=True)

# c) Market Complexity (Hurst Exponent)
print("Applying Hurst Exponent...")
def get_hurst(x):
    H, _, _ = compute_Hc(x, simplified=True)
    return H
data['Hurst'] = data['Close'].rolling(window=100).apply(get_hurst, raw=True)


# --- NEW: Load, Merge, and Align P/E Data ---
print("Loading and Merging P/E Data...")
try:
    # 1. Load the P/E data from 'NIFTY 50.csv'
    pe_data = pd.read_csv('NIFTY 50.csv')
    
    # 2. Convert the 'Date' column to datetime objects
    # We use format='mixed' for robustness
    pe_data['Date'] = pd.to_datetime(pe_data['Date'], format='mixed')
    
    # 3. Set the 'Date' as the index
    pe_data.set_index('Date', inplace=True)
    
    # 4. Select P/E and rename for clarity
    pe_data_subset = pe_data[['P/E']].rename(columns={'P/E': 'PE_Ratio'})

    # 5. *** CRITICAL CHANGE: Use an 'inner' merge ***
    # This aligns both datasets to the common dates (2000-2024)
    # and drops the 1996-1999 "warm-up" data automatically.
    data = pd.merge(data, pe_data_subset, left_index=True, right_index=True, how='inner')

    # 6. ADD THE PAPER'S RISK BANDS AS A FEATURE
    # [cite: 1000-1014, 1066-1070 from ssrn-5422735.pdf]
    print("Creating Valuation Regime Feature...")
    def get_valuation_regime(pe):
        if pe < 13: return 0 # "No Risk"
        elif pe < 16: return 1 # "Low Risk"
        elif pe < 22: return 2 # "Moderate Risk"
        elif pe < 27: return 3 # "High Risk"
        elif pe >= 27: return 4 # "Very High Risk"
        else: return np.nan
    
    data['Valuation_Regime'] = data['PE_Ratio'].apply(get_valuation_regime)

except FileNotFoundError:
    print("\n!!! ERROR: 'NIFTY 50.csv' not found. !!!")
    print("!!! Please make sure the file is in the same folder as the script. !!!")
    exit() # Exit the script if the file isn't found
# --- END OF NEW FEATURES ---


# Clean up NaNs
# This will now only drop the first 100 rows of the 2000 data
# (which are NaN from the Hurst rolling window)
data.dropna(inplace=True)

print("Features Created. Data shape:", data.shape)
print("Data analysis starts from:", data.index[0])
print(data.tail())

# --- Step 3: Create the Target Variable (y) ---
print("Creating Target Variable (y)...")
look_forward_days = 30
crash_threshold = -0.10 # 10% drop ("Market Correction")

data['y'] = 0
for i in range(len(data) - look_forward_days):
    future_window = data['Close'].iloc[i+1 : i + 1 + look_forward_days]
    current_price = data['Close'].iloc[i]
    
    if (future_window / current_price - 1).min() <= crash_threshold:
        # Use .loc to avoid SettingWithCopyWarning
        data.loc[data.index[i], 'y'] = 1

print(f"Class Distribution:\n{data['y'].value_counts(normalize=True)}")

# --- Step 4: Prepare Data for LSTM ---

# a) Define X (features) and y (target)
# *** THIS IS YOUR NEW, UPGRADED FEATURE SET ***
features = [
    'Returns', 'Volatility', 
    'Wavelet_cA_mean', 'Wavelet_cA_std', 'Wavelet_cD_mean', 'Wavelet_cD_std', 
    'Hurst', 
    'PE_Ratio', # <-- NEW: Valuation metric
    'Valuation_Regime' # <-- NEW: Paper's risk bands
]
X = data[features]
y = data['y']

# b) Scale Features
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)

# c) Create Sequences
TIME_STEPS = 60
X_sequences, y_sequences = [], []
for i in range(TIME_STEPS, len(X_scaled)):
    X_sequences.append(X_scaled[i - TIME_STEPS : i, :])
    y_sequences.append(y.iloc[i])
X_sequences, y_sequences = np.array(X_sequences), np.array(y_sequences)

# d) Split into Train and Test sets
split_point = int(len(X_sequences) * 0.7)
X_train, X_test = X_sequences[:split_point], X_sequences[split_point:]
y_train, y_test = y_sequences[:split_point], y_sequences[split_point:]

print(f"X_train shape: {X_train.shape}") # (n_samples, 60, 9) <-- 9 features now
print(f"y_train shape: {y_train.shape}")


# --- Step 5: Build and Compile the LSTM Architecture ---
print("Building LSTM Model...")

model = Sequential()
# Layer 1
model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))

# Layer 2 (NEW) - Must also have return_sequences=True
model.add(LSTM(units=100, return_sequences=True)) # <-- NEW LAYER
model.add(Dropout(0.2))

# Layer 3 (Old Layer 2) - Now has return_sequences=False
model.add(LSTM(units=100, return_sequences=False)) # <-- OLD LAYER 2
model.add(Dropout(0.2))

# Layer 4 (Dense)
model.add(Dense(units=50, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# --- Step 6: Train the Model ---
print("Training Model...")

# !! START YOUR TUNING OVER !!
# The old value of 23.75 is INVALID.
# Start your binary search again. Try 10, 20, 30...
class_weights = {0: 1.0, 1: 78.125} # <-- START TUNING FROM HERE
print(f"Using class weights: {class_weights}")

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping],
    class_weight=class_weights,
    shuffle=False
)

print("Model Training Complete.")

# --- Step 7: Evaluate the Model ---
print("Evaluating Model...")

y_pred_proba = model.predict(X_test)

# !! START YOUR TUNING OVER !!
# The old value is INVALID. Start back at 0.5.
prediction_threshold = 0.5 # <-- START TUNING FROM HERE
y_pred = (y_pred_proba > prediction_threshold).astype(int)

print(f"Test Set Labels: {np.unique(y_test, return_counts=True)}")
print(f"Prediction Labels: {np.unique(y_pred, return_counts=True)}")

print("\n--- Classification Report ---")
# Added zero_division=0 to prevent warnings if precision is 0.
print(classification_report(y_test, y_pred, target_names=['0 (Normal)', '1 (Crash Imminent)'], zero_division=0))