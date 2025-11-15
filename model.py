import yfinance as yf
import pandas as pd
import numpy as np
import pywt # For Wavelets
from hurst import compute_Hc # For Hurst Exponent
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# --- Step 1: Get Data (NIFTY 50) ---
# Note: yfinance free data for ^NSEI may not go back to 1990.
# It starts around 1996, which still covers all 3 major crashes.
ticker = yf.Ticker("^NSEI")
data = ticker.history(start="1996-01-01", end="2024-12-31")
data = data[['Close']] # We only need the 'Close' price for this
data.dropna(inplace=True)

# --- Step 2: Feature Engineering (Sharma-Inspired) ---
print("Starting Feature Engineering...")

# a) Baseline Features
data['Returns'] = data['Close'].pct_change()
data['Volatility'] = data['Returns'].rolling(window=21).std() # 21-day (1-month) rolling volatility

# b) Multi-scale Analysis (Wavelet Decomposition)
# We must create 4 separate functions because rolling().apply()
# can only return a single value (not a tuple).

def get_wavelet_cA_mean(x):
    coeffs = pywt.wavedec(x, 'db4', level=1)
    return coeffs[0].mean() # [0] is cA

def get_wavelet_cA_std(x):
    coeffs = pywt.wavedec(x, 'db4', level=1)
    return coeffs[0].std() # [0] is cA

def get_wavelet_cD_mean(x):
    coeffs = pywt.wavedec(x, 'db4', level=1)
    return coeffs[1].mean() # [1] is cD

def get_wavelet_cD_std(x):
    coeffs = pywt.wavedec(x, 'db4', level=1)
    return coeffs[1].std() # [1] is cD

# Apply each function one by one.
# We use raw=True for a significant speed boost.
print("Applying Wavelet cA mean...")
data['Wavelet_cA_mean'] = data['Close'].rolling(window=30).apply(get_wavelet_cA_mean, raw=True)
print("Applying Wavelet cA std...")
data['Wavelet_cA_std'] = data['Close'].rolling(window=30).apply(get_wavelet_cA_std, raw=True)
print("Applying Wavelet cD mean...")
data['Wavelet_cD_mean'] = data['Close'].rolling(window=30).apply(get_wavelet_cD_mean, raw=True)
print("Applying Wavelet cD std...")
data['Wavelet_cD_std'] = data['Close'].rolling(window=30).apply(get_wavelet_cD_std, raw=True)
# c) Market Complexity (Hurst Exponent)
# Calculate Hurst over a longer window (e.g., 100 days)
def get_hurst(x):
    # compute_Hc returns H, c, data
    H, _, _ = compute_Hc(x, simplified=True)
    return H

data['Hurst'] = data['Close'].rolling(window=100).apply(get_hurst, raw=True)

# Clean up NaNs created by rolling windows
data.dropna(inplace=True)

print("Features Created. Data shape:", data.shape)
print(data.tail())

# --- Step 3: Create the Target Variable (y) ---
print("Creating Target Variable (y)...")

# Define our prediction window and crash threshold
look_forward_days = 30
crash_threshold = -0.10 # A 1% drop

# Initialize 'y' column with 0 (Normal)
data['y'] = 0

# This loop is slow, but explicit.
# For each day, it looks into the future to see if a crash occurs.
for i in range(len(data) - look_forward_days):
    # Get the price window to check
    future_window = data['Close'].iloc[i+1 : i + 1 + look_forward_days]
    # Get today's price
    current_price = data['Close'].iloc[i]
    
    # If any price in the future window is <= 80% of today's price
    if (future_window / current_price - 1).min() <= crash_threshold:
        data['y'].iloc[i] = 1 # Label today as '1' (Crash Imminent)

print("Target Variable Created.")
print(f"Class Distribution:\n{data['y'].value_counts(normalize=True)}")

# --- Step 4: Prepare Data for LSTM ---

# a) Define X (features) and y (target)
features = ['Returns', 'Volatility', 'Wavelet_cA_mean', 'Wavelet_cA_std', 'Wavelet_cD_mean', 'Wavelet_cD_std', 'Hurst']
X = data[features]
y = data['y']

# b) Scale Features
# LSTMs are sensitive to scale. We scale from 0 to 1.
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)

# c) Create Sequences
# We'll use the last 60 days of features to predict the label for the *next* day.
TIME_STEPS = 60 # This is your lookback window

X_sequences = []
y_sequences = []

for i in range(TIME_STEPS, len(X_scaled)):
    X_sequences.append(X_scaled[i - TIME_STEPS : i, :])
    y_sequences.append(y.iloc[i])

X_sequences = np.array(X_sequences)
y_sequences = np.array(y_sequences)

# d) Split into Train and Test sets
# IMPORTANT: Do NOT shuffle time series data.
split_point = int(len(X_sequences) * 0.7)
X_train, X_test = X_sequences[:split_point], X_sequences[split_point:]
y_train, y_test = y_sequences[:split_point], y_sequences[split_point:]

print(f"X_train shape: {X_train.shape}") # (n_samples, 60, 7)
print(f"y_train shape: {y_train.shape}") # (n_samples,)


# --- Step 5: Build and Compile the LSTM Architecture ---
print("Building LSTM Model...")

model = Sequential()

# Layer 1: LSTM layer with 50 units. 
# 'return_sequences=True' is needed to pass the sequence to the next LSTM layer.
# 'input_shape' is (timesteps, num_features)
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2)) # Dropout for regularization

# Layer 2: Another LSTM layer
model.add(LSTM(units=50, return_sequences=False)) # 'False' because we are feeding to a Dense layer next
model.add(Dropout(0.2))

# Layer 3: A standard Dense layer
model.add(Dense(units=25, activation='relu'))

# Layer 4: The Output Layer
# 'units=1' for a binary outcome
# 'activation='sigmoid'' to squash the output between 0 and 1 (a probability)
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
# 'binary_crossentropy' is the standard loss function for a 2-class problem.
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# --- Step 6: Train the Model ---
print("Training Model...")

# !! CRITICAL: Handle Imbalanced Data !!
# Your data is 'imbalanced' (mostly 0s). We use 'class_weight' to
# tell the model to "pay more attention" to the rare '1' class.
# Manually force the model to care about '1's
# This says "Missing a '1' is 50x worse than missing a '0'"
class_weights = {0: 1.0, 1: 23.75}

# --- Experimentation Notes for your thesis ---
# If 50.0 gives you too many '1's (high recall, low precision),
# try a smaller number like 25.0 or 30.0.
# If it still gives you 0.02, try a bigger number like 75.0 or 100.0

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping],
    class_weight=class_weights, # <--- VERY IMPORTANT
    shuffle=False # Do not shuffle time series data
)

print("Model Training Complete.")

# --- Step 7: Evaluate the Model ---
# For imbalanced data, 'accuracy' is useless.
# We need to look at Precision and Recall for the '1' class.
from sklearn.metrics import classification_report

# Get predictions (as probabilities)
y_pred_proba = model.predict(X_test)
# Convert probabilities to classes (0 or 1)
y_pred = (y_pred_proba > 0.5025).astype(int)

# --- ADD THESE TWO LINES FOR DIAGNOSTICS ---
print(f"Test Set Labels: {np.unique(y_test, return_counts=True)}")
print(f"Prediction Labels: {np.unique(y_pred, return_counts=True)}")
# --- END OF ADDITION ---

print("\n--- Classification Report ---")
# This report is the TRUE measure of your model's performance.
# Look for a good 'precision' and 'recall' for class '1'.

print(classification_report(y_test, y_pred, target_names=['0 (Normal)', '1 (Crash Imminent)']))