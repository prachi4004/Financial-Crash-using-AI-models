import yfinance as yf
import pandas as pd
import numpy as np
import pywt # For Wavelets
from hurst import compute_Hc # For Hurst Exponent
# from sklearn.preprocessing import MinMaxScaler  <-- REMOVE THIS
from sklearn.preprocessing import StandardScaler  # <-- ADD THIS
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from nolds import sampen, lyap_r
from tensorflow.keras.optimizers import Adam

# --- Step 1: Get Data (from your CSV file) ---
print("Loading data from 'NIFTY 50.csv'...")
try:
    data = pd.read_csv('NIFTY 50.csv')
    data['Date'] = pd.to_datetime(data['Date'], format='mixed')
    data.set_index('Date', inplace=True)
    
    # Create the two main columns we need
    data_close = data[['Close']]
    pe_data_subset = data[['P/E']].rename(columns={'P/E': 'PE_Ratio'})
    
    # Our main 'data' dataframe is now the 'Close' prices
    data = data_close
    data.dropna(inplace=True) # Drop any missing 'Close' values

except FileNotFoundError:
    print("\n!!! ERROR: 'NIFTY 50.csv' not found. !!!")
    exit()
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit()

print(f"Successfully loaded data. Starts: {data.index.min()}, Ends: {data.index.max()}")

# --- Step 2: Feature Engineering ---
print("Starting Feature Engineering...")

# a) Baseline Features
data['Returns'] = data['Close'].pct_change()
data['Volatility'] = data['Returns'].rolling(window=21).std()

# b) Multi-scale Analysis (Wavelet Decomposition)
print("Applying Wavelet Features...")
def get_wavelet_cA_mean(x):
    try:
        coeffs = pywt.wavedec(x, 'db4', level=1)
        return coeffs[0].mean()
    except ValueError: return np.nan
def get_wavelet_cA_std(x):
    try:
        coeffs = pywt.wavedec(x, 'db4', level=1)
        return coeffs[0].std()
    except ValueError: return np.nan
def get_wavelet_cD_mean(x):
    try:
        coeffs = pywt.wavedec(x, 'db4', level=1)
        return coeffs[1].mean()
    except ValueError: return np.nan
def get_wavelet_cD_std(x):
    try:
        coeffs = pywt.wavedec(x, 'db4', level=1)
        return coeffs[1].std()
    except ValueError: return np.nan

data['Wavelet_cA_mean'] = data['Close'].rolling(window=30).apply(get_wavelet_cA_mean, raw=True)
data['Wavelet_cA_std'] = data['Close'].rolling(window=30).apply(get_wavelet_cA_std, raw=True)
data['Wavelet_cD_mean'] = data['Close'].rolling(window=30).apply(get_wavelet_cD_mean, raw=True)
data['Wavelet_cD_std'] = data['Close'].rolling(window=30).apply(get_wavelet_cD_std, raw=True)

# c) Market Complexity (Hurst Exponent)
print("Applying Hurst Exponent...")
def get_hurst(x):
    try:
        H, _, _ = compute_Hc(x, simplified=True)
        return H
    except ValueError: return np.nan
data['Hurst'] = data['Close'].rolling(window=100).apply(get_hurst, raw=True)


# d) Market Complexity (Sample Entropy)
print("Applying Sample Entropy (SampEn)...")
def get_sampen(x):
    try:
        return sampen(x)
    except ValueError: return np.nan
data['SampEn'] = data['Close'].rolling(window=100).apply(get_sampen, raw=True)

# e) Market Complexity (Largest Lyapunov Exponent)
print("Applying Largest Lyapunov Exponent (LLE)...")
def get_lle(x):
    try:
        # emb_dim=5 is a good starting point
        return lyap_r(x, emb_dim=5)
    # Add OverflowError for robustness
    except (ValueError, np.linalg.LinAlgError, OverflowError): return np.nan
data['LLE'] = data['Close'].rolling(window=100).apply(get_lle, raw=True)

# --- Step 3: Load, Merge, and Align P/E Data ---
print("Merging P/E Data...")

# Use 'left' merge to keep all 'Close' data and add 'P/E'
data = pd.merge(data, pe_data_subset, left_index=True, right_index=True, how='left')
data['PE_Ratio'] = data['PE_Ratio'].bfill() # Back-fill any missing P/E values

# --- NEW: FEATURE ENGINEERING (P/E BANDS) ---
print("Creating Valuation Regime Feature...")

#
def get_valuation_regime(pe):
    if pe < 13: return 0 # "No Risk"
    elif pe < 16: return 1 # "Low Risk"
    elif pe < 22: return 2 # "Moderate Risk"
    elif pe < 27: return 3 # "High Risk"
    elif pe >= 27: return 4 # "Very High Risk"
    else: return np.nan

data['Valuation_Regime_Category'] = data['PE_Ratio'].apply(get_valuation_regime)
# --- NEW: ONE-HOT ENCODING ---
data = pd.get_dummies(data, columns=['Valuation_Regime_Category'], prefix='Regime')
# --- END OF NEW FEATURES ---

# --- Clean up NaNs (NEW METHOD) ---
# We back-fill the NaNs created by the rolling windows.
# This is critical to keep the 2000 crash data.
print("Cleaning NaNs using back-fill (bfill)...")

# Define all feature columns that have rolling windows
feature_columns_to_bfill = [
    'Returns', 'Volatility', 
    'Wavelet_cA_mean', 'Wavelet_cA_std', 'Wavelet_cD_mean', 'Wavelet_cD_std', 
    'Hurst',
    'SampEn',  # <-- NEW
    'LLE'      # <-- NEW
]

# Back-fill only these columns
for col in feature_columns_to_bfill:
    if col in data.columns:
        data[col] = data[col].bfill()

# Now, drop any other NaNs (e.g., from the first row's 'Returns')
data.dropna(inplace=True) 
# --- End of New Section ---

# --- Step 4: Create the Target Variable (y) ---
print("Creating Target Variable (y)...")
look_forward_days = 30
crash_threshold = -0.10 # 10% drop

data['y'] = 0
for i in range(len(data) - look_forward_days):
    future_window = data['Close'].iloc[i+1 : i + 1 + look_forward_days]
    current_price = data['Close'].iloc[i]
    
    if (future_window / current_price - 1).min() <= crash_threshold:
        data.loc[data.index[i], 'y'] = 1

print(f"Class Distribution:\n{data['y'].value_counts(normalize=True)}")

# --- Step 5: Prepare Data for LSTM ---

# --- NEW: Define Feature List Programmatically ---
# This automatically finds the new 'Regime_0', 'Regime_1', etc. columns
base_features = [
    'Returns', 'Volatility', 
    'Wavelet_cA_mean', 'Wavelet_cA_std', 'Wavelet_cD_mean', 'Wavelet_cD_std', 
    'Hurst', 
    'SampEn',  # <-- ADD THIS
    'LLE',     # <-- ADD THIS
    'PE_Ratio'
]
regime_features = [col for col in data.columns if col.startswith('Regime_')]
features = base_features + regime_features

print(f"\n--- Model using {len(features)} features ---")
print(features)
# --- END NEW ---

X = data[features]
y = data['y']

# b) Scale Features
# scaler = MinMaxScaler(feature_range=(0, 1))  <-- REMOVE THIS
scaler = StandardScaler()  # <-- USE THE NEW SCALER
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

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# --- Calculate Class Weights (Just before Step 6) ---

# We need to check the distribution of the y_train array
counts = np.bincount(y_train)
weight_for_0 = 1.0
weight_for_1 = counts[0] / counts[1]               #the actual imbalance is 2715 / 173 = 15.6

print(f"\n--- Class Imbalance ---")
print(f"Normal (0) count in training: {counts[0]}")
print(f"Crash (1) count in training: {counts[1]}")
print(f"Calculated weight for '1': {weight_for_1:.2f}")

weight_for_1_tuned = weight_for_1 / 3  

class_weights = {0: weight_for_0, 1: weight_for_1_tuned}
print(f"Using TUNED class weight for '1': {weight_for_1_tuned:.2f}\n")

# --- Step 6: Build and Compile the LSTM Architecture ---
print("Building LSTM Model...")

model = Sequential()
# A single, powerful LSTM layer is often a better baseline
model.add(LSTM(units=100, return_sequences=False, 
               input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))

# A dense layer to learn combinations
model.add(Dense(units=50, activation='relu'))
model.add(Dropout(0.2))

# The final output
model.add(Dense(units=1, activation='sigmoid'))
# Define a new, more stable optimizer with a lower learning rate
# The default is 0.001. We are making it 10x slower.
stable_optimizer = Adam(learning_rate=0.0001)

model.compile(optimizer=stable_optimizer, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# --- Step 7: Train the Model ---
print("Training Model...")

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping],
    class_weight=class_weights,  # <-- ADD THIS LINE BACK IN
    shuffle=False
)

print("Model Training Complete.")

# --- Save the Trained Model ---
print("Saving model to 'crash_predictor_model.keras'...")
model.save('crash_predictor_model.keras')
print("Model saved.")

# --- Step 8: Evaluate the Model ---
print("Evaluating Model...")

# Get the raw predicted probabilities (e.g., 0.05, 0.25, 0.60)
y_pred_proba = model.predict(X_test)

# --- NEW: Loop through thresholds to find the "sweet spot" ---
print("\n--- Finding Best Prediction Threshold ---")

# You can adjust this range
for threshold in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
    print(f"\n--- Classification Report for threshold = {threshold} ---")
    
    # Apply the threshold manually to the probabilities
    y_pred = (y_pred_proba > threshold).astype(int)
    
    print(classification_report(
        y_test, 
        y_pred, 
        target_names=['0 (Normal)', '1 (Crash Imminent)'], 
        zero_division=0
    ))
    
    print("--- Confusion Matrix ---")
    print(confusion_matrix(y_test, y_pred))


# --- Step 9: Visualize Results (Full History Plot) ---
print("\nGenerating FULL history (Train + Test) plot...")

# --- 1. Get Predictions for BOTH Train and Test Sets ---
y_pred_proba_train = model.predict(X_train)
y_pred_proba_test = y_pred_proba # Already have this from Step 8

# Combine them to create a full probability history
full_probabilities = np.concatenate((y_pred_proba_train, y_pred_proba_test))

# --- 2. Get Full Date, Price, and Label Data ---
full_dates = data.index[TIME_STEPS:]
full_close_prices = data['Close'].iloc[TIME_STEPS:]
full_actual_labels = y_sequences # This is the y_train + y_test combined

# Find the date where the test set begins
split_date = data.index[split_point + TIME_STEPS]

# Find all actual crash events in the *entire* dataset
actual_crash_indices = np.where(full_actual_labels == 1)[0]
actual_crash_dates = full_dates[actual_crash_indices]
actual_crash_prices = full_close_prices.iloc[actual_crash_indices]

# --- 3. Create the Full Plot ---
fig, ax1 = plt.subplots(figsize=(15, 7))

# Plot NIFTY 50 Close Price for the full period
ax1.plot(full_dates, full_close_prices, label='NIFTY 50 Price', color='blue', alpha=0.6)
ax1.set_xlabel('Date')
ax1.set_ylabel('NIFTY 50 Close Price', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_title('Model Crash Probability vs. NIFTY 50 Price (Full History)')

# Plot ALL actual crash events
ax1.scatter(actual_crash_dates, actual_crash_prices, color='red', s=100, 
            label='Actual Crash Event (y=1)', marker='v', zorder=5)

# --- 4. Add Train/Test Split Line (CRITICAL) ---
# This line clearly separates data the model trained on vs. data it predicted
ax1.axvline(x=split_date, color='black', linestyle='--', linewidth=2, 
            label=f'Train/Test Split ({split_date.date()})')

# --- 5. Plot Probabilities (Second Y-Axis) ---
ax2 = ax1.twinx()

# Plot the combined train + test probabilities
ax2.plot(full_dates, full_probabilities, label='Model Probability', color='orange', alpha=0.7)
ax2.set_ylabel('Model Crash Probability', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')
ax2.set_ylim(0, 1)

# --!! Change this value based on your findings from Step 8 !! --
CHOSEN_THRESHOLD = 0.45
ax2.axhline(y=CHOSEN_THRESHOLD, color='red', linestyle='--', label=f'Chosen Threshold ({CHOSEN_THRESHOLD})')

# Combine legends
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper left')

fig.tight_layout()
plt.show()


print("Script Complete.")
