import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

# Carica i dati
stocks = pd.read_csv('stocks.csv', parse_dates=['Date'], index_col='Date')
crypto = pd.read_csv('crypto.csv', parse_dates=['Date'], index_col='Date')
data = pd.concat([stocks, crypto], axis=1, join='inner')
returns = data.pct_change().dropna()

# Impostazioni per l'animazione
window_size = 90  # Finestra mobile di 3 mesi
step = 5  # Aggiorniamo ogni 5 giorni per velocità

# Prepara la figura con 3 subplot
plt.figure(figsize=(12, 8))

# Subplot 1: Componente Stock
ax1 = plt.subplot(3, 1, 1)
plt.title('Canonical Evolution Stock')
plt.ylabel('Value')
stock_line, = plt.plot([], [], 'b-')
plt.grid(True)

# Subplot 2: Componente Crypto
ax2 = plt.subplot(3, 1, 2)
plt.title('Canonical Evolution Crypto')
plt.ylabel('Value')
crypto_line, = plt.plot([], [], 'r-')
plt.grid(True)

# Subplot 3: Entrambe le componenti
ax3 = plt.subplot(3, 1, 3)
plt.title('Canonical Components')
plt.ylabel('Value')
both_stock_line, = plt.plot([], [], 'b-', label='Stock')
both_crypto_line, = plt.plot([], [], 'r-', label='Crypto')
plt.legend()
plt.grid(True)

# Formattazione date per tutti i subplot
for ax in [ax1, ax2, ax3]:
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

# Testo per la correlazione
corr_text = ax3.text(0.02, 0.95, '', transform=ax3.transAxes, bbox=dict(facecolor='white', alpha=0.8))

# Liste per salvare i dati
dates = []
stock_vals = []
crypto_vals = []
correlations = []

# Animazione principale
for i in range(window_size, len(returns), step):
    # Seleziona la finestra corrente
    current_data = returns.iloc[i-window_size:i]
    
    # Separa stock e crypto
    stock_returns = current_data[['GOOG', 'MSFT', 'NVDA']]
    crypto_returns = current_data[['BTC-USD', 'ETH-USD', 'SOL-USD']]
    
    # Standardizza
    scaler_stock = StandardScaler()
    scaler_crypto = StandardScaler()
    X = scaler_stock.fit_transform(stock_returns)
    Y = scaler_crypto.fit_transform(crypto_returns)
    
    # Calcola CCA (solo prima componente)
    cca = CCA(n_components=1)
    cca.fit(X, Y)
    X_c, Y_c = cca.transform(X, Y)
    
    # Calcola correlazione
    corr = np.corrcoef(X_c.squeeze(), Y_c.squeeze())[0, 1]
    
    # Salva l'ultimo valore (giorno corrente)
    current_date = returns.index[i]
    dates.append(current_date)
    stock_vals.append(X_c[-1][0])
    crypto_vals.append(Y_c[-1][0])
    correlations.append(corr)
    
    # Aggiorna i grafici
    # Subplot 1: Solo stock
    stock_line.set_data(dates, stock_vals)
    ax1.relim()
    ax1.autoscale_view()
    
    # Subplot 2: Solo crypto
    crypto_line.set_data(dates, crypto_vals)
    ax2.relim()
    ax2.autoscale_view()
    
    # Subplot 3: Entrambe
    both_stock_line.set_data(dates, stock_vals)
    both_crypto_line.set_data(dates, crypto_vals)
    ax3.relim()
    ax3.autoscale_view()
    
    # Testo correlazione
    corr_text.set_text(f'Date: {current_date.strftime("%Y-%m-%d")}\nCorrelation: {corr:.3f}')
    
    # Aggiorna la figura
    plt.tight_layout()
    plt.pause(0.03)

plt.tight_layout()
plt.show()