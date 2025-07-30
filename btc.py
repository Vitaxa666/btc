import logging
import os
import numpy as np
import pandas as pd
import requests
import tensorflow as tf
import matplotlib.pyplot as plt

from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.preprocessing import MinMaxScaler

from telegram import Update, InputFile
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler

from apscheduler.schedulers.background import BackgroundScheduler

# === 🔐 Конфіг ===
TELEGRAM_TOKEN = "YOUR_BOT_TOKEN"
CHAT_ID = YOUR_CHAT_ID
MODEL_PATH = "btc_lstm_model.h5"

# === 📥 Binance API дані ===
def get_btc_data():
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": "BTCUSDT",
        "interval": "1m",
        "limit": 500  # ~8 годин
    }
    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_base', 'taker_quote', 'ignore'
    ])
    df['close'] = pd.to_numeric(df['close'])
    df['volume'] = pd.to_numeric(df['volume'])
    df['quote_volume'] = pd.to_numeric(df['quote_volume'])
    return df[['close', 'volume', 'quote_volume']]

# === ⚙️ Додавання індикаторів ===
def prepare_features(df):
    df = df.copy()
    df['RSI'] = RSIIndicator(close=df['close'], window=14).rsi()
    macd = MACD(close=df['close'])
    df['MACD'] = macd.macd_diff()
    df.dropna(inplace=True)
    return df

# === 🧠 Створення моделі ===
def create_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# === 🧠 Прогноз з тренуванням/завантаженням ===
def predict_lstm(df, forecast_minutes=60):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    X = []
    for i in range(60, len(scaled)):
        X.append(scaled[i-60:i])
    X = np.array(X)
    y = scaled[60:, 0]

    # Створення або завантаження моделі
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
    else:
        model = create_model((X.shape[1], X.shape[2]))
        model.fit(X, y, epochs=7, batch_size=32, verbose=0)
        model.save(MODEL_PATH)

    # Прогноз майбутнього
    last_seq = scaled[-60:]
    predictions_scaled = []

    for _ in range(forecast_minutes):
        input_seq = np.expand_dims(last_seq, axis=0)
        pred = model.predict(input_seq, verbose=0)
        predictions_scaled.append(pred[0][0])

        # готуємо наступний крок
        next_input = np.append(last_seq[1:], [[pred[0][0]] + [0]*(scaled.shape[1]-1)], axis=0)
        last_seq = next_input

    predictions = scaler.inverse_transform(
        np.hstack([
            np.array(predictions_scaled).reshape(-1,1),
            np.zeros((forecast_minutes, scaled.shape[1] - 1))
        ])
    )[:, 0]

    current_price = df['close'].iloc[-1]
    return current_price, predictions

# === 📈 Побудова графіку ===
def plot_prediction(actual, predicted):
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(actual)), actual, label="Actual", color="blue")
    plt.plot(range(len(actual), len(actual) + len(predicted)), predicted, label="Forecast", color="orange")
    plt.xlabel("Час (хвилини)")
    plt.ylabel("Ціна BTC")
    plt.title("Прогноз ціни Bitcoin")
    plt.legend()
    plt.tight_layout()
    plt.savefig("forecast.png")
    plt.close()

# === 📤 Надсилання результату ===
async def send_forecast(app, minutes=60):
    try:
        df = get_btc_data()
        df_feat = prepare_features(df)
        actual = df_feat['close'][-60:].tolist()
        current, forecast = predict_lstm(df_feat, forecast_minutes=minutes)

        diff = forecast[-1] - current
        trend = "⬆ Зростання" if diff > 0 else "⬇ Падіння"

        plot_prediction(actual, forecast)

        msg = (
            f"📊 Прогноз BTC на {minutes} хв:\n"
            f"💰 Поточна ціна: {current:.2f} USDT\n"
            f"🔮 Прогноз: {forecast[-1]:.2f} USDT\n"
            f"📈 Очікуваний рух: {trend} ({diff:+.2f} USDT)"
        )

        await app.bot.send_message(chat_id=CHAT_ID, text=msg)
        with open("forecast.png", "rb") as img:
            await app.bot.send_photo(chat_id=CHAT_ID, photo=InputFile(img))
    except Exception as e:
        logging.error(f"❌ Помилка прогнозу: {e}")

# === 🤖 Telegram команди ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👋 Я BTC-прогноз бот. Напиши /predict щоб отримати прогноз.")

async def manual_predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await send_forecast(context.application, minutes=60)

async def predict30(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await send_forecast(context.application, minutes=30)

async def predict120(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await send_forecast(context.application, minutes=120)

# === 🟢 Запуск бота ===
def main():
    logging.basicConfig(level=logging.INFO)
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("predict", manual_predict))
    app.add_handler(CommandHandler("predict30", predict30))
    app.add_handler(CommandHandler("predict120", predict120))

    scheduler = BackgroundScheduler()
    scheduler.add_job(lambda: app.create_task(send_forecast(app, minutes=60)), 'interval', minutes=10)
    scheduler.start()

    print("✅ Бот працює")
    app.run_polling()

if __name__ == "__main__":
    main()
