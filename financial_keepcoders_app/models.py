from utils.utils_bigquery import *
from google.cloud.exceptions import NotFound
from google.oauth2 import service_account
from google.cloud import bigquery
import os
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import talib as ta
import torch
import matplotlib.patches as mpatches
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import nltk
from nltk.corpus import wordnet as wn
import joblib
import io
import base64
import tensorflow as tf
from tensorflow.keras.models import load_model
from dotenv import load_dotenv

# Descargar recursos de NLTK
nltk.download('wordnet', quiet=True)
load_dotenv('./variables_flask.env')

# Conexión a BigQuery

key_path = key_path
project = project_id
dataset = 'gold'
table = 'gold_main_sp500'
table_conca = f'{project}.{dataset}.{table}'

bigquery = BigQueryUtils(key_path)


def load_data():
    """Cargamos los datos desde un CSV si existe, o desde BigQuery si no."""
    data_path = str(os.environ['DATA_PATH'])

    if os.path.exists(data_path):
        # Si el archivo CSV existe, lo leemos y eliminamos la primera columna
        df = pd.read_csv(data_path)
        return df.iloc[:, 1:]
    else:
        # Si el archivo CSV no existe, nos conectamos a BigQuery y guardamos el archivo
        print("El archivo CSV no existe. Conectando a BigQuery...")

        df = bigquery.run_query(
            f"""
            SELECT
                *
            FROM {project}.{dataset}.{table}
            """
        )

        # Guardamos el csv para trabajar de forma más ágil con él
        df.to_csv(data_path, index=False)
        return df


def select_sector_and_cluster(df, sector_elegido, cluster_elegido):
    """Seleccionamos sector y cluster basados en inputs del usuario."""
    sector_to_columns = {
        'Consumo': ('sector_group_Consumer__Energy__and_Utilities', 'industry_group_Retail_and_Consumer_Goods'),
        'Financiero': ('sector_group_Financials_and_Real_Estate', 'industry_group_Financial_Services_and_Banking'),
        'Salud': ('sector_group_Healthcare_and_Defensive_Products', 'industry_group_Health_and_Biotechnology'),
        'Industria': ('sector_group_Industrials_and_Basic_Materials', 'industry_group_Industry_and_Utilities'),
        'Tecnología': ('sector_group_Technology_and_Communications', 'industry_group_Technology_and_Communications')
    }

    # Seleccionamos las columnas correspondientes al sector e industria
    sector_col, industry_col = sector_to_columns[sector_elegido]

    # Filtramos el DataFrame por el sector e industria elegidos
    df_filtrado = df[(df[sector_col] == 1) & (df[industry_col] == 1)]

    # Filtramos por el cluster elegido
    df_cluster_filtrado = df_filtrado[df_filtrado['cluster']
                                      == cluster_elegido]

    return df_cluster_filtrado


def filter_by_volume(df):
    """Filtramos los tickers cuyo volumen medio sea superior al del sector."""
    mean_volume_sector = df['Volume'].mean()
    mean_volume_per_ticker = df.groupby('Ticker')['Volume'].mean()
    tickers_validos = mean_volume_per_ticker[mean_volume_per_ticker >
                                             mean_volume_sector].index
    return df[df['Ticker'].isin(tickers_validos)]


def align_dates(df):
    """Eliminamos duplicados en la columna 'date' por cada ticker."""
    df_sin_duplicados = df.groupby('Ticker').apply(lambda x: x.drop_duplicates(
        subset='date', keep='first')).reset_index(drop=True)
    return df_sin_duplicados


def get_selected_tickers(df, num_tickers=4):
    """Seleccionamos aleatoriamente un conjunto de tickers y devolvemos la lista de tickers."""
    unique_tickers = df['Ticker'].unique()
    tickers_random = random.sample(
        list(unique_tickers), min(num_tickers, len(unique_tickers)))
    return tickers_random


def filter_by_selected_tickers(df, tickers_random):
    """Filtramos el DataFrame en función de los tickers seleccionados y lo ordenamos."""
    df_filtrado = df[df['Ticker'].isin(tickers_random)]
    df_filtrado['Ticker'] = pd.Categorical(
        df_filtrado['Ticker'], categories=tickers_random, ordered=True)
    df_filtrado = df_filtrado.sort_values(by=['Ticker', 'date'])
    return df_filtrado


def calculate_indicators(df):
    """Calculamos indicadores técnicos y patrones de velas."""
    df['SMA_20'] = ta.SMA(df['Close'], timeperiod=20)
    df['EMA_50'] = ta.EMA(df['Close'], timeperiod=50)
    df['ADX'] = ta.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['RSI'] = ta.RSI(df['Close'], timeperiod=14)

    df['CDL_DOJI'] = ta.CDLDOJI(df['Open'], df['High'], df['Low'], df['Close'])
    df['CDL_HAMMER'] = ta.CDLHAMMER(
        df['Open'], df['High'], df['Low'], df['Close'])
    df['CDL_MORNING_STAR'] = ta.CDLMORNINGSTAR(
        df['Open'], df['High'], df['Low'], df['Close'])
    df['CDL_ENGULFING'] = ta.CDLENGULFING(
        df['Open'], df['High'], df['Low'], df['Close'])
    df['CDL_LONGLINE'] = ta.CDLLONGLINE(
        df['Open'], df['High'], df['Low'], df['Close'])

    df['bb_bbh'], df['bb_bbm'], df['bb_bbl'] = ta.BBANDS(
        df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['daily_return'] = df['Close'].pct_change()
    df['macd'], df['macd_signal'], df['macd_hist'] = ta.MACD(
        df['Close'], slowperiod=26, fastperiod=12, signalperiod=9)
    df['stoch'], df['stoch_d'] = ta.STOCH(
        df['High'], df['Low'], df['Close'], fastk_period=21, slowk_period=3, slowd_period=3)

    df['Tomorrow Close'] = df['Close'].shift(-1)
    df['Target'] = np.where(df['Tomorrow Close'] > df['Close'], 1, 0)
    df = df.drop(['Tomorrow Close'], axis=1)
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Volatility'] = df['Log_Return'].rolling(window=5).std()
    df['Volume_Change'] = df['Volume'].pct_change()

    return df


def reset_and_set_index(df):
    """Reseteamos el índice y configuramos 'date' como nuevo índice."""
    df = df.reset_index(drop=True)
    df['date'] = pd.to_datetime(df['date'])
    return df.set_index('date')


def process_data_df(df, sector, cluster, num_tickers=4):
    """Ejecutamos todo el flujo: seleccionamos sector, tickers, filtramos y procesamos el DataFrame."""
    df_filtrado = select_sector_and_cluster(df, sector, cluster)
    df_filtrado = filter_by_volume(df_filtrado)
    tickers_random = get_selected_tickers(df_filtrado, num_tickers)
    df_filtrado = filter_by_selected_tickers(df_filtrado, tickers_random)
    df_filtrado = align_dates(df_filtrado)
    df_procesado = df_filtrado.groupby(
        'Ticker', group_keys=False).apply(calculate_indicators)
    df_procesado = reset_and_set_index(df_procesado)
    return df_procesado, tickers_random


# -------------------------------------------------------------
# Aquí van las funciones relacionadas con noticias y sentimiento
# -------------------------------------------------------------

nltk.download('wordnet', quiet=True)


def perform_sentiment_analysis(texts):
    """Analizamos el sentimiento de las noticias usando FinBERT."""
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained(
        "ProsusAI/finbert")

    results = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt",
                           truncation=True, max_length=512)
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

        sentiment_score = probabilities[0][2].item(
        ) - probabilities[0][0].item()  # Positive - Negative
        results.append(sentiment_score)

    return results


def get_synonyms(word, pos):
    """Obtenemos sinónimos de una palabra en función de su tipo gramatical."""
    synsets = wn.synsets(word, pos=pos)
    return list(set([lemma.name() for synset in synsets for lemma in synset.lemmas()]))


def generate_news_titles(row):
    """Generamos títulos de noticias sintéticas basados en el cambio de precio."""
    price_change = row['Close'] - row['Open']
    percent_change = (price_change / row['Open']) * 100

    templates = [
        "{stock} stock {movement} {percent}% as {market_condition}",
        "Investors {reaction} as {stock} shares {movement} ${price_change}",
        "{stock} {performance} in {volume} trading session",
        "{market_condition} leads to {movement} in {stock} stock",
        "{stock} shares {movement} amid {market_factor}"
    ]

    movement_words = get_synonyms(
        'increase', 'v') if price_change > 0 else get_synonyms('decrease', 'v')
    reaction_words = get_synonyms(
        'optimistic', 'a') if price_change > 0 else get_synonyms('pessimistic', 'a')
    performance_words = get_synonyms(
        'excel', 'v') if price_change > 0 else get_synonyms('struggle', 'v')
    volume_words = ['high-volume', 'active',
                    'busy', 'low-volume', 'quiet', 'subdued']
    market_conditions = ['market volatility',
                         'economic uncertainty', 'sector trends', 'global factors']
    market_factors = ['earnings expectations', 'analyst reports',
                      'industry news', 'technological advancements']

    titles = []
    headline = random.choice(templates).format(
        movement=random.choice(movement_words).replace('_', ' '),
        percent=abs(round(percent_change, 2)),
        price_change=abs(round(price_change, 2)),
        market_condition=random.choice(market_conditions),
        reaction=random.choice(reaction_words).replace('_', ' '),
        performance=random.choice(performance_words).replace('_', ' '),
        volume=random.choice(volume_words),
        market_factor=random.choice(market_factors),
        stock=row['Ticker']
    )
    titles.append(headline)

    return titles


def generate_news_dataframe(df):
    """Generamos un DataFrame con noticias sintéticas para cada fila del DataFrame original."""
    all_news = []
    for date, row in df.iterrows():
        titles = generate_news_titles(row)
        for title in titles:
            all_news.append({
                'Date': date,
                'News_Title': title,
                'Ticker': row['Ticker']
            })

    news_df = pd.DataFrame(all_news)
    return news_df


def analyze_news_sentiment(news_df):
    """Realizamos el análisis de sentimiento de las noticias generadas."""
    text_column = 'News_Title'
    sentiment_scores = perform_sentiment_analysis(news_df[text_column])

    news_df['sentiment_score'] = sentiment_scores
    news_df['sentiment'] = pd.cut(news_df['sentiment_score'],
                                  bins=[-np.inf, -0.05, 0.05, np.inf],
                                  labels=['NEGATIVE', 'NEUTRAL', 'POSITIVE'])
    return news_df


def aggregate_sentiment_by_ticker(news_df):
    """Agregamos el sentimiento por ticker y calculamos el indicador de sentimiento."""
    grouped_sint = news_df.groupby('Ticker')

    agg_sint = pd.DataFrame({
        'positive_sentiment': grouped_sint.apply(lambda x: x[x['sentiment'] == 'POSITIVE']['sentiment_score']),
        'negative_sentiment': grouped_sint.apply(lambda x: x[x['sentiment'] == 'NEGATIVE']['sentiment_score']),
        'neutral_sentiment': grouped_sint.apply(lambda x: x[x['sentiment'] == 'NEUTRAL']['sentiment_score']),
    })

    agg_sint = agg_sint.fillna(0)

    conditions = [
        (agg_sint['positive_sentiment'] != 0),
        (agg_sint['negative_sentiment'] != 0),
        (agg_sint['neutral_sentiment'] != 0)
    ]
    choices = [1, -1, 0]
    agg_sint['sentiment_indicator'] = np.select(conditions, choices, default=0)

    return agg_sint


def combine_stock_and_news(stock_df, sentiment_df):
    """Combinamos los datos de acciones con el sentimiento de las noticias."""
    sentiment_df = sentiment_df.reset_index()

    if 'Ticker' not in stock_df.columns:
        raise ValueError(
            "El DataFrame de acciones no tiene la columna 'Ticker'")

    sentiment_df['sequence'] = sentiment_df.groupby('Ticker').cumcount()
    stock_df = stock_df.reset_index(drop=True)
    stock_df['sequence'] = stock_df.groupby('Ticker').cumcount()

    final_df = pd.merge(stock_df, sentiment_df, left_on=[
                        'Ticker', 'sequence'], right_on=['Ticker', 'sequence'], how='left')
    final_df['sentiment_indicator'] = final_df['sentiment_indicator'].ffill()

    final_df.drop(columns=['level_1'], inplace=True, errors='ignore')

    return final_df


def encode_tickers(final_df):
    """Codificamos los tickers utilizando one-hot encoding y eliminamos columnas innecesarias."""
    encoded_tickers = pd.get_dummies(final_df['Ticker'])
    encoded_tickers = encoded_tickers.astype(int)

    # Concatenamos las codificaciones con el DataFrame final
    final_df = pd.concat([final_df, encoded_tickers], axis=1)

    # Eliminamos las columnas 'Ticker' y 'sequence' y reseteamos el índice
    final_df.drop(['Ticker', 'sequence'], axis=1, inplace=True)
    final_df = final_df.reset_index(drop=True)

    return final_df


def process_news_and_sentiment(df):
    """
    Función principal que engloba todo el proceso de:
    - Generación de noticias sintéticas
    - Análisis de sentimiento
    - Agregación por ticker
    - Combinación con los datos de acciones
    - Codificación de los tickers
    """
    # Generamos el DataFrame de noticias sintéticas
    news_sint = generate_news_dataframe(df)

    # Analizamos el sentimiento de las noticias
    news_sint = analyze_news_sentiment(news_sint)

    # Agregamos el sentimiento por ticker
    agg_sint = aggregate_sentiment_by_ticker(news_sint)

    # Combinamos las noticias con los datos de acciones
    final_df = combine_stock_and_news(df, agg_sint)

    # Codificamos los tickers
    final_df = encode_tickers(final_df)

    return final_df

# ----------------------------------------------
# Funciones de partición y preprocesamiento de datos
# ----------------------------------------------


def save_train_test(train, test, train_filename, test_filename):
    """
    Guardamos los conjuntos de train y test en archivos CSV.
    """
    train.to_csv(train_filename, sep=';', decimal='.', index=False)
    test.to_csv(test_filename, sep=';', decimal='.', index=False)


def apply_market_clustering(df):
    """
    Aplicamos clustering para detectar regímenes de mercado.
    """
    X_for_clustering = df[['Volatility']]
    kmeans = KMeans(n_clusters=3, random_state=42).fit(X_for_clustering)
    df['Market_Regime'] = kmeans.predict(X_for_clustering)
    return df


def conditions_buy(df):
    """
    Definimos las condiciones para la compra según los patrones de velas e indicadores técnicos.
    """
    conditions = [
        (df['CDL_ENGULFING'] == 1) & (df['Low'] <
                                      df['bb_bbl']) & (df['High'] < df['bb_bbm']),
        (df['CDL_LONGLINE'] == 1) & (df['Close'] >
                                     df['bb_bbm']) & (df['High'] < df['bb_bbh']),
        (df['CDL_DOJI'] == 1) & (df['Low'] < df['bb_bbl']) & (
            df['High'] < df['bb_bbm']),
        (df['CDL_HAMMER'] == 1) & (df['Low'] < df['bb_bbl']) & (
            df['High'] < df['bb_bbm']),
        (df['CDL_MORNING_STAR'] == 1) & (df['Low'] <
                                         df['bb_bbl']) & (df['High'] < df['bb_bbm']),
        (df['stoch_d'] > df['stoch']),
        (df['macd_signal'] > df['macd'])
    ]
    return conditions

def process_data(df, df_name="DataFrame"):
    """
    Procesamos el DataFrame (train o test) para aplicar condiciones de compra y eliminar columnas innecesarias.
    """
    # Reemplazamos valores en los patrones de velas
    columns_to_replace = ['CDL_DOJI', 'CDL_HAMMER',
                          'CDL_MORNING_STAR', 'CDL_ENGULFING', 'CDL_LONGLINE']
    for column in columns_to_replace:
        df[column] = df[column].replace({100: 1, -100: 0})

    # Aplicamos las condiciones de compra
    conditions = conditions_buy(df)
    df['low_buy'] = np.select(conditions, [1] * len(conditions), default=0)

    # Eliminamos columnas innecesarias
    columns_to_drop = ['sector_group_Consumer__Energy__and_Utilities', 'sector_group_Technology_and_Communications',
                       'industry_group_Retail_and_Consumer_Goods', 'industry_group_Technology_and_Communications',
                       'sector_group_Financials_and_Real_Estate', 'sector_group_Healthcare_and_Defensive_Products',
                       'sector_group_Industrials_and_Basic_Materials', 'industry_group_Energy_and_Natural_Resources',
                       'industry_group_Financial_Services_and_Banking', 'industry_group_Health_and_Biotechnology',
                       'industry_group_Industry_and_Utilities', 'industry_group_Real_Estate_and_Infrastructure',
                       'cluster', 'Open', 'High', 'Low', 'stoch_d', 'stoch', 'id', 'adjclose', 'BOPGSTB', 'CPIAUCSL',
                       'PPIACO', 'RSAFS', 'macd_signal', 'macd', 'macd_hist', 'bb_bbm', 'bb_bbh', 'bb_bbl',
                       'daily_return', 'positive_sentiment', 'negative_sentiment', 'neutral_sentiment', 'EMA_50', 'SMA_20']

    df = df.drop(columns_to_drop, axis=1)

    # Eliminamos valores nulos
    df = df.dropna()

    # Aplicamos clustering de regímenes de mercado
    df = apply_market_clustering(df)
    df.reset_index(drop=True, inplace=True)

    print(
        f"Dimensiones del conjunto de {df_name} tras el procesamiento: {df.shape}")

    return df

def split_train_test(final_df, tickers, test_size=1):
    """Dividimos los datos por ticker para obtener conjuntos de train y test."""
    dfs_por_ticker = {
        ticker: final_df[final_df[ticker] == 1] for ticker in tickers}
    train = pd.concat(dfs_por_ticker[ticker]
                      for ticker in tickers[:-test_size])
    test = dfs_por_ticker[tickers[-test_size]]
    return train, test


def split_data_classification(df_train, df_test):
    """Divide los datos en características (X) y etiquetas (y) para clasificación."""
    X_train = df_train.drop(['Target', 'Close'], axis=1)
    y_train = df_train['Target']
    X_test = df_test.drop(['Target', 'Close'], axis=1)
    y_test = df_test['Target']
    return X_train, y_train, X_test, y_test


def split_data_regression(df_train, df_test):
    """Divide los datos en características (X) y etiquetas (y) para regresión."""
    X_train = df_train.drop(['Log_Return', 'Close'], axis=1)
    y_train = df_train['Log_Return']
    X_test = df_test.drop(['Log_Return', 'Close'], axis=1)
    y_test = df_test['Log_Return']
    return X_train, y_train, X_test, y_test


def scale_data_classification(X_train, X_test, y_train, y_test, df_train, df_test, tickers_train, ticker_test):
    """Escala los datos de entrenamiento y prueba para clasificación."""
    scalers_X = {}
    X_train_scaled_ = {}
    X_test_scaled_ = {}

    for ticker in tickers_train:
        scalers_X[ticker] = MinMaxScaler()
        X_train_ticker = X_train[df_train[ticker] == 1]
        X_train_scaled_[ticker] = scalers_X[ticker].fit_transform(
            X_train_ticker)

    X_test_ticker = X_test[df_test[ticker_test] == 1]
    train_size = int(0.5 * len(X_test_ticker))
    X_train_test = X_test_ticker[:train_size]
    X_test_test_final = X_test_ticker[train_size:]
    scaler_X_test = MinMaxScaler()
    X_train_scaled_[ticker_test] = scaler_X_test.fit_transform(X_train_test)
    X_test_scaled_[ticker_test] = scaler_X_test.transform(X_test_test_final)
    for ticker in tickers_train:
        print(
            f"{ticker} - X_train_scaled shape: {X_train_scaled_[ticker].shape}")
        print(
            f"{ticker} - y_train shape: {y_train[df_train[ticker] == 1].shape}")

    print(
        f"{ticker_test} - X_test_scaled shape: {X_test_scaled_[ticker_test].shape}")
    print(
        f"{ticker_test} - y_test shape: {y_test[df_test[ticker_test] == 1].shape}")
    return X_train_scaled_, X_test_scaled_, y_test


def scale_data_regression(X_train, X_test, y_train, y_test, df_train, df_test, tickers_train, ticker_test):
    """Escala los datos de entrenamiento y prueba para regresión."""
    scalers_X = {}
    scalers_y = {}
    X_train_scaled_ = {}
    y_train_scaled_ = {}
    X_test_scaled_ = {}
    y_test_scaled_ = {}

    for ticker in tickers_train:
        scalers_X[ticker] = MinMaxScaler()
        scalers_y[ticker] = MinMaxScaler()
        X_train_ticker = X_train[df_train[ticker] == 1]
        y_train_ticker = y_train[df_train[ticker] == 1].values.reshape(-1, 1)
        X_train_scaled_[ticker] = scalers_X[ticker].fit_transform(
            X_train_ticker)
        y_train_scaled_[ticker] = scalers_y[ticker].fit_transform(
            y_train_ticker)

    X_test_ticker = X_test[df_test[ticker_test] == 1]
    y_test_ticker = y_test[df_test[ticker_test] == 1].values.reshape(-1, 1)
    train_size = int(0.5 * len(X_test_ticker))
    X_train_test = X_test_ticker[:train_size]
    X_test_test_final = X_test_ticker[train_size:]
    y_train_test = y_test_ticker[:train_size]
    y_test_test_final = y_test_ticker[train_size:]

    scaler_X_test = MinMaxScaler()
    X_train_scaled_[ticker_test] = scaler_X_test.fit_transform(X_train_test)
    scaler_y_test = MinMaxScaler()
    y_train_scaled_[ticker_test] = scaler_y_test.fit_transform(y_train_test)
    X_test_scaled_[ticker_test] = scaler_X_test.transform(X_test_test_final)
    y_test_scaled_[ticker_test] = scaler_y_test.transform(y_test_test_final)
    return X_train_scaled_, X_test_scaled_, y_train_scaled_, y_test_scaled_, scaler_y_test


# Walk-Forward Validation
def walk_forward_validation(X, y, initial_train_size, step_size):
    """Aplica validación walk-forward."""
    splits = []
    for i in range(initial_train_size, len(X), step_size):
        X_train_split = X[:i]
        X_test_split = X[i:i + step_size]
        y_train_split = y[:i]
        y_test_split = y[i:i + step_size]
        splits.append((X_train_split, X_test_split,
                      y_train_split, y_test_split))
    return splits


def cargar_modelos(modelos, X_test_scaled_, ticker_test):
    """
    Cargamos y preparamos los modelos de Keras y Random Forest para la evaluación.
    """
    predicciones = {}

    for nombre_modelo, (modelo_keras_path, modelo_rf_path) in modelos.items():
        print(f"Evaluando el modelo: {nombre_modelo}")

        # Cargamos modelo de Keras si existe
        if modelo_keras_path:
            model_keras = load_model(modelo_keras_path)
            X_test_reshaped = X_test_scaled_[ticker_test].reshape(
                (X_test_scaled_[ticker_test].shape[0], 1, X_test_scaled_[ticker_test].shape[1]))
            predicciones_keras = (model_keras.predict(
                X_test_reshaped) > 0.5).astype(int)
        else:
            predicciones_keras = None

        # Cargamos modelo Random Forest si existe
        if modelo_rf_path:
            rf_model = joblib.load(modelo_rf_path)
            if predicciones_keras is not None:
                predicciones_rf = rf_model.predict(np.column_stack(
                    [X_test_scaled_[ticker_test], predicciones_keras]))
            else:
                predicciones_rf = rf_model.predict(X_test_scaled_[ticker_test])
            predicciones[nombre_modelo] = predicciones_rf
        else:
            predicciones[nombre_modelo] = predicciones_keras

    return predicciones


def backtesting_estrategia_clasificacion(df_test, predicciones, initial_cash=10000):
    """
    Realizamos el backtesting para la estrategia de clasificación basada en las predicciones del modelo.
    """
    cash = initial_cash
    holdings = 0
    portfolio_value_classification = []

    for i in range(len(df_test) - 1):
        current_price = df_test['Close'].iloc[i]
        predicted_class = predicciones[i]

        # Ejecutamos la estrategia de compra/venta según las predicciones
        if predicted_class == 1 and holdings == 0:
            holdings = cash / current_price
            cash = 0
        elif predicted_class == 0 and holdings > 0:
            cash = holdings * current_price
            holdings = 0

        portfolio_value_classification.append(cash + holdings * current_price)

    portfolio_value_classification = np.array(portfolio_value_classification)
    cumulative_return = (
        portfolio_value_classification[-1] - initial_cash) / initial_cash

    return portfolio_value_classification, cumulative_return


def estrategia_buy_and_hold(df_test, initial_cash=10000):
    """
    Simulamos la estrategia Buy and Hold.
    """
    buy_and_hold_holdings = initial_cash / df_test['Close'].iloc[0]
    buy_and_hold_final_value = buy_and_hold_holdings * \
        df_test['Close'].iloc[-1]
    buy_and_hold_cumulative_return = (
        buy_and_hold_final_value - initial_cash) / initial_cash

    return buy_and_hold_holdings, buy_and_hold_final_value, buy_and_hold_cumulative_return


def evaluar_modelos_clasificacion(modelos, df_test, X_test_scaled_, ticker_test, initial_cash=10000):
    """
    Evaluamos todos los modelos de clasificación, realizamos backtesting y comparamos el rendimiento.
    """
    predicciones_modelos = cargar_modelos(modelos, X_test_scaled_, ticker_test)

    best_return = -np.inf
    best_model = None
    best_portfolio_value_classification = None
    best_buy_and_hold_portfolio_value = None
    best_df_test = None

    for nombre_modelo, predicciones in predicciones_modelos.items():
        df_test = df_test.tail(len(predicciones))
        df_test['Predictions'] = predicciones

        # Backtesting para clasificación
        portfolio_value_classification, classification_cumulative_return = backtesting_estrategia_clasificacion(
            df_test, predicciones, initial_cash)

        # Estrategia Buy and Hold
        buy_and_hold_holdings, buy_and_hold_final_value, buy_and_hold_cumulative_return = estrategia_buy_and_hold(
            df_test)

        # Comparamos retornos y elegimos el mejor modelo
        if classification_cumulative_return > best_return:
            best_return = classification_cumulative_return
            best_model = nombre_modelo
            best_portfolio_value_classification = portfolio_value_classification
            best_buy_and_hold_portfolio_value = df_test['Close'].values * \
                buy_and_hold_holdings
            best_df_test = df_test.copy()

        print(f"{nombre_modelo} - Classification Cumulative Return: {classification_cumulative_return * 100:.2f}%")

    print(
        f"Buy and Hold Cumulative Return: {buy_and_hold_cumulative_return * 100:.2f}%")
    print(
        f"El mejor modelo es {best_model} con un retorno acumulado de {best_return * 100:.2f}%")

    return best_model, best_portfolio_value_classification, best_buy_and_hold_portfolio_value, best_df_test, best_return, buy_and_hold_cumulative_return


def graficar_resultados(best_model, best_portfolio_value_classification, best_buy_and_hold_portfolio_value, best_df_test):
    """
    Graficamos los resultados del mejor modelo y la estrategia Buy and Hold.
    """
    graphs = []  # Lista para almacenar los gráficos generados

    # Gráfico 1: Valor del Portafolio con Estrategia del Mejor Modelo
    img1 = io.BytesIO()
    plt.figure()
    plt.plot(best_portfolio_value_classification,
             label=f'{best_model} Strategy Portfolio Value')
    plt.title(f"Valor del Portafolio con Estrategia de {best_model}")
    plt.xlabel("Días")
    plt.ylabel("Valor del Portafolio")
    plt.legend()
    plt.savefig(img1, format='png')
    img1.seek(0)
    graphs.append(base64.b64encode(img1.getvalue()).decode('utf8'))
    plt.close()

    # Gráfico 2: Valor del Portafolio con Estrategia Buy and Hold
    img2 = io.BytesIO()
    plt.figure()
    plt.plot(best_buy_and_hold_portfolio_value,
             label='Buy and Hold Portfolio Value', color='green')
    plt.title(f"Valor del Portafolio con Estrategia Buy and Hold")
    plt.xlabel("Días")
    plt.ylabel("Valor del Portafolio")
    plt.legend()
    plt.savefig(img2, format='png')
    img2.seek(0)
    graphs.append(base64.b64encode(img2.getvalue()).decode('utf8'))
    plt.close()

    # Gráfico 3: Últimos 30 días de precios de cierre con predicción para el día 31
    img3 = io.BytesIO()
    last_30_closes = best_df_test['Close'].iloc[-30:].values
    next_day_prediction = best_df_test['Predictions'].iloc[-1]

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 31), last_30_closes, marker='o',
             color='blue', label='Precio de Cierre')

    if next_day_prediction == 1:
        plt.annotate('', xy=(31, last_30_closes[-1]), xytext=(30, last_30_closes[-1] - 0.5),
                     arrowprops=dict(facecolor='green', shrink=0.05, width=1.5, headwidth=6))
    else:
        plt.annotate('', xy=(31, last_30_closes[-1]), xytext=(30, last_30_closes[-1] + 0.5),
                     arrowprops=dict(facecolor='red', shrink=0.05, width=1.5, headwidth=6))

    plt.title(
        f'Últimos 30 Días de Precios de Cierre con Predicción para el Día 31 ({best_model})')
    plt.xlabel('Día')
    plt.ylabel('Precio de Cierre')

    green_arrow = mpatches.Patch(color='green', label='Alcista (Predicción)')
    red_arrow = mpatches.Patch(color='red', label='Bajista (Predicción)')
    plt.legend(handles=[green_arrow, red_arrow])

    plt.grid(True)
    plt.savefig(img3, format='png')
    img3.seek(0)
    graphs.append(base64.b64encode(img3.getvalue()).decode('utf8'))
    plt.close()

    return graphs


def cargar_modelos_regresion(modelos, X_test_scaled_, ticker_test):
    """
    Cargamos y preparamos los modelos de Keras y Random Forest para la evaluación.
    """
    predicciones = {}
    secuencias = {}

    for nombre_modelo, (modelo_keras_path, modelo_rf_path) in modelos.items():
        print(f"Evaluando el modelo: {nombre_modelo}")

        # Cargamos modelo de Keras si existe
        if modelo_keras_path:
            model_keras = load_model(modelo_keras_path)
            X_test_reshaped = X_test_scaled_[ticker_test].reshape(
                (X_test_scaled_[ticker_test].shape[0], 1, X_test_scaled_[ticker_test].shape[1]))
            predicciones_keras = model_keras.predict(X_test_reshaped).flatten()

            # Guardamos la última secuencia para predecir el día 31
            last_sequence = X_test_scaled_[
                ticker_test][-1].reshape(1, 1, X_test_scaled_[ticker_test].shape[1])
            secuencia_pred_31 = model_keras.predict(last_sequence).flatten()[0]
        else:
            predicciones_keras = None
            last_sequence = None

        # Cargamos modelo Random Forest si existe
        if modelo_rf_path:
            rf_model = joblib.load(modelo_rf_path)
            if predicciones_keras is not None:
                predicciones_rf = rf_model.predict(np.column_stack(
                    [X_test_scaled_[ticker_test], predicciones_keras]))
            else:
                predicciones_rf = rf_model.predict(X_test_scaled_[ticker_test])
            predicciones[nombre_modelo] = predicciones_rf
            secuencias[nombre_modelo] = last_sequence
        else:
            predicciones[nombre_modelo] = predicciones_keras
            secuencias[nombre_modelo] = secuencia_pred_31

    return predicciones, secuencias


def backtesting_estrategia_regresion(df_test, predicciones, initial_cash=10000):
    """
    Realizamos el backtesting para la estrategia de regresión basada en las predicciones del modelo.
    """
    cash = initial_cash
    holdings = 0
    portfolio_value = []

    for i in range(len(df_test) - 1):
        current_price = df_test['Close'].iloc[i]
        predicted_log_return = predicciones[i]

        # Estimamos el cambio porcentual usando log_return
        predicted_next_price = current_price * np.exp(predicted_log_return)

        # Si sube, compramos
        if predicted_next_price > current_price and holdings == 0:
            holdings = cash / current_price
            cash = 0
        # Si baja, vendemos
        elif predicted_next_price <= current_price and holdings > 0:
            cash = holdings * current_price
            holdings = 0

        portfolio_value.append(cash + holdings * current_price)

    portfolio_value = np.array(portfolio_value)
    cumulative_return = (portfolio_value[-1] - initial_cash) / initial_cash

    return portfolio_value, cumulative_return


def evaluar_modelos_regresion(modelos, df_test, X_test_scaled_, ticker_test, initial_cash=10000):
    """
    Evaluamos todos los modelos de regresión, realizamos backtesting y comparamos el rendimiento.
    """
    predicciones_modelos, secuencias_pred_31 = cargar_modelos_regresion(
        modelos, X_test_scaled_, ticker_test)

    best_return = -np.inf
    best_model = None
    best_portfolio_value = None
    best_buy_and_hold_value = None
    best_df_test = None
    best_pred_31 = None

    for nombre_modelo, predicciones in predicciones_modelos.items():
        df_test = df_test.tail(len(predicciones))
        df_test['Predictions'] = predicciones

        # Backtesting para regresión
        portfolio_value, cumulative_return = backtesting_estrategia_regresion(
            df_test, predicciones, initial_cash)

        # Estrategia Buy and Hold
        buy_and_hold_holdings, buy_and_hold_final_value, buy_and_hold_cumulative_return = estrategia_buy_and_hold(
            df_test)

        # Comparamos retornos y elegimos el mejor modelo
        if cumulative_return > best_return:
            best_return = cumulative_return
            best_model = nombre_modelo
            best_portfolio_value = portfolio_value
            best_buy_and_hold_value = df_test['Close'].values * \
                buy_and_hold_holdings
            best_df_test = df_test.copy()
            best_pred_31 = secuencias_pred_31[nombre_modelo]

        print(f"{nombre_modelo} - Cumulative Return: {cumulative_return * 100:.2f}%")
        df_test = df_test.drop(['Predictions'], axis=1)

    print(
        f"Buy and Hold Cumulative Return: {buy_and_hold_cumulative_return * 100:.2f}%")
    print(
        f"El mejor modelo es {best_model} con un retorno acumulado de {best_return * 100:.2f}%")

    return best_model, best_portfolio_value, best_buy_and_hold_value, best_df_test, best_pred_31, best_return, buy_and_hold_cumulative_return


def graficar_resultados_regresion(best_model, best_portfolio_value, best_buy_and_hold_value, best_df_test, best_pred_31):
    """
    Graficamos los resultados del mejor modelo de regresión y la estrategia Buy and Hold.
    """
    graphs = []  # Lista para almacenar los gráficos generados

    # Gráfico 1: Valor del Portafolio con Estrategia del Mejor Modelo
    img1 = io.BytesIO()
    plt.figure()
    plt.plot(best_portfolio_value,
             label=f'{best_model} Strategy Portfolio Value')
    plt.title(f"Valor del Portafolio con Estrategia de {best_model}")
    plt.xlabel("Días")
    plt.ylabel("Valor del Portafolio")
    plt.legend()
    plt.savefig(img1, format='png')
    img1.seek(0)
    graphs.append(base64.b64encode(img1.getvalue()).decode('utf8'))
    plt.close()

    # Gráfico 2: Valor del Portafolio con Estrategia Buy and Hold
    img2 = io.BytesIO()
    plt.figure()
    plt.plot(best_buy_and_hold_value,
             label='Buy and Hold Portfolio Value', color='green')
    plt.title(f"Valor del Portafolio con Estrategia Buy and Hold")
    plt.xlabel("Días")
    plt.ylabel("Valor del Portafolio")
    plt.legend()
    plt.savefig(img2, format='png')
    img2.seek(0)
    graphs.append(base64.b64encode(img2.getvalue()).decode('utf8'))
    plt.close()

    # Gráfico 3: Precios de Cierre con Predicción del Día 31
    img3 = io.BytesIO()
    prices_real = best_df_test['Close'].values[-30:]
    P_30 = prices_real[-1]
    P_31_pred = P_30 * np.exp(best_pred_31)

    plt.figure()
    plt.plot(range(1, 31), prices_real, label='Precio real')
    plt.plot(31, P_31_pred, 'ro', label='Precio predicho día 31')
    plt.axvline(x=30, color='gray', linestyle='--')
    plt.legend()
    plt.title(f"Precios de Cierre con Predicción del Día 31 ({best_model})")
    plt.xlabel("Días")
    plt.ylabel("Precio de Cierre")
    plt.savefig(img3, format='png')
    img3.seek(0)
    graphs.append(base64.b64encode(img3.getvalue()).decode('utf8'))
    plt.close()

    return graphs
