from dotenv import load_dotenv
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, url_for
import models  # Importamos el archivo models.py
import os
import io
import base64
import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use('Agg')  # Usamos 'Agg' para evitar abrir ventanas de gráficos

import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

# Ruta para seleccionar el sector

@app.route('/sector', methods=['GET', 'POST'])
def select_problem():
    if request.method == 'POST':
        problem = request.form['problem']
    else:
        problem = request.args.get('problem')

    # Recogemos el mensaje de error si existe
    error = request.args.get('error')
    return render_template('sector.html', problem=problem, error=error)

# Ruta para seleccionar el cluster

@app.route('/cluster', methods=['GET', 'POST'])
def select_sector():
    if request.method == 'POST':
        sector = request.form['sector']
        problem = request.form['problem']
    else:
        sector = request.args.get('sector')
        problem = request.args.get('problem')

    # Recogemos el mensaje de error si existe
    error = request.args.get('error')
    return render_template('cluster.html', sector=sector, problem=problem, error=error)

# Ruta para mostrar los resultados después de seleccionar el cluster

@app.route('/resultados', methods=['POST'])
def select_cluster():
    cluster = int(request.form['cluster'])
    sector = request.form['sector']
    problem = request.form['problem']

    # Cargar los datos y ejecutar el procesamiento del DataFrame
    df = models.load_data()

    # Procesamos los datos basados en el sector y cluster seleccionados
    df, tickers_random = models.process_data_df(df, sector, cluster)
    print(tickers_random)

    # Si no tenemos 4 tickers en tickers_random, redirigir a la selección de sector con un mensaje de error
    if len(tickers_random) != 4:
        return redirect(url_for('select_problem', sector=sector, problem=problem, error="No se encontraron 4 tickers. Por favor, seleccione de nuevo sector y cluster:"))


    final_df = models.process_news_and_sentiment(df)
    train, test = models.split_train_test(final_df, tickers_random)
    tickers_train = tickers_random[:-1]
    ticker_test = tickers_random[-1]
    # Diccionarios para almacenar los datos escalados y los splits
    X_train_splits = {}
    y_train_splits = {}

    if problem == 'clasificacion':
        models.save_train_test(train, test, train_filename='./train_classification.csv',
                        test_filename='./test_classification.csv')
        df_train = models.process_data(pd.read_csv(
            './train_classification.csv', sep=';', decimal='.'), df_name="df_train")
        df_test = models.process_data(pd.read_csv(
            './test_classification.csv', sep=';', decimal='.'), df_name="df_test")
        X_train, y_train, X_test, y_test = models.split_data_classification(
            df_train, df_test)
        X_train_scaled_, X_test_scaled_, y_test = models.scale_data_classification(
            X_train, X_test, y_train, y_test, df_train, df_test, tickers_train, ticker_test)
        # Aplicamos Walk-Forward Validation a cada acción de entrenamiento
        for ticker in tickers_train:
            X_train_splits[ticker] = models.walk_forward_validation(X_train_scaled_[
                ticker], y_train[df_train[ticker] == 1], initial_train_size=200, step_size=30)

    elif problem == 'regresion':
        models.save_train_test(
            train, test, train_filename='./train_regression.csv', test_filename='./test_regression.csv')
        df_train = models.process_data(pd.read_csv(
            './train_regression.csv', sep=';', decimal='.'), df_name="df_train")
        df_test = models.process_data(pd.read_csv(
            './test_regression.csv', sep=';', decimal='.'), df_name="df_test")
        X_train, y_train, X_test, y_test = models.split_data_regression(
            df_train, df_test)
        X_train_scaled_, X_test_scaled_, y_train_scaled_, y_test_scaled_, scaler_y_test = models.scale_data_regression(
            X_train, X_test, y_train, y_test, df_train, df_test, tickers_train, ticker_test)
        # Aplicamos Walk-Forward Validation a cada acción de entrenamiento
        for ticker in tickers_train:
            X_train_splits[ticker] = models.walk_forward_validation(X_train_scaled_[
                ticker], y_train_scaled_[ticker], initial_train_size=200, step_size=30)

    # Combinamos los splits de las acciones de entrenamiento
    train_splits_combined = []
    num_splits = len(X_train_splits[tickers_train[0]])

    for i in range(num_splits):  # Iterar sobre cada split
        # Concatenamos los splits de cada acción (X_train, X_test, y_train, y_test)
        X_train_combined = np.concatenate(
            [X_train_splits[ticker][i][0] for ticker in tickers_train])
        X_test_combined = np.concatenate(
            [X_train_splits[ticker][i][1] for ticker in tickers_train])
        y_train_combined = np.concatenate(
            [X_train_splits[ticker][i][2] for ticker in tickers_train])
        y_test_combined = np.concatenate(
            [X_train_splits[ticker][i][3] for ticker in tickers_train])

        train_splits_combined.append(
            (X_train_combined, X_test_combined, y_train_combined, y_test_combined))

    # Asegurarnos de que las dimensiones de entrada sean (samples, 1, features) para redes neuronales
    X_train_combined_reshaped = X_train_combined.reshape(
        X_train_combined.shape[0], 1, X_train_combined.shape[1])
    X_test_combined_reshaped = X_test_combined.reshape(
        X_test_combined.shape[0], 1, X_test_combined.shape[1])
    
    # Guardar df_test y X_test_scaled_ para Backtesting
    df_test.to_csv('./df_test_flask.csv', index=True)
    X_test_scaled_list = {key: value.tolist()
                          for key, value in X_test_scaled_.items()}
    with open('./X_test_scaled_flask.json', 'w') as f:
        json.dump(X_test_scaled_list, f)

    # Mostramos las acciones seleccionadas (tickers_random) y las dimensiones de los datos de entrenamiento
    return render_template('resultados.html',
                           tickers_random=tickers_random,
                           X_train_shape=X_train_combined_reshaped.shape,
                           X_test_shape=X_test_combined_reshaped.shape, sector=sector, problem=problem,
                           )


# Ruta para ver el backtesting

@app.route('/ver_backtesting', methods=['POST'])
def ver_backtesting():
    sector = request.form['sector']
    problem = request.form['problem']
    # Recogemos la lista de tickers seleccionados
    tickers_random = request.form.getlist('tickers_random')
    ticker_test = tickers_random[-1]
    # Recuperar df_test desde la sesión
    df_test = pd.read_csv('./df_test_flask.csv')

    try:
        with open('./X_test_scaled_flask.json', 'r') as f:
            X_test_scaled_list = json.load(f)
    except FileNotFoundError:
        print("Error: X_test_scaled_flask.json not found.")
        return "Error: No se encontró el archivo X_test_scaled_flask.json"
    except json.JSONDecodeError:
        print("Error: JSON format issue.")
        return "Error: Fallo al decodificar JSON."

    if X_test_scaled_list:
        # Convertimos las listas de vuelta a arrays de NumPy
        X_test_scaled_ = {key: np.array(value)
                          for key, value in X_test_scaled_list.items()}
    else:
        print("Error: X_test_scaled_list is empty or None")
        return "Error: X_test_scaled_list is empty or None"
    
    # Listas para almacenar los resultados y las gráficas
    graph_urls = []
    result_texts = []

    # Ejecutamos el backtesting según el tipo de problema
    if problem == 'clasificacion':
        modelos = {
            f'RF - Clasificación - {ticker_test}': ('', './rf_model_classification_EXPO.pkl'),
            f'GRU - Clasificación - {ticker_test}': ('./gru_model_classification_EXPO.h5', ''),
            f'LSTM+RF - Clasificación - {ticker_test}': ('./lstm_model_hybrid_classification_EXPO.h5', './random_forest_lstm_hybrid_classification_EXPO.pkl')
        }
        best_model, best_portfolio_value_classification, best_buy_and_hold_portfolio_value, best_df_test, best_return, buy_and_hold_cumulative_return = models.evaluar_modelos_clasificacion(
            modelos, df_test, X_test_scaled_, ticker_test
        )

        # Capturar el print de retornos con los valores correctos
        result_texts.append(
            f"El mejor modelo es {best_model} con un retorno acumulado de {best_return * 100:.2f}%")
        result_texts.append(
            f"Buy and Hold Cumulative Return: {buy_and_hold_cumulative_return * 100:.2f}%")


        # Generamos las gráficas para clasificación
        graph_urls += models.graficar_resultados(best_model, best_portfolio_value_classification,
                                                 best_buy_and_hold_portfolio_value, best_df_test)



    elif problem == 'regresion':
        modelos = {
            f'RF - Clasificación - {ticker_test}': ('./gru_model_regression_EXPO.h5', ''),
            f'LSTM - Regresión - {ticker_test}': ('./lstm_model_regression_EXPO.h5', ''),
            f'RF - Regresión - {ticker_test}': ('', './rf_model_regression_EXPO.pkl')
        }
        best_model, best_portfolio_value, best_buy_and_hold_value, best_df_test, best_pred_31, best_return, buy_and_hold_cumulative_return = models.evaluar_modelos_regresion(
            modelos, df_test, X_test_scaled_, ticker_test)
        best_pred_31_final  = best_df_test['Close'].iloc[-1] * np.exp(best_pred_31)
        # Capturar el print de retornos con los valores correctos
        result_texts.append(
            f"El mejor modelo es {best_model} con un retorno acumulado de {best_return * 100:.2f}%")
        result_texts.append(
            f"Buy and Hold Cumulative Return: {buy_and_hold_cumulative_return * 100:.2f}%")
        result_texts.append(
            f"Precio de cierre del día 30 (real): {best_df_test['Close'].iloc[-1]:.2f}")
        result_texts.append(
            f"Precio de cierre del día 31 (predicho): {best_pred_31_final:.2f}")

        # Generamos las gráficas para regresión
        graph_urls += models.graficar_resultados_regresion(
            best_model, best_portfolio_value, best_buy_and_hold_value, best_df_test, best_pred_31)



    # Renderizamos la página con los resultados del backtesting
    return render_template('backtesting.html',
                           graph_urls=graph_urls,
                           result_texts=result_texts,
                           ticker_test=ticker_test)


if __name__ == '__main__':
    app.run(debug=True)
