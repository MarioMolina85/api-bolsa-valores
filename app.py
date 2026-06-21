import streamlit as st
import pandas as pd#importacion de las librerias para el analisis 
import yfinance as yf

import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
#reresentacion del titulo 
st.set_page_config(page_title="Análisis Bolsa de Valores",layout="wide")
#obtencion de las empresas o fondos de inversion o criptomonedas
def obtener_empresas():
    return {"Apple": "AAPL",  "Microsoft": "MSFT", "Amazon": "AMZN",  "Google (Alphabet)": "GOOGL",
            "Meta": "META",  "NVIDIA": "NVDA","Tesla": "TSLA", "Intel": "INTC", "AMD": "AMD","IBM": "IBM","JPMorgan": "JPM",
        "Bank of America": "BAC",
        "Visa": "V",
        "Mastercard": "MA",
        "ExxonMobil": "XOM",
        "Chevron": "CVX",
        "Coca Cola": "KO",
        "Pepsi": "PEP",
        "Nike": "NKE",
        "Walmart": "WMT",
        "S&P 500 ETF": "SPY",
        "Nasdaq ETF": "QQQ",
        "Dow Jones ETF": "DIA",
        "Bitcoin ETF": "BITO"
    }

#funcion para cargar la data 
@st.cache_data
def descargar_datos(ticker, fecha_inicio, fecha_fin):
    df = yf.download( ticker, start=fecha_inicio, end=fecha_fin, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df = df.reset_index()
    return df


#funcion para mostrar la tabala
def mostrar_tabla(df, ticker):
    st.subheader("📋 Datos")
    st.dataframe(df)
    st.write(f"Ticker: {ticker}")
    st.write(f"Registros: {len(df)}")

# ==========================================
# mostrando las estadisticas de la bolsa de valores
# ==========================================
def mostrar_estadisticas(df):
    st.subheader("📊 Estadísticas Descriptivas")
    st.dataframe(df.describe())
    st.subheader("📊 Analisis descriptivo")
    st.write("Veer los valores nulos",df.isnull().sum())
    st.write("")

#mostrando la grafica lineal para ver el tiempo y los precios de cierre
def grafico_cierre(df, ticker):
    fig = px.line( df, x="Date",  y="Close",  title=f"Precio de Cierre - {ticker}" )
    st.plotly_chart(fig, use_container_width=True)

#funcion para ver el volumne de transaciones realezada
def grafico_volumen(df, ticker):
    fig = px.bar(df,  x="Date",  y="Volume", title=f"Volumen - {ticker}")
    st.plotly_chart(fig, use_container_width=True)


def histograma_precio(df):
    fig = px.histogram(  df, x="Close", nbins=50,title="Distribución Precio Cierre" )
    st.plotly_chart(fig, use_container_width=True)

#funcion para ver el retorno
def histograma_retorno(df):
    df["Retorno"] = df["Close"].pct_change()
    fig = px.histogram( df, x="Retorno",nbins=50,title="Distribución de Retornos" )
    st.plotly_chart(fig, use_container_width=True)

#funcion para ver la correlacion de precios cierre, abiertos, volumnes, bajos
def matriz_correlacion(df):
    columnas = [ "Open","High","Low","Close",  "Volume"]
    corr = df[columnas].corr()
    st.subheader("🔗 Correlación")
    st.dataframe(corr)


def reporte_ydata(df, ticker):

    st.subheader("📑 YData Profiling")
    profile = ProfileReport( df,title=f"Perfil {ticker}",explorative=True )
    st_profile_report(profile)

def preparar_modelo(df):
    modelo = df.copy()
    modelo["Lag1"] = modelo["Close"].shift(1)
    modelo["Lag2"] = modelo["Close"].shift(2)
    modelo["Lag3"] = modelo["Close"].shift(3)

    modelo.dropna(inplace=True)
    X = modelo[["Lag1", "Lag2", "Lag3"]]
    y = modelo["Close"]
    return train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False
    )

#funcion para aplicar medias moviles
def medias_moviles(df):
    df["MA30"] = df["Close"].rolling(30).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()
    return df

def grafico_tendencia(df, ticker):
    fig = go.Figure()
    fig.add_trace( go.Scatter( x=df["Date"], y=df["Close"], name="Precio"))

    fig.add_trace(
        go.Scatter(x=df["Date"],y=df["MA30"],name="MA30") )

    fig.add_trace(
        go.Scatter(x=df["Date"], y=df["MA50"],name="MA50"))

    fig.add_trace(
        go.Scatter( x=df["Date"], y=df["MA200"],name="MA200") )

    fig.update_layout(title=f"Tendencia de {ticker}")
    return fig

def descomponer_serie(df):
    serie = df.set_index("Date")["Close"]
    resultado = seasonal_decompose( serie, model='additive', period=360 )
    return resultado

def show_componets_seriestime(df):
    resultado = descomponer_serie(df)
    st.subheader("Tendencia")
    st.line_chart(resultado.trend)
    st.subheader("Estacionalidad")
    st.line_chart(resultado.seasonal)
    st.subheader("Residuo")
    st.line_chart(resultado.resid)

def modelo_arima(df):
    serie = df["Close"].dropna()
    modelo = ARIMA(serie, order=(5,1,0))
    resultado = modelo.fit()
    predicciones = resultado.forecast(steps=30)
    return resultado, predicciones

def visualizar_modelosARIMA_SARIMA(df):
    st.subheader("📈 Pronóstico ARIMA")
    resultado_arima, pred_arima = modelo_arima(df)
    st.write(resultado_arima.summary())
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(   y=pred_arima, mode="lines",name="ARIMA"))
    fig.update_layout( title="ARIMA vs SARIMA")
    st.plotly_chart(fig,use_container_width=True)

def evaluar_modelo(nombre, y_real, y_pred):
    mae = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    r2 = r2_score(y_real, y_pred)
    st.write(f"### {nombre}")
    st.write(f"MAE : {mae:.4f}")
    st.write(f"RMSE: {rmse:.4f}")
    st.write(f"R²  : {r2:.4f}")


def ejecutar_modelos(df):

    st.subheader("Ver la Prediccion de los Datos en Modelo regresion linear y Random Forest")
    X_train, X_test, y_train, y_test = preparar_modelo(df)
    # Regresión Lineal
    model_linear = LinearRegression()
    model_linear.fit(X_train, y_train)
    pred_lr = model_linear.predict(X_test)
    evaluar_modelo( "Regresión Lineal", y_test,pred_lr)
    fig = px.line( title="Real vs Predicción" )
    fig.add_trace(
        go.Scatter( x=y_test.index, y=y_test,mode="markers", name="Datos reales",  marker=dict(size=6) )
    )

    # Predicción del modelo
    fig.add_trace(
        go.Scatter(
            x=y_test.index,
            y=pred_lr,
            mode="lines+markers",
            name="Predicción (Regresión Lineal)"
        )
    )

    fig.update_layout(
        title="📈 Regresión Lineal - Real vs Predicción",
        xaxis_title="Índice de tiempo",
        yaxis_title="Precio de cierre",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)


def mostrar_kpis_avanzados(df):

    retorno_diario = df["Close"].pct_change()

    rendimiento_anual = retorno_diario.mean() * 252 * 100

    riesgo_anual = retorno_diario.std() * np.sqrt(252) * 100

    sharpe = (
        rendimiento_anual / riesgo_anual
        if riesgo_anual != 0 else 0
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "📈 Rendimiento Anual",
            f"{rendimiento_anual:.2f}%"
        )

    with col2:
        st.metric(
            "⚠️ Riesgo Anual",
            f"{riesgo_anual:.2f}%"
        )

    with col3:
        st.metric(
            "🏆 Ratio Sharpe",
            f"{sharpe:.2f}"
        )

def mostrar_kpis(df):
    st.subheader("📊 Indicadores Clave (KPIs)")

    precio_actual = df["Close"].iloc[-1]
    precio_max = df["Close"].max()
    precio_min = df["Close"].min()

    retorno = (
        (df["Close"].iloc[-1] - df["Close"].iloc[0])
        / df["Close"].iloc[0]
    ) * 100

    volumen_promedio = df["Volume"].mean()

    volatilidad = (
        df["Close"].pct_change().std()
    ) * 100

    registros = len(df)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "💰 Precio Actual",
            f"${precio_actual:,.2f}"
        )

    with col2:
        st.metric(
            "📈 Precio Máximo",
            f"${precio_max:,.2f}"
        )

    with col3:
        st.metric(
            "📉 Precio Mínimo",
            f"${precio_min:,.2f}"
        )

    with col4:
        st.metric(
            "🔄 Retorno",
            f"{retorno:.2f}%"
        )

    col5, col6, col7 = st.columns(3)

    with col5:
        st.metric(
            "📊 Volumen Promedio",
            f"{volumen_promedio:,.0f}"
        )

    with col6:
        st.metric(
            "⚡ Volatilidad",
            f"{volatilidad:.2f}%"
        )

    with col7:
        st.metric(
            "📝 Registros",
            registros
        )
# ==========================================
# MAIN
# ==========================================

def main():

    st.title("📈 Análisis Bolsa de Valores")
    opciones = obtener_empresas()
    empresa = st.selectbox("Seleccione empresa", list(opciones.keys()))
    ticker = opciones[empresa]
    col1, col2 = st.columns(2)

    with col1:
        fecha_inicio = st.date_input(
            "Fecha Inicio",
            pd.to_datetime("2020-01-01")
        )

    with col2:
        fecha_fin = st.date_input(
            "Fecha Fin",
            pd.to_datetime("today")
        )

    if st.button("Consultar Datos"):

        df = descargar_datos( ticker, fecha_inicio,fecha_fin )
        
        if df.empty:
            st.error("No hay datos.")
            return

        mostrar_tabla(df, ticker)

        mostrar_estadisticas(df)
        mostrar_kpis(df)
        mostrar_kpis_avanzados(df)

        grafico_cierre(df, ticker)

        grafico_volumen(df, ticker)

        histograma_precio(df)

        histograma_retorno(df)
        
        show_componets_seriestime(df)
        st.subheader("📈 Tendencias con Medias Móviles")

        df = medias_moviles(df)
        fig_tendencia = grafico_tendencia(df, ticker)
        st.plotly_chart(
            fig_tendencia,
            use_container_width=True
        )
        matriz_correlacion(df)

        ejecutar_modelos(df)
        visualizar_modelosARIMA_SARIMA(df)

        reporte_ydata(df, ticker)


if __name__ == "__main__":
    main()