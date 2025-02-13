import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Cargar el modelo y el escalador guardados
modelo_knn = joblib.load('modelo_knn.bin')
escalador = joblib.load('escalador.bin')

# Nombres de las columnas que usó el modelo durante el entrenamiento
columnas = ['edad', 'colesterol']

# Título de la aplicación
st.title('Asistente IA para Cardiólogos')
st.markdown('### Realizado por Juan Pablo Pérez Bayona')

# Introducción de la aplicación
st.write("""
    Esta aplicación utiliza un modelo de inteligencia artificial para predecir si una persona tiene problemas cardíacos basándose en sus factores de riesgo.
    Los factores que se toman en cuenta son la edad y el colesterol, y con estos valores el modelo predictivo determina si la persona tiene problemas cardíacos o no.
""")

# Crear las pestañas
tabs = st.tabs(['Captura de Datos', 'Predicción'])

with tabs[0]:
    st.header('Captura de Datos')

    # Crear un formulario para la entrada de los datos
    edad = st.slider('Edad', 18, 80, 40)  # Edad entre 18 y 80 años
    colesterol = st.slider('Colesterol', 50, 600, 200)  # Colesterol entre 50 y 600

    # Guardar los datos en el estado de la aplicación (esto lo podemos hacer en esta pestaña)
    st.session_state.edad = edad
    st.session_state.colesterol = colesterol

    st.write(f"Edad: {edad}")
    st.write(f"Colesterol: {colesterol}")

with tabs[1]:
    st.header('Predicción de Problemas Cardíacos')

    # Verificar si los datos fueron capturados
    if hasattr(st.session_state, 'edad') and hasattr(st.session_state, 'colesterol'):
        edad = st.session_state.edad
        colesterol = st.session_state.colesterol

        # Crear un DataFrame con los datos de entrada y las columnas correspondientes
        datos_entrada = pd.DataFrame([[edad, colesterol]], columns=columnas)

        # Preprocesar la entrada
        datos_normalizados = escalador.transform(datos_entrada)

        # Realizar la predicción con el modelo KNN
        prediccion = modelo_knn.predict(datos_normalizados)

        # Mostrar la predicción
        if prediccion == 0:
            st.write("**Resultado**: La persona **NO tiene problemas cardíacos**.")
            st.image('https://s28461.pcdn.co/wp-content/uploads/2017/07/Tu-corazo%CC%81n-consejos-para-mantenerlo-sano-y-fuerte.jpg', caption="Corazón sano")
        else:
            st.write("**Resultado**: La persona **TIENE problemas cardíacos**.")
            st.image('https://as01.epimg.net/deporteyvida/imagenes/2017/10/28/portada/1509177885_209365_1509178036_noticia_normal.jpg', caption="Problema cardíaco")
    else:
        st.warning("Por favor, capture los datos primero en la pestaña 'Captura de Datos'.")
