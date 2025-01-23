import pandas as pd
import streamlit as st
import joblib


st.markdown("# **Predict Rio de Janeiro (Brazil) Airbnb Prices**")
# Carregando os dados
dados = pd.read_csv('dados.csv')
colunas = list(dados.columns)[1:-1]

# Inicializando variáveis
x_numericos = {'latitude': 0, 'longitude': 0, 'accommodates': 0, 'bathrooms': 0, 'bedrooms': 0, 'beds': 0, 'extra_people': 0,
               'minimum_nights': 0, 'ano': 0, 'mes': 0, 'n_amenitites': 0, 'host_listings_count': 0}

x_tf = {'host_is_superhost': 0, 'instant_bookable': 0}

x_listas = {'property_type': ['Apartment', 'Bed and breakfast', 'Condominium', 'Guest suite', 'Guesthouse', 'Hostel', 'House', 'Loft', 'Outros', 'Serviced apartment'],
            'room_type': ['Entire home/apt', 'Hotel room', 'Private room', 'Shared room'],
            'cancellation_policy': ['flexible', 'moderate', 'strict', 'strict_14_with_grace_period']}

# Retirando as colunas bed_type
dicionario = {}

# Gerar entradas para o Streamlit
for item in x_listas:
    for valor in x_listas[item]:
        dicionario[f'{item}_{valor}'] = 0

for item in x_numericos:
    if item == 'latitude' or item == 'longitude':
        valor = st.number_input(f'{item}', step=0.00001, value=0.0, format="%.5f")
    elif item == 'extra_people':
        valor = st.number_input(f'{item}', step=0.01, value=0.0)
    else:
        valor = st.number_input(f'{item}', step=1, value=0)
    x_numericos[item] = valor

for item in x_tf:
    valor = st.selectbox(f'{item}', ('Sim', 'Não'))
    x_tf[item] = 1 if valor == 'Sim' else 0

for item in x_listas:
    valor = st.selectbox(f'{item}', x_listas[item])
    dicionario[f'{item}_{valor}'] = 1

botao = st.button('Predict property value')

if botao:
    dicionario.update(x_numericos)
    dicionario.update(x_tf)
    valores_x = pd.DataFrame(dicionario, index=[0])

    # Verificar se todas as colunas necessárias estão presentes
    colunas_faltando = [col for col in colunas if col not in valores_x.columns]
    for col in colunas_faltando:
        valores_x[col] = 0  # Adiciona colunas ausentes com valor padrão 0

    # Filtrar somente as colunas que o modelo espera
    valores_x = valores_x[colunas]

    # Carregar o modelo
    modelo = joblib.load('modelo.joblib')

    # Fazer a previsão
    preco = modelo.predict(valores_x)
    st.write(f'R$ {preco[0]}')
