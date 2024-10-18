# Atividade-Supervisionada-Otimiza-o-de-Sistemas
Este trabalho tem como foco a construção de um algoritmo de otimização baseado no  problema da mochila para montar uma carteira de investimentos. 
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Coletar dados históricos dos ativos
def coletar_dados(tickers, inicio, fim):
    dados = yf.download(tickers, start=inicio, end=fim)
    dados.to_csv('data/raw_data.csv')
    return dados

# Função de retorno esperado
def retorno_carteira(pesos, retornos):
    return np.sum(retornos.mean() * pesos) * 252

# Função de risco (volatilidade)
def volatilidade_carteira(pesos, retornos):
    return np.sqrt(np.dot(pesos.T, np.dot(retornos.cov() * 252, pesos)))

# Função objetivo para minimização (razão risco/retorno)
def funcao_objetivo(pesos, retornos):
    return volatilidade_carteira(pesos, retornos) / retorno_carteira(pesos, retornos)

# Otimização da carteira
def otimizar_carteira(retornos, tickers):
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(len(tickers)))
    initial_weights = np.array([1 / len(tickers)] * len(tickers))
    optimal_portfolio = minimize(funcao_objetivo, initial_weights, args=(retornos,), method='SLSQP', bounds=bounds, constraints=constraints)
    return optimal_portfolio.x

# Visualização da Fronteira Eficiente
def plotar_fronteira_eficiente(retornos, pesos_otimos, tickers):
    retornos_arr = []
    volatilidade_arr = []
    
    for _ in range(1000):
        pesos = np.random.random(len(tickers))
        pesos /= np.sum(pesos)
        retornos_arr.append(retorno_carteira(pesos, retornos))
        volatilidade_arr.append(volatilidade_carteira(pesos, retornos))
    
    plt.figure(figsize=(10, 7))
    plt.scatter(volatilidade_arr, retornos_arr, c='blue', marker='o')
    plt.scatter(volatilidade_carteira(pesos_otimos, retornos), retorno_carteira(pesos_otimos, retornos), c='red', marker='x')
    plt.xlabel('Risco (Volatilidade)')
    plt.ylabel('Retorno Esperado')
    plt.title('Fronteira Eficiente')
    plt.grid(True)
    plt.show()

# Visualização da Distribuição de Ativos
def plotar_distribuicao_ativos(pesos_otimos, tickers):
    plt.figure(figsize=(10, 7))
    plt.pie(pesos_otimos, labels=tickers, autopct='%1.1f%%', startangle=140)
    plt.title('Distribuição de Ativos')
    plt.show()

def main():
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    inicio = '2020-01-01'
    fim = '2023-01-01'
    
    # Coleta de dados
    dados = coletar_dados(tickers, inicio, fim)
    retornos = dados['Adj Close'].pct_change().dropna()
    
    # Otimização
    pesos_otimos = otimizar_carteira(retornos, tickers)
    print("Pesos ótimos:", pesos_otimos)
    
    # Visualização
    plotar_fronteira_eficiente(retornos, pesos_otimos, tickers)
    plotar_distribuicao_ativos(pesos_otimos, tickers)

if __name__ == "__main__":
    main()

