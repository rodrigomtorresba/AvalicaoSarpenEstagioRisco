# Avaliação Estágio - Risco - Sarpen Quant Investments
# Rodrigo Melo Torres Barros
# Dados: 'series_retornos.csv' Risk-free rate (Rf): 13,15%

# Importando pacotes que serão utilizados no código
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# Cria um DataFrame e carrega o arquivo com os dados da carteira, e o delimitador ';'. Declara a CDI de 13,15% como a Rf
dados_carteira = pd.read_csv('/Users/rodrigomtorres/Documents/Git/AnaliseCarteira/resources/series_retornos.csv', delimiter=';')
Rf = 0.1315 

def equal_weights(dados_carteira): 

    # Conta todas as colunas depois das colunas de data
    num_ativos = dados_carteira.iloc[:, 3:].shape[1] 

    # Retorna um array numpy com pesos iguais para todos os ativos
    return np.array ([1 / num_ativos] * num_ativos)

def calc_retorno_anualizado_carteira(dados_carteira, pesos): 
    # Verifica se a soma dos pesos é igual a 1 com aproximação de 8 casas decimais
    if not np.isclose(np.sum(pesos), 1.0, atol=1e-8):
        raise ValueError("A soma dos pesos da carteira não é igual a 1!")
    
    # Desconsiderando as colunas de data (year, month, day)
    retornos_diarios = dados_carteira.iloc[:, 3:]  

    # Calcula o retorno anualizado para cada ativo. (252 dias úteis em um ano contábil)
    retorno_anualizado = ((1 + retornos_diarios.mean())**252) - 1

    retorno_ponderado = retorno_anualizado * pesos
    return retorno_ponderado.sum() 

def calc_retorno_anualizado_serie(serie_retornos):

    # Seleciona a coluna 'retorno_diario_carteira'
    retornos_serie= serie_retornos['retorno_diario_carteira']
    # Anualiza os retornos
    retorno_anualizado = ((1 + retornos_serie.mean())**252) - 1

    return retorno_anualizado

def calc_volatilidade_anualizada_carteira(dados_carteira, pesos):
    # Verifica se a soma dos pesos é igual a 1.
    if not round(sum(pesos), 5) == 1.0:
        raise ValueError("A soma dos pesos da carteira não é igual a 1!")
    
    # Desconsiderando as colunas de data (year, month, day)
    retornos_diarios = dados_carteira.iloc[:, 3:]
    
    # Retorna o desvio padrão (volatilidade) da carteira
    return (np.dot(pesos.T, np.dot(retornos_diarios.cov() * 252, pesos))) ** 0.5

def calc_volatilidade_serie(serie_retornos):
    # Seleciona a coluna 'retorno_diario_carteira'
    retornos_carteira = serie_retornos['retorno_diario_carteira']
    
    # Calcula o desvio padrão
    return retornos_carteira.std() * (252 ** 0.5)

def calc_sharpe_ratio(dados_carteira, pesos, Rf):
    Ri = calc_retorno_anualizado_carteira(dados_carteira, pesos)
    Volatilidade = calc_volatilidade_anualizada_carteira(dados_carteira, pesos)
    return ((Ri - Rf)/Volatilidade)

def inverso_sharpe_ratio(dados_carteira, pesos, Rf=Rf):
    sharpe = calc_sharpe_ratio(dados_carteira, pesos, Rf)
    return -sharpe

def calc_retorno_diario_carteira(dados_carteira, pesos):

    dados_carteira_copy = dados_carteira.copy()
    num_ativos = len(pesos)

    # Unifica a data em uma só coluna
    dados_carteira_copy['date'] = pd.to_datetime(dados_carteira[['year', 'month', 'day']])
    
    # Calcula o retorno diário da carteira
    dados_carteira_copy['retorno_diario_carteira'] = (dados_carteira.iloc[:, 3:3+num_ativos] * pesos).sum(axis=1)
    
    # Retorna uma coluna de data e uma coluna com os retornos diários da carteira
    return dados_carteira_copy[['date', 'retorno_diario_carteira']]

def calc_retorno_acumulado(retornos_carteira, pesos=None):

    retornos_carteira_copy = retornos_carteira.copy()

    # Se 'retorno_diario_carteira' não existir no DataFrame que foi passado, calcula usando calc_retorno_diario_carteira
    # Dessa forma podemos usar o DataFrame original sem necessariamente calcular o retorno diário.
    if 'retorno_diario_carteira' not in retornos_carteira_copy.columns and pesos is not None:
        retornos_carteira_copy = calc_retorno_diario_carteira(retornos_carteira_copy, pesos)

    # Usa cumprod() do pandas para obter o produto acumulado, faz o rebase da carteira em 100
    retornos_carteira_copy['retorno_acumulado'] = 100 * (1 + retornos_carteira_copy['retorno_diario_carteira']).cumprod()

    return retornos_carteira_copy[['date', 'retorno_acumulado']]

def max_drawdown(retorno_acumulado):
   
    picos = retorno_acumulado['retorno_acumulado'].cummax()
    
    # Drawdown
    drawdown = (retorno_acumulado['retorno_acumulado'] / picos) - 1
    
    # Maior Drawdown
    return drawdown.min()

# Para maximizar o índice de Sharpe mudando os pesos podemos simplesmente minimizar o seu inverso usando algoritmos de otimização do scipy 
def otimizacao_pesos(dados_carteira):

    num_ativos = dados_carteira.iloc[:, 3:].shape[1]

    # Limites de peso entre 0 e 1
    bounds = [(0, 1) for asset in range(num_ativos)]

    # Soma dos pesos deve ser 1
    constraints = ({'type': 'eq', 'fun': lambda pesos: np.sum(pesos) - 1})  

    # Passa o cenário base como "benchmark"
    initial_guess = [1./num_ativos for asset in range(num_ativos)] 

    # Função auxiliar para facilitar no processo de otimização dos pesos
    def objetivo(pesos):
        return inverso_sharpe_ratio(dados_carteira, pesos)

    # Minimizando a função usando o algo SLSQP
    result_slsqp = minimize(objetivo, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

    # Arredonda os pesos para 6 casas decimais
    pesos_arredondados = np.round(result_slsqp.x, 6)

    return pesos_arredondados

def estrategia_melhor_acao_mensal(dados_carteira):

    # Copia o DataFrame e converte as colunas de data em uma só, e a seta como índice
    retornos_carteira = dados_carteira.copy()
    retornos_carteira['date'] = pd.to_datetime(retornos_carteira[['year', 'month', 'day']])
    retornos_carteira.set_index('date', inplace=True)
    retornos_carteira.drop(['year', 'month', 'day'], axis=1, inplace=True)

    # Calcula a média dos retornos diários para cada ação em cada mês e converte para taxas mensais
    media_retornos_diarios_mensais = retornos_carteira.resample('M').mean()
    retornos_mensais = (1 + media_retornos_diarios_mensais) ** 30 - 1

    # Identificando a ação com o maior retorno em cada mês
    acao_escolhida_mensal = retornos_mensais.idxmax(axis=1)

    # Shifta a série para escolher a melhor ação do para o mês seguinte
    acao_escolhida_mensal = acao_escolhida_mensal.shift(1).dropna()

    # Cria o DataFrame com o resultado da operação
    retornos_melhor_performer_mensal = pd.DataFrame(columns=['date', 'retorno_diario_carteira', 'ativo_investido_mes'])
    retornos_melhor_performer_mensal['date'] = retornos_carteira.index

    # Iterando pelos meses e preenchendo o DataFrame resultado com a ação escolhida para o mês seguinte
    for mes, acao_escolhida in acao_escolhida_mensal.items():
        mask_mes = (retornos_carteira.index.month == mes.month) & (retornos_carteira.index.year == mes.year)
        retornos_melhor_performer_mensal.loc[mask_mes, 'retorno_diario_carteira'] = retornos_carteira.loc[mask_mes, acao_escolhida].values
        retornos_melhor_performer_mensal.loc[mask_mes, 'ativo_investido_mes'] = acao_escolhida

    return retornos_melhor_performer_mensal

def analise_estrategia_mensal(dataframe_estrategia_mensal):
    # Garantindo que 'date' seja do tipo datetime
    dataframe_estrategia_mensal['date'] = pd.to_datetime(dataframe_estrategia_mensal['date'])
    
    # Criando uma coluna para representar o mês e o ano
    dataframe_estrategia_mensal['ano_mes'] = dataframe_estrategia_mensal['date'].dt.to_period('M')

    # Contando o número de meses únicos
    numero_de_meses = dataframe_estrategia_mensal['ano_mes'].nunique()

    # Contando quantos meses cada ativo foi escolhido
    ativos_escolhidos_por_mes = dataframe_estrategia_mensal.groupby('ativo_investido_mes')['ano_mes'].nunique()

    # Resultado
    resultado = {
        'numero_de_meses': numero_de_meses,
        'ativos_escolhidos_por_mes': ativos_escolhidos_por_mes
    }
    return resultado


retornos_estrategia_mensal = estrategia_melhor_acao_mensal(dados_carteira)
resultado_analise_estrategia_mensal = (analise_estrategia_mensal(retornos_estrategia_mensal))

# Resultados para a carteira com pesos iguais (cenário base)
pesos_iguais = equal_weights(dados_carteira)
retorno_anualizado_carteira_base = calc_retorno_anualizado_carteira(dados_carteira, pesos_iguais) * 100
volatilidade_carteira_base = calc_volatilidade_anualizada_carteira(dados_carteira, pesos_iguais) * 100
sharpe_carteira_base = calc_sharpe_ratio(dados_carteira, pesos_iguais, Rf) * 100
max_dd_carteira_base = max_drawdown(calc_retorno_acumulado(dados_carteira, pesos_iguais)) * 100
retorno_acumulado_carteira_base = calc_retorno_acumulado(dados_carteira, pesos_iguais)

# Resultados para a carteira otimizada
pesos_otimizados = otimizacao_pesos(dados_carteira)
retorno_anualizado_carteira_otimizada = calc_retorno_anualizado_carteira(dados_carteira, pesos_otimizados) * 100
volatilidade_carteira_otimizada = calc_volatilidade_anualizada_carteira(dados_carteira, pesos_otimizados) * 100
sharpe_carteira_otimizada = calc_sharpe_ratio(dados_carteira, pesos_otimizados, Rf) * 100
max_dd_carteira_otimizada = max_drawdown(calc_retorno_acumulado(dados_carteira, pesos_otimizados)) * 100
retorno_acumulado_carteira_otimizada = calc_retorno_acumulado(dados_carteira, pesos_otimizados)

# Resultados para a carteira com a estratégia de melhor performer mensal
peso_um = 1.0
retorno_anualizado_carteira_estrategia = calc_retorno_anualizado_serie(retornos_estrategia_mensal) * 100
volatilidade_carteira_estrategia = calc_volatilidade_serie(retornos_estrategia_mensal) * 100
sharpe_carteira_estrategia = ((retorno_anualizado_carteira_estrategia - (Rf*100))/volatilidade_carteira_estrategia)* 100
max_dd_carteira_estrategia = max_drawdown(calc_retorno_acumulado(retornos_estrategia_mensal)) * 100
retorno_acumulado_carteira_estrategia = calc_retorno_acumulado(retornos_estrategia_mensal)

resultados = {
    'Cenário': ['Base (pesos iguais)', 'Otimizada', 'Estratégia'],
    'Retorno Anualizado (%)': [retorno_anualizado_carteira_base, retorno_anualizado_carteira_otimizada, retorno_anualizado_carteira_estrategia],
    'Volatilidade Anualizada (%)': [volatilidade_carteira_base, volatilidade_carteira_otimizada, volatilidade_carteira_estrategia],
    'Índice de Sharpe (%)': [sharpe_carteira_base, sharpe_carteira_otimizada, sharpe_carteira_estrategia],
    'Drawdown Máximo (%)': [max_dd_carteira_base, max_dd_carteira_otimizada, max_dd_carteira_estrategia]
}

df_resultados = pd.DataFrame(resultados)
print(df_resultados)
file_name = "Resultados_Analise.csv"
df_resultados.to_csv(file_name, encoding='utf-8', index=False, sep=';')

# Função para criar o primeiro gráfico com os retornos acumulados
def plot_comp_retornos():
    plt.figure(1)
    
    # Plotando o retorno acumulado para a carteira no cenário base
    plt.plot(retorno_acumulado_carteira_base['date'], retorno_acumulado_carteira_base['retorno_acumulado'], label='Carteira do Cenário Base (pesos iguais)')
    
    # Plotando o retorno acumulado para a carteira otimizada
    plt.plot(retorno_acumulado_carteira_otimizada['date'], retorno_acumulado_carteira_otimizada['retorno_acumulado'], label='Carteira Otimizada')

    # Plotando o retorno acumulado para a carteira de estratégia do melhor performer de cada mês
    plt.plot(retorno_acumulado_carteira_estrategia['date'], retorno_acumulado_carteira_estrategia['retorno_acumulado'], label='Carteira com a Estratégia Proposta')

    # Adicionando título e rótulos
    plt.title('Comparação do Retorno Acumulado')
    plt.xlabel('Data')
    plt.ylabel('Retorno Acumulado (%)')
    plt.legend()

    # Rotacionando as etiquetas do eixo x, se necessário
    plt.xticks(rotation=45, ha='right')

plot_comp_retornos()

# Exibir os gráficos
plt.show()