
## Rodando o Código:

### Primeiramente:

#### Alterar as seguintes linhas para seus respectivos diretórios:
```
diretorio_entrada = "./resources/"
diretorio_saida = "./output/"
```

#### Instalar requerimentos:
```
pip install -r requirements.txt
```

#### Executar o código:
```
python AnaliseCarteira.py
```
---

## Abordagem dos problemas propostos:

### Problema 1 - Foram criadas funções para a maioria das etapas do processo:

Assim facilita adaptar o código para novas _features_ no futuro:
- Retorno anualizado: (_calc_retorno_anualizado_carteira_) e (_calc_retorno_anualizado_serie_);
- Volatilidade anualizada: (_calc_volatilidade_anualizada_carteira_) e (_calc_volatilidade_anualizada_serie_);
- índice de Sharpe: (_calc_sharpe_ratio_carteira_), (_calc_sharpe_ratio_) e (_inverso_sharpe_ratio_);
- Máximo Drawdown: (_max_drawdown_) → calcula máximos acumulados, divide pela série e retorna o menor valor
#### **Que entregaram os resultados:**

| Cenário             | Retorno Anualizado (%) | Volatilidade Anualizada (%) | Índice de Sharpe (%) | Drawdown Máximo (%) |
|---------------------|------------------------|-----------------------------|----------------------|---------------------|
| Base                | 19.690974              | 25.084450                   | 26.075812            | -42.863114          |


#### **Gráfico do Retorno Acumulado Obtido:**
![Figure_1](https://github.com/rodrigomtorresba/AvalicaoSarpenEstagioRisco/assets/122982400/96244e9a-1a83-49da-be70-1de6b4c3aebd)

---
### Problema 2 - Para se obter um Índice de Sharpe maior que o da carteira no cenário base:
- Foi utilizado o algoritmo _Sequential Least Squares Programming_ (SLSQP) para minimizar a função inversa do Índice de Sharpe, com o limite de que cada peso deve variar entre 0 e 1, e a restrição de que a soma dos pesos seja 1 (representando a totalidade do capital investido.)
-  A função _otimizacao_pesos_ recebe um DataFrame que segue os padrões do arquivo fornecido: 3 colunas de data seguidas das séries de retorno, e retorna um array com os pesos otimizados encontrados.

#### O algoritmo utilizado retornou os seguintes pesos:
```
 A:0.0,
 B:0.0,
 C:0.0,
 D:0.050464,
 E:0.949536
```

#### **Que entregaram os resultados:**

| Cenário             | Retorno Anualizado (%) | Volatilidade Anualizada (%) | Índice de Sharpe (%) | Drawdown Máximo (%) |
|---------------------|------------------------|-----------------------------|----------------------|---------------------|
| Base                | 19.690974              | 25.084450                   | 26.075812            | -42.863114          |
| Otimizada           | 31.772901              | 30.908849                   | 60.251035            | -47.353132          |

#### **Gráfico Comparando os Retornos Acumulados Obtidos:**
![2](https://github.com/rodrigomtorresba/AvalicaoSarpenEstagioRisco/assets/122982400/8f5c3449-7fdb-4052-b032-845f7208cc43)

---
### Problema 3: 

**A Função _estrategia_melhor_acao_mensal_ faz todo o _heavy lifting_ para a estratégia proposta**:
- Copia o DataFrame passado e converte as colunas de data em uma só
- Calcula a média dos retornos diários para cada ação em cada mês e converte para taxas mensais
- Identifica a ação com o maior retorno em cada mês
- Move a série para escolher a melhor ação do mês para o mês seguinte
- Cria um novo DataFrame
- Itera pelos meses e preenche o DataFrame criado com a ação escolhida para o mês seguinte
- Retorna o DataFrame criado


>**Como o código é estruturado em funções, não foi difícil calcular os outros indicadores para essa estratégia também.**
#### Resultados obtidos:

| Cenário             | Retorno Anualizado (%) | Volatilidade Anualizada (%) | Índice de Sharpe (%) | Drawdown Máximo (%) |
|---------------------|------------------------|-----------------------------|----------------------|---------------------|
| Base                | 19.690974              | 25.084450                   | 26.075812            | -42.863114          |
| Otimizada           | 31.772901              | 30.908849                   | 60.251035            | -47.353132          |
| Estratégia Proposta | 18.088005              | 36.585238                   | 13.497260            | -59.215019          |       |
#### **Gráfico Comparando os Retornos Acumulados Obtidos:**
![3](https://github.com/rodrigomtorresba/AvalicaoSarpenEstagioRisco/assets/122982400/317bc199-343f-4ee7-9489-0578afbb8ab1)

---
### Extra:

**Comentando apenas uma linha da função que calcula a carteira da estratégia dos melhores _performers_ de cada mês podemos chegar numa situação onde o gestor “adivinha” qual será o ativo com maior retorno em cada mês, e investe 100% naquele ativo, naquele mês**:

É exatamente o que a função _extra_melhor_acao_mensal_mes0_ faz:

```
#acao_escolhida_mensal = acao_escolhida_mensal.shift(1).dropna()
```

**Dessa forma, a função retorna um portfolio com retorno acumulado cômico:**
#### **Gráfico Comparando os Retornos:**
![4](https://github.com/rodrigomtorresba/AvalicaoSarpenEstagioRisco/assets/122982400/ebd576e4-3ef2-4b0a-86f1-8172a28cefcb)

---

## Pacotes Python Utilizados:

```
pandas
numpy
matplotlib.pyplot
scipy.optimize minimize
```

