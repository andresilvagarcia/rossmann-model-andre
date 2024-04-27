![image](https://user-images.githubusercontent.com/44026423/125207967-27f76b80-e266-11eb-9f03-de62c59f6699.png)

**Aviso:** O seguinte contexto é completamente fictício, a empresa, o contexto, o CEO e as questões de negócios.

A Rossmann planeja uma reforma geral em todas as suas lojas, porem atualmente a previsão de vendas é feitas através de uma planilha em excel considerando a média das vendas, achando essa metodologia falha e antiga, a administração da empresa busca novas soluções.

Foi realizada uma reunião com os gerentes, diretores e CEO da Rossmann, para planejarem as reformas nas lojas Rossmann a fim de renovar a estética e padronizar as lojas, para essa reforma ser possível a equipe de negócios da Rossmann precisaria contar com as previsões de vendas de todas as lojas para as proximas 6 semanas, porem como atualmente as previsões de vendas eram feitas somente por planilha de média, é impossível ter noção da grandeza de vendas individuais e oscilações de vendas entre cada loja. Diante disso a empresa decidiu contratar um cientista de dados, para ficar ciente de qual é a melhor solução para o problema.

Durante a reunião o cientista de dados explicou os detalhes de uma previsão de vendas, bem como os metódos que poderiam ser utilizados. Ao final da reunião ficou a cargo do cientista de dados montar um modelo de previsão de vendas com base em machine learning para responder a questão de negócio da empresa.

Questão de negócios é: **"Quanto cada loja venderá nas próximas seis semanas?"**

[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

# Business Assumptions
As seguintes suposições foram feitas sobre o problema de negócio:
- Para lojas que não tinham informação de CompetitionDistance, assumiu-se que a distancia seria 2 vezes maior que a maior distancia de um competidor mais próximo.
- Como foi assumido que existe competidor mesmo que muito longe, caso não haja a data em que o competidor abriu ou dados em relação aos periodos promocionais, trabalhase-se com a data da loja pensando na premissa que algumas variáveis derivadas do tempo são extremamente importantes para representar um comportamento.
- Os dados de Customers sendo difíceis prever foi descartado, podendo ser escopo para outro projeto complementar á este.
- Os dias em que as loja estavam fechadas foram descartados.
- Só foram consideradas entradas em que o valores de **Sales** fossem superiores a 0.

## **Lista de Atributos:**

| Atributos                        | Explicação                                                      |
| -------------------------------- | ------------------------------------------------------------ |
| Id                               | Um Id que representa uma dupla (Store, Date) dentro do conjunto de teste |
| Store                            | Um id único para cada loja                                   |
| Sales                            | Volume de vendas                         |
| Customers                        | Número de clientes em um determinado dia                       |
| Open                             | Indicador para saber se a loja estava aberta: 0 = fechada, 1 = aberta |
| StateHoliday                     | Indica um feriado estadual. Normalmente todas as lojas, com poucas exceções, fecham nos feriados estaduais. Observe que todas as escolas fecham nos feriados e finais de semana. a = feriado, b = feriado da Páscoa, c = Natal, 0 = Nenhum |
| SchoolHoliday                    | Indica se (Loja, Data) foi afetado pelo fechamento de escolas públicas |
| StoreType                        | Diferencia entre 4 modelos de loja diferentes: a, b, c, d  |
| Assortment                       | Descreve um nível de estoque: a = básico, b = extra, c = estendido |
| CompetitionDistance              | Distancia em metros do competidor mais proximo           |
| CompetitionOpenSince[Month/Year] | Dá o ano e mês aproximados em que o concorrente mais próximo foi aberto |
| Promo                            | Indica se uma loja está fazendo uma promoção naquele dia         |
| Promo2                           | Promo2 é uma promoção contínua e consecutiva para algumas lojas: 0 = a loja não está participando, 1 = a loja está participando |
| Promo2Since[Year/Week]           | Descreve o ano e a semana em que a loja começou a participar da Promo2. |
| PromoInterval                    | Descreve os intervalos consecutivos de início da promoção 2, nomeando os meses em que a promoção é iniciada novamente. |

# Estratégia de solução

Planejamento da solução:

**Etapa 01. Descrição dos dados:** O objetivo é usar métricas estatísticas para identificar dados fora do escopo do negócio.

**Etapa 02. Feature Engineering:** Derivar novos atributos com base nas variáveis originais para descrever melhor o fenômeno que será modelado.

**Passo 03. Filtragem de Dados:** Filtar linhas e selecionar colunas que não contenham informações para modelagem ou que não correspondam ao escopo do negócio.

**Etapa 04. Análise exploratória de dados:** Explorar os dados para encontrar insights e entender melhor o impacto das variáveis no aprendizado do modelo.

**Etapa 05. Preparação dos dados:** Preparar os dados para que os modelos de aprendizado de máquina possam aprender o comportamento específico.

**Etapa 06. Seleção de recursos:** Selecionar os atributos mais significativos para treinar o modelo.

**Etapa 07. Machine Learning Modelling:** Treinamento do modelo de aprendizado de máquina

**Etapa 08. Hyperparameter Fine Tunning:** Escolher os melhores valores para cada um dos parâmetros do modelo selecionado na etapa anterior.

**Etapa 09. Conversão do desempenho do modelo em valores de negócios:** Converter o desempenho do modelo em um resultado de negócios.

# Top 4 Data Insights

**Hipótese 9** - Lojas deveriam vender mais ao longo dos anos.

**FALSA**: Lojas vendem menos ao longo dos anos.

**Hipótese 10.** - Lojas deveriam vender mais no segundo semestre do ano.

**FALSA**: Lojas vendem menos no segundo semestre do ano.

**Hipótese 11.** - Lojas deveriam vender mais depois do dia 10 de cada mês.

**VERDADEIRA** Lojas vendem mais depois do dia 10 de cada mês.

**H12.** Lojas deveriam vender menos aos finais de semana.

**FALSA** Lojas vendem menos aos finais de semana.


#  Modelos de Machine Learning aplicados
Teste foram realizados usando os seguintes algoritmos:

**Average Model**

**Linear Regression Model**

**Linear Regression Regularized Model - Lasso**

**Random Forest Regressor**

**XGBoost Regressor**

# 6. Machine Learning Model Performance

**Single Performance**

| Model Name | MAE    | MAPE      | RMSE |
|-----------|---------|-----------|---------|
|  Random Forest Regressor  | 678.296634 | 0.099816   | 1008.248950 |
|  XGBoost Regressor	  | 805.016209 | 0.117862   | 1179.548085 |
|  Average Model  | 1354.800353 | 0.206400   | 1835.135542 |
|  Linear Regression	  | 1867.089774 | 0.292694   | 2671.049215 |
|  Linear Regression - Lasso	  | 1891.704881 | 0.289106	   | 2744.451737 |


**Real Performance - Cross Validation**

| Model Name | MAE CV   | MAPE CV      | RMSE CV |
|-----------|---------|-----------|---------|
|  Random Forest Regressor  | 838.18 +/- 218.74| 0.12 +/- 0.02  | 1256.87 +/- 319.67 |
|  XGBoost Regressor	  | 993.4 +/- 176.31 | 0.14 +/- 0.02   | 1431.48 +/- 244.42 |
|  Linear Regression	  | 2081.73 +/- 295.63 | 0.3 +/- 0.02   | 2952.52 +/- 468.37 |
|  Linear Regression - Lasso	  | 2116.38 +/- 341.5 | 0.29 +/- 0.01	   | 3057.75 +/- 504.26 |



Embora o modelo de Random Forest tenha demonstrado ser superior aos demais, em alguns casos esse modelo acaba por exigir muito espaço para ser publicado, acarretando um custo a mais para a empresa mante-lo funcionando. Diante disso o algoritmo escolhido foi o **XGBoost Regressor** que em sequência passou para a etapa de Hyperparameter Fine Tunning.


**Final Performance - Hyperparameter Fine Tunning Cross Validation**

Após encontrar os melhores parâmetros para o modelo através do metódo Random Search as métricas finais para o modelo foram as seguintes:

| Model Name | MAE CV   | MAPE CV      | RMSE CV |
|-----------|---------|-----------|---------|
|  XGBoost Regressor	  | 993.4 +/- 176.31 | 0.14 +/- 0.02   | 1431.48 +/- 244.42 |

# Desempenho do modelo em valores de negócios
Com base no atual Status Quo da empresa é possível analisar, a diferença de perfomance entre o atual modelo utilizado pela empresa (**Average Model**) e o modelo proposto pelo cientista de dados (**XGBoost Regressor**)

**Modelo atual baseado na média de vendas**

| Cenário | Valores   |                              
|-----------|---------|
|  Predições	  | R$ 243,723,748.12|

**Modelo com XGBoost Regressor**

| Cenário | Valores   |
|-----------|---------|
|  Predições	  | R$ 285,819,328.00 |
|  Pior Cenário	  | R$ 284,951,812.74 |
|  Melhor Cenário	  | R$ 286,686,822.41 |

#  Conclusão

Por fim é possível ter a compreensão que por mais que o modelo baseado em médias seja simples, ainda sim é um modelo coerente e coeso, embora ele não consiga analisar as oscilções de cada loja ele continua sendo uma opção dentro do leque das soluções.

O Modelo XGBoost para o primeiro ciclo (Metodologia CRISP-DM) apresentou um resultado dentro da faixa do aceitavel, embora algumas lojas se apresentaram difíceis de terem o comportamento previsto apresentando o MAPE (Mean Absolute Percenage Error) entre 0.32 a 0.58, esse primeiro resultado será apresentado para empresa, para informar em como está indo o projeto e o que ja se tem como solução.


#  Próximos Passos

Iniciar um segundo ciclo para analisar o problema buscando abordagens diferentes, tendo em vista principalmente as lojas com o comportamento difíceis de serem previsto.

Possíveis pontos para serem abordados no segundo ciclo:

-**Trabalhar com os dados NA de maneira diferente**

-**Rescaling e Encode dos dados com metodologias diferentes**

-**Trabalhar com novas Features para previsão**

-**Trabalhar com um método mais robusto para achar os melhores Hyper parametros para o modelo**

# Anotações Finais

**Remember that this project is related to the exercises carried out in the Comunidade DS (https://www.comunidadeds.com).**

**Lembrando que este projeto são relacionados a exercícios feitos na Comunidade DS (https://www.comunidadeds.com).**
