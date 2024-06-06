# Relatório Final: Otimização de Precificação de Ativos de Renda Fixa Sintética

## Contexto e Importância

No competitivo setor financeiro, a precisão na precificação de ativos é fundamental para a integridade do mercado e a confiança dos investidores. O desafio enfrentado pelo BTG Pactual na precificação de ativos sintéticos de renda fixa, que simulam o comportamento de títulos de renda fixa através de derivativos, destaca a complexidade de gerir produtos financeiros inovadores em um ambiente de mercado rápido e volátil. A otimização da rentabilidade desses ativos em relação ao Certificado de Depósito Interbancário (CDI) exige uma análise detalhada de dados de mercado.

## Problema

O núcleo do problema estava em identificar as combinações ótimas de negociações à vista e à termo que maximizem a rentabilidade dos ativos sintéticos, mantendo o risco dentro de limites aceitáveis. Isso implicava na necessidade de uma solução capaz de explorar sistematicamente um vasto espaço de soluções possíveis, adaptando-se continuamente às mudanças do mercado.

## Descrição do Problema e Abordagem de Solução

O projeto em questão tem como objetivo a precificação de um ativo denominado "sintético de renda fixa". Este ativo é composto por uma posição vendida em termo de ação, combinada com uma posição comprada na mesma ação objeto do termo. A finalidade primordial consiste em encontrar combinações ótimas de negociações à vista de ações que correspondam aos contratos a termo, com a intenção de balancear a rentabilidade do ativo sintético em torno de 100% do CDI (Certificado de Depósito Interbancário).

No entanto, caso a equalização em torno de 100% do CDI seja inviável, o projeto busca alternativas que resultem em uma rentabilidade próxima a 100%, preferencialmente entre 98% e 104%, evitando valores extremos que possam distorcer a realidade financeira. Para atingir tal propósito, propõe-se a aplicação de técnicas de aprendizado por reforço, empregando redes neurais para otimizar o processo de identificação das combinações ideais.

A entrega final do projeto consistirá em um Jupyter Notebook contendo a modelagem do ambiente, agente, estados, ações e função de recompensa. O ambiente é concebido como o espaço onde as transações de compra e venda estão disponíveis, sendo onde o agente aprenderá seus comportamentos e tomará suas ações. As ações dentro do modelo compreendem o ato de casar, não casar ou realizar um descasamento. Os estados representam a situação atual de combinação de negociações, funcionando como um instante do ambiente em um determinado momento. Abaixo está listado de modo mais detalhado como é realizado a escolha de um casamento.

1. Determinação da quantidade a ser casada: A quantidade a ser casada é decidida como o menor valor entre a quantidade pendente para completar o termo e a quantidade disponível para ser casada daquela compra. Por exemplo, se existem 1000 unidades da compra disponíveis, mas apenas 800 são necessárias para completar o termo, então serão casadas apenas 800 unidades.

2. Atualização da quantidade pendente: Após determinar a quantidade casada, é necessário diminuir essa quantidade da variável que representa a quantidade pendente para completar o termo.

3. Registro do casamento: A quantidade casada é então inserida na coluna "matched" da tabela de compras, para manter um registro claro das transações realizadas.

4. Atualização da disponibilidade: Por fim, é necessário diminuir a quantidade casada da coluna "available_quantity" na tabela de compras, indicando que essas unidades já foram comprometidas.

Após a execução dos passos acima para realizar o casamento entre as negociações à vista e a termo, o processo de escolha do casamento ideal pode ser detalhado da seguinte maneira:

1. Calcular a média ponderada para o step atual, utilizando as transações realizadas até o momento.

2. Calcular a diferença entre a média ponderada do step atual e a média ponderada do step anterior, o que resulta no valor de A_escolhida.

3. Calcular a diferença entre a média ponderada do step atual para todas as ações e identificar qual resultado se aproxima mais do preço ideal, definido como A_melhor.

4. Aplicar os resultados obtidos na função linear, utilizando a fórmula R_max \* (A_escolhida / A_melhor), onde R_max é a recompensa máxima definida.

Normalizar as recompensas aplicando o resultado da função linear na tangente hiperbólica.

Esses passos visam garantir que o agente aprenda a tomar decisões de casamento que otimizem a rentabilidade do ativo sintético de renda fixa, buscando sempre se aproximar do objetivo de 100% do CDI ou de uma rentabilidade próxima a esse valor. A integração dessas etapas no processo de escolha do casamento contribui para a eficácia do modelo de aprendizado por reforço proposto no projeto.

### Técnicas de Aprendizado por Reforço Utilizadas

Algoritmos de RL: Especificamente, a implementação utilizou Deep Q-Networks (DQN) para aprender políticas de decisão que maximizam o 100% ao 104% em relação ao CDI.
Treinamento e Avaliação Iterativa: O modelo foi exposto a dados históricos de mercado e simulações, aprendendo a identificar as combinações de negociações mais rentáveis.

## Resultados Obtidos

### Impacto na Precificação e Operações

O gráfico fornecido ilustra a distribuição dos Valores de DI alcançados pelo modelo proposto. Observa-se que a maioria dos resultados concentra-se abaixo do valor de DI = 1, que seria o benchmark de 100% do CDI. Isso indica que, enquanto a solução de Aprendizado por Reforço desenvolvida ofereceu insights valiosos e possíveis melhorias operacionais, ela não conseguiu superar consistentemente a solução de precificação já existente.

![Distribuição dos Valores de DI](/artefatos/img/DistribuiçãoDosValoresDeDI.PNG)

## Benefícios Estratégicos e Operacionais

Eficiência Operacional: A abordagem proposta permitiu uma análise mais rápida dos ativos, embora a precisão da precificação não tenha alcançado o patamar desejado.
Mitigação de Riscos: A maior compreensão dos padrões de mercado tem o potencial de mitigar riscos, mas requer ajustes adicionais para alcançar o nível de confiabilidade necessário.

### Conclusão

A introdução de métodos baseados em Aprendizado por Reforço no processo de precificação de ativos de renda fixa sintética do BTG Pactual representou um avanço na busca por soluções inovadoras. No entanto, os resultados obtidos até o momento indicam que a solução proposta não supera a abordagem atual. É necessário um exame mais aprofundado dos parâmetros do modelo, da estrutura de dados de entrada e da adaptação do algoritmo para otimizar ainda mais o processo de precificação.
