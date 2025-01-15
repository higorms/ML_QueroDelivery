<h1 align="center"> Projeto de Deep Learning (Processo Seletivo QueroDelivery) </h1>

<img alt="Static Badge" src="https://img.shields.io/badge/VS_Code-007ACC?logo=visualstudiocode"> <img alt="Static Badge" src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white"> <br>
<img alt="Static Badge" src="https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white"> <img alt="Static Badge" src="https://img.shields.io/badge/Tensor_Flow-FF6F00?logo=tensorflow&logoColor=white"> <img alt="Static Badge" src="https://img.shields.io/badge/Flask-3caabf?logo=flask&logoColor=white">
 <br>

<br>
<p>&nbsp;&nbsp;&nbsp;&nbsp;Projeto consiste em construir um pipeline de dados e uma rede neural de recomendação baseado na experiência do cliente dentro do aplicativo QueroDelivery.</p>

<p>&nbsp;&nbsp;&nbsp;&nbsp;Na pasta raiz temos os código de entrypoints, que servem como inicializadores (gatilhos) de cada fluxo de dados. No entrypoint_treinamento é onde o pipeline é inicializado extraindo 
os dados dos arquivos .csv, aplicando as devidas transformações e implementando tais dados preparados no treinamento de um modelo neural.</p>
<p>&nbsp;&nbsp;&nbsp;&nbsp;O main.py é o código onde a API feita pelo flask é inicializada permitindo que o usuário consulte o modelo da rede neural já treinado, tendo como entrada para a API o ID do usuário e retornando a lista de recomendações ordenada.</p>
<p>&nbsp;&nbsp;&nbsp;&nbsp;Na pasta Extrator fica o código responsável por fazer a extração dos dados de dentro dos arquivos .csv que estão contidos na pasta Dados, e armazená-los em dataframes do pandas.</p>
<p>&nbsp;&nbsp;&nbsp;&nbsp;Na pasta Transformador temos o código que faz a transformação do dataframe obtido, agregando os dados necessários em um dataframe único, organizando-os de forma que seja compreensivel para 
um aprendizado de modelo.</p>
<p>&nbsp;&nbsp;&nbsp;&nbsp;Na pasta Modelo temos os códigos que fazem uso do tensorflow para criar e gerenciar o modelo neural. O arquivo "Treinamento.py" contêm todas as funções e parâmetros necessários para o treinamento do modelo 
a partir dos dataframne gerado pelas funções de transformação. Esse arquivo também gera uma saída gráfica através do matplotlib exibindo métricas de erro para auxiliar nos ajustes do modelo.</p>
<p>&nbsp;&nbsp;&nbsp;&nbsp;Ainda na pasta Modelo temos o código "recomendacao.py". Esse código contêm a função que carrega o modelo treinado no arquivo anterior, e faz as recomendações para o usuário escolhido.</p>
<p>&nbsp;&nbsp;&nbsp;&nbsp;O arquivo "recomendacao_querodelivery.keras" é onde o modelo fica armazenado, permitindo que ele seja carregado e utilizado para gerar as recomendações de estabelecimentos para cada usuário.</p>


