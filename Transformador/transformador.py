import pandas as pd


# Função que tranforma os dataframes extraidos  dos arquivos .csv em um dataframe unico com as informações necessarias para o treinamento do modelo
def transformar_tabelas(df_visitas, df_pedidos, df_estabelecimentos):
    # Faz o agrupamento dos usuarios e estabelecimentos no dataframe de acordo com o número de visitas
    df_qtd_visitas = df_visitas.copy()
    df_qtd_visitas['qtd_visitas'] = df_qtd_visitas.groupby(['usuario_id', 'estabelecimento_id'])['visita_id'].transform('count')
    df_qtd_visitas = pd.DataFrame(df_qtd_visitas.groupby(['usuario_id', 'estabelecimento_id'])['qtd_visitas'].mean())

    # Faz o agrupamento dos usuarios e estabelecimentos no dataframe de acordo com o número de pedidos
    df_qtd_pedidos = df_pedidos.copy()
    df_qtd_pedidos['qtd_pedidos'] = df_qtd_pedidos.groupby(['usuario_id', 'estabelecimento_id'])['pedido_id'].transform('count')
    df_qtd_pedidos = pd.DataFrame(df_qtd_pedidos.groupby(['usuario_id', 'estabelecimento_id'])['qtd_pedidos'].mean())

    # Reune os dois dataframes em um unico, compilando assim todas as informações
    df_qtd = pd.merge(df_qtd_visitas, df_qtd_pedidos, how = 'outer', on = ['usuario_id', 'estabelecimento_id'])
    df_qtd = df_qtd.reset_index()

    # Tratamento das lacunas do merge para os clientes que visitaram a pagina do estabelecimento mas nunca fizeram pedidos (0)
    df_qtd = df_qtd.fillna({'qtd_visitas':0, 'qtd_pedidos':0})

    # Reune o dataframe com as informações dos estabelecimento como: nota e categoria
    df_qtd = pd.merge(df_qtd, df_estabelecimentos, how = 'left', on = 'estabelecimento_id')

    # Definição do tipo de variável dentgro do dataframe
    df_qtd = df_qtd.astype({
    'qtd_visitas':int,
    'qtd_pedidos':int
    })
    
    return df_qtd

# Função que faz a seleção dos atributos que serão realmente necessário dentro do treinamento do modelo
def preparar_tabela_ML(df_qtd):
    interacoes = df_qtd[['usuario_id', 'estabelecimento_id', 'qtd_pedidos']]
    # Criar um conjunto de dados de usuários e itens únicos
    unique_users = interacoes['usuario_id'].unique()  # Extrai IDs únicos de usuários
    unique_items = interacoes['estabelecimento_id'].unique()  # Extrai IDs únicos de estabelecimentos
    
    return interacoes, unique_users, unique_items