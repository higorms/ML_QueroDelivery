# Importando funções
from Extrator.extrator import extrair_dados
from Transformador.transformador import transformar_tabelas, preparar_tabela_ML
from Modelo.recomendacao import recommend


# Realiza a extração dos dados
df_visitas, df_pedidos, df_estabelecimentos = extrair_dados()

# Verifica se nenhum retorno é None antes de continuar
if (df_visitas is not None and 
    df_pedidos is not None and 
    df_estabelecimentos is not None):
    interacoes, unique_users, unique_items = preparar_tabela_ML(
        transformar_tabelas(df_visitas, df_pedidos, df_estabelecimentos)
    )

# Checa a quantidade de estabelecimentos para ordenar as recomendações e coleta o id do usuario que receberá a recomendação
tamanho = len(unique_items)
user_id = input('Informe o ID do usuário conforme a base de dados:')

# Verifica se o user_id existe na lista de usuários únicos
if user_id in unique_users:
    recomendacoes = recommend(user_id, unique_items, num_recommendations=tamanho)
    print(f"Recomendações para o usuário {user_id}: {recomendacoes}")
else:
    print("O usuário não existe na base de dados.")
    exit()  # Encerra o programa
