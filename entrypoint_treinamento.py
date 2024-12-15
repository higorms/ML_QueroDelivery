from Extrator.extrator import extrair_dados
from Transformador.transformador import transformar_tabelas, preparar_tabela_ML
from Modelo.Treinamento import treinar_modelo


# Realiza a extração dos dados
df_visitas, df_pedidos, df_estabelecimentos = extrair_dados()

# Verifica se nenhum retorno é None antes de continuar
if (df_visitas is not None and 
    df_pedidos is not None and 
    df_estabelecimentos is not None):
    interacoes, unique_users, unique_items = preparar_tabela_ML(
        transformar_tabelas(df_visitas, df_pedidos, df_estabelecimentos)
    )

treinar_modelo(interacoes)