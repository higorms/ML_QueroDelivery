from Extrator.extrator import extrair_dados
from Transformador.transformador import transformar_tabelas


# Realiza a extração dos dados
df_visitas, df_pedidos, df_estabelecimentos = extrair_dados()

# Verifica se nenhum retorno é None antes de continuar
if df_visitas is not None and df_pedidos is not None and df_estabelecimentos is not None:
    transformar_tabelas(df_visitas, df_pedidos, df_estabelecimentos)

