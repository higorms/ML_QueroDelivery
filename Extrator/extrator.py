# Importando biblioteca
import pandas as pd

# Definindo função que fará a extração de dados necessários dos arquivos .csv
def extrair_dados():
    try:
        df_estabelecimentos = pd.read_csv("Dados/base_estabelecimentos.csv")
        df_pedidos = pd.read_csv("Dados/base_pedidos.csv")
        df_visitas = pd.read_csv("Dados/base_visitas.csv")
        
        return df_visitas, df_pedidos, df_estabelecimentos
    except FileNotFoundError as e:
        print(f"Arquivo não encontrado: {e}")
    except pd.errors.EmptyDataError as e:
        print(f"Arquivo vazio: {e}")
    except pd.errors.ParserError as e:
        print(f"Erro ao ler o arquivo: {e}")