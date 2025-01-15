# Importando bibliotecas e funções necessárias
from flask import Flask, request, jsonify
from Extrator.extrator import extrair_dados
from Transformador.transformador import transformar_tabelas, preparar_tabela_ML
from Modelo.recomendacao import recommend


app = Flask(__name__)   # Criando instância do Flask (classe)

# Realiza a extração dos dados e prepara as tabelas
df_visitas, df_pedidos, df_estabelecimentos = extrair_dados()

# Verifica se os dados foram extraídos corretamente
if df_visitas is not None and df_pedidos is not None and df_estabelecimentos is not None:
    # Prepara a tabela para Machine Learning a partir dos dados extraídos
    interacoes, unique_users, unique_items = preparar_tabela_ML(
        transformar_tabelas(df_visitas, df_pedidos, df_estabelecimentos)
    )

# Define a rota para obter recomendações
@app.route('/recomendacao', methods=['GET'])
def get_recommendations():
    # Obtém o ID do usuário a partir dos parâmetros da requisição
    user_id = request.args.get('user_id')
    
    # Verifica se o usuário existe na lista de usuários únicos
    if user_id in unique_users:
        tamanho = len(unique_items) # Obtém o número de itens únicos
        # Gera recomendações para o usuário
        recomendacoes = recommend(user_id, unique_items, num_recommendations=tamanho)
        # Retorna as recomendações em formato JSON
        return jsonify({"user_id":user_id, "recommendations":recomendacoes.tolist()})
    else:
        # Retorna um erro caso o usuário não exista
        return jsonify({"error": "O usuário não existe na base de dados."}), 404

# Executa a aplicação Flask em modo de depuração
if __name__ == '__main__':
    app.run(debug=True) 