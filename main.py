from flask import Flask, request, jsonify
from Extrator.extrator import extrair_dados
from Transformador.transformador import transformar_tabelas, preparar_tabela_ML
from Modelo.recomendacao import recommend


app = Flask(__name__)

# Realiza a extração dos dados e prepara as tabelas
df_visitas, df_pedidos, df_estabelecimentos = extrair_dados()

if df_visitas is not None and df_pedidos is not None and df_estabelecimentos is not None:
    interacoes, unique_users, unique_items = preparar_tabela_ML(
        transformar_tabelas(df_visitas, df_pedidos, df_estabelecimentos)
    )

@app.route('/recomendacao', methods=['GET'])
def get_recommendations():
    user_id = request.args.get('user_id')
    
    if user_id in unique_users:
        tamanho = len(unique_items)
        recomendacoes = recommend(user_id, unique_items, num_recommendations=tamanho)
        return jsonify({"user_id": user_id, "recommendations": recomendacoes.tolist()})
    else:
        return jsonify({"error": "O usuário não existe na base de dados."}), 404

if __name__ == '__main__':
    app.run(debug=True) 