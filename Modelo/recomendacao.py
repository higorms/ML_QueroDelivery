# Importando bibliotecas e funções
import tensorflow as tf
from Modelo.Treinamento import RecommenderModel  # Importa a classe

# Função para fazer recomendações
def recommend(user_id, unique_items, num_recommendations=5):
    # Carrega o modelo salvo do treinamento
    model = tf.keras.models.load_model('recomendacao_querodelivery.keras', 
                                       custom_objects={'RecommenderModel': RecommenderModel})
    # Obtém o vetor de embedding do usuário
    user_vector = model.user_embedding(tf.constant([user_id]))
    # Obtém os scores para todos os itens
    scores = model.item_embedding(tf.constant(unique_items))
    # Calcula a similaridade entre o vetor do usuário e os itens
    scores = tf.matmul(user_vector, scores, transpose_b=True)
    # Obtém os índices dos itens com as maiores pontuações
    top_indices = tf.argsort(scores, axis=1, direction='DESCENDING')[0][:num_recommendations]
    
    return unique_items[top_indices]  # Retorna os itens recomendados