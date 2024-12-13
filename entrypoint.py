from Extrator.extrator import extrair_dados
from Transformador.transformador import transformar_tabelas, preparar_tabela_ML


# Realiza a extração dos dados
df_visitas, df_pedidos, df_estabelecimentos = extrair_dados()

# Verifica se nenhum retorno é None antes de continuar
if df_visitas is not None and df_pedidos is not None and df_estabelecimentos is not None:
    interacoes = preparar_tabela_ML(
                transformar_tabelas(df_visitas, df_pedidos, df_estabelecimentos)
            )



# TESTE 2

import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs
import matplotlib.pyplot as plt


# Criar um conjunto de dados de usuários e itens únicos
unique_users = interacoes['usuario_id'].unique()  # Extrai IDs únicos de usuários
unique_items = interacoes['estabelecimento_id'].unique()  # Extrai IDs únicos de estabelecimentos

# Definição do modelo de recomendação
class RecommenderModel(tfrs.Model):
    def __init__(self):
        super().__init__()
        # Embedding para usuários
        self.user_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(),  # Converte IDs de usuários em índices
            tf.keras.layers.Embedding(input_dim=len(unique_users) + 1, output_dim=128),  # Cria embeddings de 128 dimensões
            tf.keras.layers.Dense(64, activation='relu')  # Adiciona uma camada de ativação
        ])
        
        # Embedding para itens
        self.item_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(),  # Converte IDs de itens em índices
            tf.keras.layers.Embedding(input_dim=len(unique_items) + 1, output_dim=64)  # Cria embeddings de 64 dimensões
        ])
        
        # Adaptar os StringLookups com os dados únicos
        self.user_embedding.layers[0].adapt(unique_users)  # Ajusta o StringLookup para usuários
        self.item_embedding.layers[0].adapt(unique_items)  # Ajusta o StringLookup para itens
        
    def compute_loss(self, features, training=False):
        # Obtém embeddings para usuários e itens
        user_embeddings = self.user_embedding(features["usuario_id"])
        item_embeddings = self.item_embedding(features["estabelecimento_id"])
        
        # Obtém a quantidade de pedidos e converte para float32
        qtd_pedidos = tf.cast(features["qtd_pedidos"], tf.float32)  # Converte para float32
        
        # Calcula a perda ponderada usando erro quadrático médio
        loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)(user_embeddings, item_embeddings)
        
        # Pondera a perda pela quantidade de pedidos
        weighted_loss = loss * qtd_pedidos
        
        # Retorna a média da perda ponderada
        return tf.reduce_mean(weighted_loss)

# Instancia o modelo de recomendação
model = RecommenderModel()
# Compila o modelo com um otimizador
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

# Prepara os dados para o treinamento
train_data = tf.data.Dataset.from_tensor_slices({
    "usuario_id": interacoes["usuario_id"].values,  # IDs de usuários
    "estabelecimento_id": interacoes["estabelecimento_id"].values,  # IDs de estabelecimentos
    "qtd_pedidos": interacoes["qtd_pedidos"].values  # Quantidade de pedidos
}).shuffle(1000).batch(32)  # Embaralha e agrupa os dados em lotes de 32

# Treina o modelo com os dados de interações
history = model.fit(train_data, epochs=15, verbose=1)

# Função para fazer recomendações
def recommend(user_id, model, num_recommendations=5):
    # Obtém o vetor de embedding do usuário
    user_vector = model.user_embedding(tf.constant([user_id]))
    # Obtém os scores para todos os itens
    scores = model.item_embedding(tf.constant(unique_items))
    # Calcula a similaridade entre o vetor do usuário e os itens
    scores = tf.matmul(user_vector, scores, transpose_b=True)
    # Obtém os índices dos itens com as maiores pontuações
    top_indices = tf.argsort(scores, axis=1, direction='DESCENDING')[0][:num_recommendations]
    return unique_items[top_indices]  # Retorna os itens recomendados

# Exemplo de recomendação para um usuário específico
user_id = '5a67ba2ab85947003298770e'
recommendations = recommend(user_id, model)
print(f"Recomendações para o usuário {user_id}: {recommendations}")

# Plotar a perda durante o treinamento
plt.plot(history.history['loss'])
plt.title('Perda durante o treinamento')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.show()