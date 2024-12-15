import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs
import matplotlib.pyplot as plt


def treinar_modelo(interacoes):
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
    history = model.fit(train_data, epochs=6, verbose=1) # Com esse formato salvo na variavel history consigo visualizar a convergência via gráfico

    # Salvar o modelo
    model.save('recomendacao_querodelivery.keras')

    # Plotar a perda durante o treinamento
    plt.plot(history.history['loss'])
    plt.title('Perda durante o treinamento')
    plt.xlabel('Épocas')
    plt.ylabel('Perda')
    plt.show()