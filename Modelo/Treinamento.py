# Importando bibliotecas e funções necessárias
import keras    # Biblioteca para modelo de deep learning
import tensorflow as tf     # Base do keras, que oferece as operações e estruturas dos modelos
import tensorflow_recommenders as tfrs  # Biblioteca do tensorflow especializada em modelos de recomendação
import matplotlib.pyplot as plt     # Biblioteca de visualização gráfica
from sklearn.model_selection import train_test_split    # Utilizado para divir os dados em teste e treino
import os   # Bilioteca do sistema operacional que permite manipulação de arquivos e diretórios


# Definição do modelos
@keras.saving.register_keras_serializable()
class RecommenderModel(tfrs.Model): # Classe que herda funções do trfs.model (modelo de recomendação)
    def __init__(self, unique_users, unique_items, **kwargs): 
        super().__init__(**kwargs)
        self.unique_users = unique_users    # Define os id's únicos dos usuários da base
        self.unique_items = unique_items    # Define os id's únicos dos estabelecimentos da base

        # Embedding para usuários
        self.user_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(),  # Converte IDs de usuários em índices númericos, já que o modelo não consegue lidar com strings para o aprendizado
            tf.keras.layers.Embedding(input_dim=len(unique_users) + 1, output_dim=8,    # Tamanho da tabela de embeddings e sua dimensão
                                      embeddings_regularizer=keras.regularizers.l2(1e-4)),      # Camada de regularização L2, para previnir overfitting do modelo
                                      keras.layers.Dropout(0.2)     # Camada de dropout com o mesmo propósito da regularização
        ])

        # Embedding para itens
        self.item_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(),  # Converte IDs de estabelecimentos em índices númericos, já que o modelo não consegue lidar com strings para o aprendizado
            tf.keras.layers.Embedding(input_dim=len(unique_items) + 1, output_dim=8,    # Tamanho da tabela de embeddings e sua dimensão
                                      embeddings_regularizer=keras.regularizers.l2(1e-4)),      # Camada de regularização L2, para previnir overfitting do modelo
                                      keras.layers.Dropout(0.2)     # Camada de dropout com o mesmo propósito da regularização
        ])

        # Aqui, os layers StringLookup são "treinados" para mapear os IDs únicos.
        self.user_embedding.layers[0].adapt(unique_users)
        self.item_embedding.layers[0].adapt(unique_items)

        # Define a tarefa de recomendação como um problema de regressão (ranking).
        self.task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM),
            metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')]        # Define o uso do Erro Quadrático Médio como métrica do modelo 
        )

        # Camadas densas adicionais para não-linearidade
        self.dense_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu'),  # Primeira camada densa
            tf.keras.layers.Dropout(0.2),                  # Dropout para regularização
            tf.keras.layers.Dense(8, activation='relu'),  # Segunda camada densa
            tf.keras.layers.Dense(1)                      # Camada final de previsão
        ])

    def compute_loss(self, features, training=False):
        # Obtém embeddings para usuários e itens (são os ids convertidos para formato númerico)
        user_embeddings = self.user_embedding(features["usuario_id"])   # Embedding do usuário.
        item_embeddings = self.item_embedding(features["estabelecimento_id"])       # Embedding do estabelecimento.

        # Combina os embeddings para verificar as relações entre pares
        combined_embeddings = tf.concat([user_embeddings, item_embeddings], axis=1)
        
        # Passa pelas camadas densas para introduzir não-linearidade
        predicted_score = self.dense_layers(combined_embeddings)
        predicted_score = tf.squeeze(predicted_score, axis=1)  # Remove dimensões extras

        # Converte a quantidade de pedidos para float32 (compatível com o TensorFlow).
        qtd_pedidos = tf.cast(features["qtd_pedidos"], tf.float32)
        
        # Calcula a perda usando a tarefa Ranking.
        return self.task(labels=qtd_pedidos, predictions=predicted_score)

    def get_config(self):   # Serializa os parâmetros necessários para salvar o modelo.
        config = super().get_config()
        config.update({
            'unique_users': self.unique_users.tolist(),
            'unique_items': self.unique_items.tolist()
        })
        return config

    @classmethod
    def from_config(cls, config): # Método para carregar o modelo a partir da configuração salva.
        unique_users = tf.convert_to_tensor(config['unique_users'])
        unique_items = tf.convert_to_tensor(config['unique_items'])
        return cls(unique_users, unique_items, **{k: v for k, v in config.items() if k not in ['unique_users', 'unique_items']})


def treinar_modelo(interacoes, unique_users, unique_items):
    # Instancia o modelo de recomendação
    model = RecommenderModel(unique_users, unique_items)

    # Compila o modelo
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

    # Divide os dados em treinamento e teste (80% treino e 20% teste)
    train_interacoes, val_interacoes = train_test_split(interacoes, test_size=0.2, random_state=42)

    # Prepara os dados de treino para TensorFlow
    train_data = tf.data.Dataset.from_tensor_slices({
        "usuario_id": train_interacoes["usuario_id"].values,
        "estabelecimento_id": train_interacoes["estabelecimento_id"].values,
        "qtd_pedidos": train_interacoes["qtd_pedidos"].values
    }).shuffle(1000).batch(32)

    # Prepara os dados de teste para TensorFlow
    val_data = tf.data.Dataset.from_tensor_slices({
        "usuario_id": val_interacoes["usuario_id"].values,
        "estabelecimento_id": val_interacoes["estabelecimento_id"].values,
        "qtd_pedidos": val_interacoes["qtd_pedidos"].values
    }).batch(32)

    # Early Stopping para parar o treino em caso do erro de validação começar a subir. Isso evita um overfitting com os dados de treino
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    # Treinamento
    history = model.fit(train_data, 
                        validation_data=val_data, 
                        callbacks=[early_stopping],
                        epochs=20, 
                        verbose=1)

    # Salva o modelo
    if os.path.exists('recomendacao_querodelivery.keras'):
        os.remove('recomendacao_querodelivery.keras')
    model.save('recomendacao_querodelivery.keras')

    # Plotando as curvas de RMSE
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['rmse'], label='RMSE - Treinamento', color='green')
    plt.plot(history.history['val_rmse'], label='RMSE - Validação', color='red')

    plt.title('Convergência do Modelo: RMSE')
    plt.xlabel('Épocas')
    plt.ylabel('Valores')
    plt.legend()
    plt.grid()
    plt.show()

    
