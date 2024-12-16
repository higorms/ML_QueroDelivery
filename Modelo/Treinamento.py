import keras
import tensorflow as tf
import tensorflow_recommenders as tfrs
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os


# Definição do modelo de recomendação
@keras.saving.register_keras_serializable()
class RecommenderModel(tfrs.Model):
    def __init__(self, unique_users, unique_items, **kwargs):
        super().__init__(**kwargs)
        self.unique_users = unique_users
        self.unique_items = unique_items

        # Embedding para usuários
        self.user_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(),  # Converte IDs de usuários em índices
            tf.keras.layers.Embedding(input_dim=len(unique_users) + 1, output_dim=8,
                                      embeddings_regularizer=keras.regularizers.l2(1e-4)),
                                      keras.layers.Dropout(0.2)
        ])

        # Embedding para itens
        self.item_embedding = tf.keras.Sequential([
            tf.keras.layers.StringLookup(),  # Converte IDs de itens em índices
            tf.keras.layers.Embedding(input_dim=len(unique_items) + 1, output_dim=8,
                                      embeddings_regularizer=keras.regularizers.l2(1e-4)),
                                      keras.layers.Dropout(0.2)
        ])

        # Adaptar os StringLookups com os dados únicos
        self.user_embedding.layers[0].adapt(unique_users)
        self.item_embedding.layers[0].adapt(unique_items)

        # Métrica e perda
        self.task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM),
            metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')]
        )

    def compute_loss(self, features, training=False):
        # Obtém embeddings para usuários e itens
        user_embeddings = self.user_embedding(features["usuario_id"])
        item_embeddings = self.item_embedding(features["estabelecimento_id"])

        # Produto entre embeddings (previsão)
        predicted_score = tf.reduce_sum(user_embeddings * item_embeddings, axis=1)

        # Obtém a quantidade de pedidos e converte para float32
        qtd_pedidos = tf.cast(features["qtd_pedidos"], tf.float32)
        
        # Calcula a perda usando a tarefa Ranking
        return self.task(labels=qtd_pedidos, predictions=predicted_score)

    def get_config(self):
        config = super().get_config()
        config.update({
            'unique_users': self.unique_users.tolist(),
            'unique_items': self.unique_items.tolist()
        })
        return config

    @classmethod
    def from_config(cls, config):
        unique_users = tf.convert_to_tensor(config['unique_users'])
        unique_items = tf.convert_to_tensor(config['unique_items'])
        return cls(unique_users, unique_items, **{k: v for k, v in config.items() if k not in ['unique_users', 'unique_items']})


def treinar_modelo(interacoes, unique_users, unique_items):
    # Instancia o modelo de recomendação
    model = RecommenderModel(unique_users, unique_items)

    # Compila o modelo
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

    # Divide os dados em treinamento e validação
    train_interacoes, val_interacoes = train_test_split(interacoes, test_size=0.2, random_state=42)

    # Prepara os dados para TensorFlow
    train_data = tf.data.Dataset.from_tensor_slices({
        "usuario_id": train_interacoes["usuario_id"].values,
        "estabelecimento_id": train_interacoes["estabelecimento_id"].values,
        "qtd_pedidos": train_interacoes["qtd_pedidos"].values
    }).shuffle(1000).batch(32)

    val_data = tf.data.Dataset.from_tensor_slices({
        "usuario_id": val_interacoes["usuario_id"].values,
        "estabelecimento_id": val_interacoes["estabelecimento_id"].values,
        "qtd_pedidos": val_interacoes["qtd_pedidos"].values
    }).batch(32)

    # Early Stopping
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

    plt.title('Convergência do Modelo: Perda e RMSE')
    plt.xlabel('Épocas')
    plt.ylabel('Valores')
    plt.legend()
    plt.grid()
    plt.show()

    
