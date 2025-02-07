from tensorflow import keras as k
from tensorflow.keras import layers

# Not commented on purpose.

class DBTransformer(k.Model):
    def __init__(self, 
                 data_shape,
                 num_heads=8,
                 embed_dim=8,
                 ff_dim=256,
                 num_tr_blocks=4,
                 activation='gelu',
                 dropout=0.1,
                 regularizer=k.regularizers.l2(1e-5)):
        super(DBTransformer, self).__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"Embedding dimension ({embed_dim}) must be divisible by number of heads ({num_heads})"
            )
        
        self.inputs = k.Input(shape=data_shape[1:], name="Input")
        self.norm_layer = k.layers.Normalization()
        self.norm_layer.adapt(data_shape)
        
        self.reshape = layers.Reshape((data_shape[1], data_shape[2], 1))
        self.embedding = layers.Dense(embed_dim)
        
        self.r_transformer_blocks = [self._create_transformer_block(embed_dim, num_heads, ff_dim, activation, dropout, regularizer, axis=1) for _ in range(num_tr_blocks)]
        self.c_transformer_blocks = [self._create_transformer_block(embed_dim, num_heads, ff_dim, activation, dropout, regularizer, axis=2) for _ in range(num_tr_blocks)]
        
        self.merge = layers.Concatenate(axis=-1)
        
        self.mlp_1 = layers.Dense(2 * ff_dim, activation=activation, kernel_regularizer=regularizer)
        self.dropout_1 = layers.Dropout(dropout)
        self.batch_norm_1 = layers.BatchNormalization()
        
        self.mlp_2 = layers.Dense(ff_dim, activation=activation, kernel_regularizer=regularizer)
        self.dropout_2 = layers.Dropout(dropout)
        self.batch_norm_2 = layers.BatchNormalization()
        
        self.global_pool = layers.GlobalAveragePooling2D()
        self.output_layer = layers.Dense(1, activation='sigmoid')
    
    def _create_transformer_block(self, embed_dim, num_heads, ff_dim, activation, dropout, regularizer, axis):
        def transformer_block(x):
            attention = layers.MultiHeadAttention(num_heads, key_dim=embed_dim, attention_axes=(axis))(x, x)
            if dropout:
                attention = layers.Dropout(dropout)(attention)
            x = layers.Add()([x, attention])
            x = layers.LayerNormalization(epsilon=1e-6)(x)

            ff = layers.Dense(ff_dim, activation=activation, kernel_regularizer=regularizer)(x)
            ff = layers.Dense(embed_dim)(ff)
            if dropout:
                ff = layers.Dropout(dropout)(ff)
            x = layers.Add()([x, ff])
            x = layers.LayerNormalization(epsilon=1e-6)(x)
            return x
        return transformer_block
    
    def call(self, inputs):
        x = self.norm_layer(inputs)
        x = self.reshape(x)
        x = self.embedding(x)
        
        row_x = x
        for block in self.r_transformer_blocks:
            row_x = block(row_x)
        
        column_x = x
        for block in self.c_transformer_blocks:
            column_x = block(column_x)
        
        merged = self.merge([row_x, column_x])
        
        x = self.mlp_1(merged)
        x = self.dropout_1(x)
        x = self.batch_norm_1(x)
        
        x = self.mlp_2(x)
        x = self.dropout_2(x)
        x = self.batch_norm_2(x)
        
        x = self.global_pool(x)
        return self.output_layer(x)
