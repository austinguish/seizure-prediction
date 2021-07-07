import tensorflow as tf

class FullyConnectedNet():

    @staticmethod
    def build_network(
                    input_dim,
                    units1,
                    units2,
                    units3,
                    dropout1,
                    dropout2,
                    dropout3,
                    learning_rate,
                    multi_layer,
                    l2_1,
                    l2_2,
                    l2_3,
                    kernel_init1,
                    kernel_init2,
                    kernel_init3
                   ):

        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.Dense(units=units1, input_dim=input_dim, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_1), kernel_initializer=kernel_init1))
        model.add(tf.keras.layers.Dropout(dropout1))

        model.add(tf.keras.layers.Dense(units=units2, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_2), kernel_initializer=kernel_init2))
        model.add(tf.keras.layers.Dropout(dropout2))

        if multi_layer:
            model.add(tf.keras.layers.Dense(units=units3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_3), kernel_initializer=kernel_init3))
            model.add(tf.keras.layers.Dropout(dropout3))

        model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

        opt = tf.keras.optimizers.Adam(lr=learning_rate)

        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

        return model