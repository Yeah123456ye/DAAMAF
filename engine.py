
import tensorflow as tf
from keras.layers import MultiHeadAttention

class engine(tf.keras.Model):
    tf.keras.backend.set_floatx('float32')
    """ENGINE framework"""

    def __init__(self, N_o):
        super(engine, self).__init__()
        self.N_o = N_o
        print (self)
        """SNP Representation Module"""
        # Encoder network, Q
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(2802,)),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(units=500, activation='elu', kernel_regularizer='L1L2'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(units=100, activation=None, kernel_regularizer='L1L2'),
            ]
        )

        # Decoder network, P
        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(50,)),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(units=500, activation='elu', kernel_regularizer='L1L2'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(units=2802, activation=None, kernel_regularizer='L1L2'),
            ]
        )

        """Imaging Representation Module"""
        self.encoder_MRI = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(116,)),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(units=200, activation='elu', kernel_regularizer='L1L2'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(units=116, activation='elu', kernel_regularizer='L1L2')
            ]
        )

        self.encoder_PET = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(116,)),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(units=200, activation='elu', kernel_regularizer='L1L2'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(units=116, activation='elu', kernel_regularizer='L1L2')
            ]
        )

        """Generative Network"""
        # Generator network, G
        self.generator = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(54,)),
                tf.keras.layers.Dense(units=100, activation='elu', kernel_regularizer='L1L2'),
                tf.keras.layers.Dense(units=232, activation='sigmoid', kernel_regularizer='L1L2'),
            ]
        )

        # Discriminator network, D
        self.discriminator = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(116,)),
                tf.keras.layers.Dense(units=1, activation='sigmoid', kernel_regularizer='L1L2'),
            ]
        )

        """Diagnostician Module"""
        # Diagnostician network, C
        self.diagnostician_share = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(116,)),
                tf.keras.layers.Dense(units=25, activation='elu', kernel_regularizer='L1L2'),
            ]
        )

        self.diagnostician_clf = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(25,)),
                tf.keras.layers.Dense(units=self.N_o, activation='Softmax', kernel_regularizer='L1L2'),
            ]
        )

        self.diagnostician_reg = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(25,)),
                tf.keras.layers.Dense(units=1, activation=None, kernel_regularizer='L1L2'),
            ]
        )

    @tf.function
    # Reconstructed SNPs sampling
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(10, 50))
        return self.decode(eps, apply_sigmoid=True)

    # Represent mu and sigma from the input SNP
    def encode(self, x_SNP):
        mean, logvar = tf.split(self.encoder(x_SNP), num_or_size_splits=2, axis=1)
        return mean, logvar

    # Construct latent distribution
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.math.exp(logvar * .5) + mean

    # Reconstruct the input SNP     
    def decode(self, z_SNP, apply_sigmoid=False):
        logits = self.decoder(z_SNP)
        if apply_sigmoid:
            probs = tf.math.sigmoid(logits)
            return probs
        return logits

    def encode_MRI(self, x_MRI):
        zb_MRI = self.encoder_MRI(x_MRI)
        n, _ = zb_MRI.shape
        weight_c = 0.001
        coefficient = weight_c * tf.random.normal((n, n))
        y = tf.matmul(coefficient, zb_MRI)
        return zb_MRI, coefficient, y

    def encode_PET(self, x_PET):
        zb_PET = self.encoder_PET(x_PET)
        n, _ = zb_PET.shape
        weight_c = 0.001
        coefficient = weight_c * tf.random.normal((n, n))
        y = tf.matmul(coefficient, zb_PET)
        return zb_PET, coefficient, y

    def cross_modal_attention(self, MRI_att, PET_att):
        x = tf.expand_dims(MRI_att, axis=1)
        y = tf.expand_dims(PET_att, axis=1)
        a1 = MultiHeadAttention(num_heads=8, key_dim=50)(x, y)
        a2 = MultiHeadAttention(num_heads=8, key_dim=50)(y, x)
        MRI_PET = a1[:, 0, :]
        PET_MRI = a2[:, 0, :]
        return MRI_PET, PET_MRI

    # Attentive vector and fake imaging representation
    def generate(self, z_SNP, c_demo):
        z = tf.concat([c_demo, z_SNP], axis=-1)
        x_MRI_fake, a = tf.split(self.generator(z), num_or_size_splits=2, axis=1)
        return x_MRI_fake, a

    # Classify the real and the fake imaging representation
    def discriminate(self, x_MRI_real_or_fake):
        return self.discriminator(x_MRI_real_or_fake)

    # Disease diagnosis
    def diagnose(self, x_MRI, a, apply_logistic_activation=False):
        feature = self.diagnostician_share(tf.multiply(x_MRI, a))
        logit_clf = self.diagnostician_clf(feature)
        logit_reg = self.diagnostician_reg(feature)
        if apply_logistic_activation:
            y_hat = tf.math.softmax(logit_clf)
            s_hat = tf.math.sigmoid(logit_reg)
            return y_hat, s_hat
        return logit_clf, logit_reg
