import tensorflow as tf

from tensorflow.keras.layers import (
    TextVectorization,
    Embedding,
    Dense,
    LayerNormalization,
    MultiHeadAttention,
    Dropout
)


# -----------------------------
# Model configuration
# -----------------------------

VOCAB_SIZE = 20000
MAX_LEN = 64
EMBED_DIM = 128
NUM_HEADS = 4
FF_DIM = 256
NUM_BLOCKS = 2
DROPOUT_RATE = 0.1


# -----------------------------
# Load vocabulary
# -----------------------------

with open("vectorizer_vocab.txt", "r", encoding="utf-8") as f:
    vocabulary = [
        line.rstrip("\n")
        for line in f
    ]


vectorizer = TextVectorization(
    vocabulary=vocabulary,
    output_mode="int",
    output_sequence_length=MAX_LEN
)


# -----------------------------
# Decoder Block
# -----------------------------

class DecoderOnlyBlock(tf.keras.layers.Layer):

    def __init__(
        self,
        embed_dim,
        num_heads,
        ff_dim,
        dropout_rate=0.1
    ):
        super().__init__()

        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads
        )

        self.ffn_dense_1 = Dense(
            ff_dim,
            activation="relu"
        )

        self.ffn_dense_2 = Dense(
            embed_dim
        )

        self.norm_1 = LayerNormalization()
        self.norm_2 = LayerNormalization()

        self.dropout_1 = Dropout(dropout_rate)
        self.dropout_2 = Dropout(dropout_rate)

    def call(
        self,
        inputs,
        attention_mask=None,
        training=False
    ):

        # Causal self-attention
        attention_output = self.attention(
            query=inputs,
            key=inputs,
            value=inputs,
            attention_mask=attention_mask
        )

        attention_output = self.dropout_1(
            attention_output,
            training=training
        )

        # Add & Norm
        x = self.norm_1(
            inputs + attention_output
        )

        # Feed Forward Network
        ffn_output = self.ffn_dense_1(x)
        ffn_output = self.ffn_dense_2(ffn_output)

        ffn_output = self.dropout_2(
            ffn_output,
            training=training
        )

        # Add & Norm
        x = self.norm_2(
            x + ffn_output
        )

        return x


# -----------------------------
# Decoder-Only Transformer
# -----------------------------

class DecoderOnlyTransformer(tf.keras.Model):

    def __init__(
        self,
        vocab_size,
        max_len,
        embed_dim,
        num_heads,
        ff_dim,
        num_blocks,
        dropout_rate=0.1
    ):
        super().__init__()

        self.token_embedding = Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim
        )

        self.position_embedding = Embedding(
            input_dim=max_len,
            output_dim=embed_dim
        )

        self.decoder_blocks = [
            DecoderOnlyBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout_rate=dropout_rate
            )
            for _ in range(num_blocks)
        ]

        self.final_norm = LayerNormalization()

        self.output_projection = Dense(
            vocab_size
        )

    def call(
        self,
        inputs,
        training=False
    ):

        seq_len = tf.shape(inputs)[1]

        # Padding mask
        padding_mask = tf.not_equal(
            inputs,
            0
        )

        padding_mask = padding_mask[
            :,
            tf.newaxis,
            :
        ]

        # Causal mask
        causal_mask = tf.linalg.band_part(
            tf.ones(
                (seq_len, seq_len),
                dtype=tf.bool
            ),
            -1,
            0
        )

        causal_mask = causal_mask[
            tf.newaxis,
            :,
            :
        ]

        # Combined mask
        attention_mask = tf.logical_and(
            causal_mask,
            padding_mask
        )

        # Token embedding
        token_vectors = self.token_embedding(
            inputs
        )

        # Position embedding
        positions = tf.range(
            start=0,
            limit=seq_len
        )

        position_vectors = self.position_embedding(
            positions
        )

        # Token + Position
        x = token_vectors + position_vectors

        # Decoder blocks
        for block in self.decoder_blocks:

            x = block(
                x,
                attention_mask=attention_mask,
                training=training
            )

        # Final normalization
        x = self.final_norm(x)

        # Vocabulary logits
        logits = self.output_projection(x)

        return logits


# -----------------------------
# Create model
# -----------------------------

model = DecoderOnlyTransformer(
    vocab_size=VOCAB_SIZE,
    max_len=MAX_LEN,
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    ff_dim=FF_DIM,
    num_blocks=NUM_BLOCKS,
    dropout_rate=DROPOUT_RATE
)


# -----------------------------
# Build model
# -----------------------------

dummy_input = tf.zeros(
    (1, MAX_LEN),
    dtype=tf.int32
)

model(
    dummy_input,
    training=False
)


# -----------------------------
# Load trained weights
# -----------------------------

model.load_weights(
    "decoder_transformer.weights.h5"
)

print("Transformer model loaded successfully.")


# -----------------------------
# Predict next token
# -----------------------------

def predict_next_token(text):

    tokens = vectorizer(
        [text]
    )[0]

    # Remove padding
    tokens = tokens[
        tokens != 0
    ]

    # Keep latest context
    tokens = tokens[
        -MAX_LEN:
    ]

    # Add batch dimension
    tokens = tf.expand_dims(
        tokens,
        axis=0
    )

    # Model prediction
    logits = model(
        tokens,
        training=False
    )

    # Get prediction for last real token
    last_logits = logits[
        0,
        -1,
        :
    ]

    # Most likely next token
    token_id = tf.argmax(
        last_logits
    ).numpy()

    return vocabulary[token_id]