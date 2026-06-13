import tensorflow as tf
import sionna
from sionna.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.utils import BinarySource
from sionna.mapping import Mapper, Demapper
from sionna.channel import AWGN

class LdpcSimulation(tf.keras.Model):
    def __init__(self, k, n, num_bits_per_symbol):
        super().__init__()
        # 1. Initialize the components from your imports
        self.source = BinarySource()
        self.encoder = LDPC5GEncoder(k, n)
        self.mapper = Mapper(constellation_type="qam", num_bits_per_symbol=num_bits_per_symbol)
        self.channel = AWGN()
        self.demapper = Demapper(demapping_method="app", constellation_type="qam", num_bits_per_symbol=num_bits_per_symbol)
        self.decoder = LDPC5GDecoder(self.encoder, num_iter=20) # 20 belief propagation iterations

    @tf.function(jit_compile=True) # Enables fast XLA execution on GPU/CPU
    def call(self, batch_size, ebno_db):
        # Calculate noise variance based on Eb/No and code rate
        coderate = self.encoder.k / self.encoder.n
        no = sionna.utils.ebnodb2no(ebno_db, num_bits_per_symbol=4, coderate=coderate)
        
        # 2. Forward Pass Pipeline
        bits = self.source([batch_size, self.encoder.k])       # Generate message bits
        codeword = self.encoder(bits)                          # 5G LDPC Encoding
        x = self.mapper(codeword)                              # Modulate to QAM symbols
        y = self.channel([x, no])                              # Add AWGN noise
        llr = self.demapper([y, no])                           # Demap to Log-Likelihood Ratios
        decoded_bits = self.decoder(llr)                       # 5G LDPC Decoding
        
        return bits, decoded_bits

# --- Execution Parameters ---
# Define code dimensions (k=information bits, n=total codeword bits)
k = 1000
n = 2000 
num_bits_per_symbol = 4 # 16-QAM

# Instantiate and run the simulation model
sim = LdpcSimulation(k=k, n=n, num_bits_per_symbol=num_bits_per_symbol)
tx_bits, rx_bits = sim(batch_size=32, ebno_db=4.0)

# Calculate Bit Error Rate (BER)
ber = tf.reduce_mean(tf.cast(tx_bits != rx_bits, tf.float32))
print(f"Simulation completed successfully!")
print(f"Bit Error Rate (BER) at 4dB Eb/No: {ber.numpy():.5f}")
