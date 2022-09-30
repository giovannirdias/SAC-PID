# Importar bibliotecas e funcoes
import tensorflow as tf
import buffer
import tensorflow.keras as keras
import tensorflow_probability as tfp

# Q-function utilizando redes neurais
class Q_Function():
    def __init__(self, n_inputs, n_actions, hidden_dim):
        super(Q_Function, self).__init__
        # Modelos de rede para ator e critico
        actor_model = create_model(n_inputs, n_actions, hidden_dim)
        critic_model = create_model(n_inputs, n_actions, hidden_dim)

    def create_model(self, n_inputs, n_actions, hidden):
        model = keras.models.Sequential()
        model.add(Dense(hidden, input_shape = (n_inputs + n_actions, ), activation='relu', kernel_initializer = 'he_normal'))
        model.add(Dense(hidden, activation='relu', kernel_initializer = 'he_normal'))    
        model.add(Dense(1))

        return model

class Policy():
    def __init__(self, configs, n_inputs, n_actions, hidden_dim):
        super(Policy, self).__init__
        mean = create_model(n_inputs, n_actions, hidden_dim)
        log_std = create_model(n_inputs, n_actions, hidden_dim)

    def create_model(self, n_inputs, n_actions, hidden):
        model = keras.models.Sequential()
        model.add(Dense(hidden, input_shape = (n_inputs, ), activation='relu'))
        model.add(Dense(hidden, activation='relu'))
        model.add(Dense(n_actions))
        return model
        
    def sample(self, state):
        mean = self.mean
        std = self.log_std
        normal = tfp.distributions.Normal(mean, std)
        y = tf.math.tanh(normal)

        log_prob = normal.log_prob(normal)
        log_prob -= tf.math.log(1-tf.math.pow(y, 2) + 1e-6) 
        log_prob = log_prob.sum(1, keepdims=True)
        mean = tf.math.tanh(mean)
        return log_prob, mean

class SAC():
    def __init__(self, env, gamma, config, buffer_size, learning_rate):
        self.state_dim = self.env.state_dim
        self.action_dim = self.env.action_dim

        self.gamma = gamma
        self.tau = config.getfloat('SAC', 'tau')
        self.alpha = config.getfloat('SAC', 'alpha')
        self.target = config.getint('SAC', )
        self.hidden_dim = config,getint('SAC', 'hidden_dim')

        self.critic = Q_Function(self.state_dim, self.action_dim, self.hidden_dim)
        self.policy = Policy(config, self.state_dim, self.action_dim, self.hidden_dim)
        self.policy_optimizer = keras.optimizers.Adam(self.policy.parameters(), learning_rate)
        self.buffer = buffer(buffer_size)


                

