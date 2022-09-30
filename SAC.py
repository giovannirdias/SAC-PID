# Importar bibliotecas e funcoes
import tensorflow as tf
import buffer
import tensorflow.keras as keras
import tensorflow_probability as tfp
import os

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

class policy():
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
        y = tf.math.tanh(normal) # Action

        log_prob = normal.log_prob(normal)
        log_prob -= tf.math.log(1-tf.math.pow(y, 2) + 1e-6) 
        log_prob = log_prob.sum(1, keepdims=True)
        mean = tf.math.tanh(mean)
        return y, log_prob, mean

class SAC():
    def __init__(self, env, gamma, config, buffer_size, learning_rate):
        self.state_dim = self.env.state_dim
        self.action_dim = self.env.action_dim

        self.gamma = gamma
        self.tau = config.getfloat('SAC', 'tau')
        self.alpha = config.getfloat('SAC', 'alpha')
        self.target = configcallin.getint('SAC', )
        self.hidden_dim = config,getint('SAC', 'hidden_dim')

        self.critic = Q_Function(self.state_dim, self.action_dim, self.hidden_dim)
        self.policy = Policy(config, self.state_dim, self.action_dim, self.hidden_dim)
        self.policy_optimizer = keras.optimizers.Adam(self.policy.parameters(), learning_rate)
        self.buffer = buffer(buffer_size)

    # Tomada de decisao determinando a action atraves do state
    def select_action(self, state):
        # Tensor com os estados do robo no periodo t
        state = tf.Tensor(state, dtype = tf.float32)
        action, _, _ = self.policy.sample(state)
        return tf.cast(action, dtype=tf.float32)[0]

    def update_parameters(self, batch_size, updates):
        # Recoler uma amostra da memoria buffer
        # State -> sta, Action -> act, Reward -> rwd, Next State -> Nsta
        # aval -> Avaliacao do objetivo
        sta, act, rwd, Nsta, aval = self.buffer.sample(batch_size)

        # Tensores dos paramentros
        sta = tf.Tensor(sta, dtype=tf.float32)
        Nsta = tf.Tensor(Nsta, dtype=tf.float32)
        act = tf.Tensor(act, dtype=tf.float32)
        rwd = tf.Tensor(rwd, dtype=tf.float32)
        aval = tf.Tensor(aval)

        # Rede sem backward
        with tf.stop_gradient():
            state_action, state_log = self.policy.sample(Nsta)
            q1_tar, q2_tar = self.critic(Nsta, state_action)
            q_min = tf.math.minimum(q1_tar, q2_tar) - self.alṕha * state_log
            q_value = rwd + (1-aval)*self.gamma*(q_min)
        
        # Paper: duas funcoes Q para evitar vies na otimizacao da politica otima
        q1, q2 = self.critic(sta, act)
        q1_loss = keras.losses.MeanSquaredError()(q1, q_value).np()  # Value Network
        q2_loss = keras.losses.MeanSquaredError()(q2, q_value).np()  # Critic Network
        q_loss = q1_loss + q2_loss

        # Calculo dos gradientes (backward) (??Duvida??)
        critic.compile(loss = q_loss, metrics=["mae"])
        
        # Politica
        pi, log_pi, _ = self.policy.sample(sta)
        q1_pi, q2_pi = self.critic(sta, pi)
        min_q_pi = tf.math.minimum(q1_pi, q2_pi)

        policy_loss = ((self.alpha * log_pi) - min_q_pi).np().mean() # Actor Network

        # Calculo dos gradientes (backward) da politica (??Duvida??)
        policy.compile(optimizer=self.policy_optimizer, loss = policy_loss, metrics=["mae"])

    def save_model(self, dir):
        model.save_weights(dir)
        model.save(dir)
    
    def load_model(self, dir)
        model.load_weights(dir)
