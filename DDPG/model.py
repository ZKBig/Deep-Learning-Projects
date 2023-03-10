import paddle.fluid as fluid
import parl
from parl import layers


class Model(parl.Model):
    def __init__(self, act_dim):
        self.actor_model = ActorModel(act_dim)
        self.critic_model = CriticModel()

    def policy(self, obs):
        return self.actor_model.policy(obs)

    def value(self, obs, act):
        return self.critic_model.value(obs, act)

    def get_actor_params(self):
        # it returns a list, which contains the names of the parameters.
        return self.actor_model.parameters()


class ActorModel(parl.Model):
    def __init__(self, act_dim):
        hid1_size = 300
        hid2_size = 600

        self.fc1 = layers.fc(size=hid1_size, act='relu')
        self.fc2 = layers.fc(size=hid2_size, act='relu')
        self.fc3 = layers.fc(size=act_dim, act='tanh')

    def policy(self, obs):
        hid1 = self.fc1(obs)
        hid2 = self.fc1(hid1)
        means = self.fc2(hid2)
        return means


class CriticModel(parl.Model):
    def __init__(self):
        hid1_size = 300
        hid2_size = 600

        self.fc1 = layers.fc(size=hid1_size, act='relu')
        self.fc2 = layers.fc(size=hid2_size, act='relu')
        self.fc3 = layers.fc(size=1, act=None)

    def value(self, obs, act):
        concat = layers.concat([obs, act], axis=1)
        hid1 = self.fc1(concat)
        hid2 = self.fc2(hid1)
        Q = self.fc2(hid2)
        Q = layers.squeeze(Q, axes=[1])
        return Q
