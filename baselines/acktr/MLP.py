import tensorflow as tf
from baselines.a2c.utils import fc, sample

class MlpPolicy(object):

   def __init__(self, sess, ob_space, ac_space, nenv=1, nsteps=10, nstack=1, task=1, reuse=False):
       nbatch = nenv*nsteps
       ns = ob_space.shape[0]
       ob_shape = (nbatch, ns * nstack)
       nact = 6
       X = tf.placeholder(tf.float32, ob_shape) #obs
       with tf.variable_scope('model' + str(task), reuse=reuse):
           h = fc(X, 'hidden', 256)
           logits = fc(h, 'logits', nact, act=lambda x:x)
           vf = fc(h, 'v', 1, act=lambda x:x)
           self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

       v0 = vf[:, 0]
       a0 = sample(logits)
       self.initial_state = [] #not stateful

       def step(ob, *_args, **_kwargs):
           a, v = sess.run([a0, v0], {X:ob})
           return a, v, [] #dummy state

       def value(ob, *_args, **_kwargs):
           return sess.run(v0, {X:ob})

       self.X = X
       self.logits = logits
       self.vf = vf
       self.step = step
       self.value = value