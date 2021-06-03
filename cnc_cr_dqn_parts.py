import chex
import jax.numpy as jnp
import jax
import haiku as hk
from dqn_zoo import networks

Array = chex.Array
Numeric = chex.Numeric

FLOAT_TYPE =  jnp.float32



def qr_atari_network(num_actions: int, num_quantiles: int) -> networks.NetworkFn:
  """QR-DQN network, expects `uint8` input."""


  def net_fn(inputs):
    """Function representing CR-DQN Q-network."""
    network = hk.Sequential([
        networks.dqn_torso(),
        networks.dqn_value_head(num_quantiles * num_actions),
    ])
    network_output = network(inputs)
    q_dist = jnp.reshape(network_output, (-1, num_quantiles, num_actions))
    q_values = jnp.mean(q_dist, axis=1)
    q_values = jax.lax.stop_gradient(q_values)
    return networks.QRNetworkOutputs(q_dist=q_dist, q_values=q_values)

  return net_fn


def nc_qr_atari_network(num_actions: int, num_quantiles: int, 
                        n_layers: int, n_nodes: int ) -> networks.NetworkFn:

  def net_fn(inputs):
    """Function representing NC-CR-DQN Q-network."""

    torso_output = networks.dqn_torso()(inputs)

    N = num_quantiles

    layers_Q0_AMP = []
    layers_QPROP = []

    layers = [layers_Q0_AMP, layers_QPROP]

    for _ in range(n_layers):
      for l in layers:
        l += [networks.linear(n_nodes), jax.nn.relu]
        
    layers_Q0_AMP += [networks.linear(2*num_actions)]
    layers_QPROP += [networks.linear(N *num_actions)]

    network_output_Q0_AMP = hk.Sequential(layers_Q0_AMP)(torso_output)
    network_output_QPROP = hk.Sequential(layers_QPROP)(torso_output)
    
    # slice and reshape to have the action as the last dimension
    Rq = jnp.reshape(network_output_QPROP, (-1, N, num_actions))
    
    Q0 =  jnp.reshape(network_output_Q0_AMP[:,0:num_actions], (-1, 1, num_actions))
    AMP =  jnp.reshape(network_output_Q0_AMP[:,num_actions:], (-1, 1, num_actions))

    AMP = jax.nn.relu(AMP)
        
    Qprop = jax.nn.softmax(Rq, axis=1)

    Q = jnp.cumsum(Qprop,axis=1)


    Q *= AMP
   
    q_dist = Q +  Q0 
      
    q_values = jnp.mean(q_dist, axis=1)
    q_values = jax.lax.stop_gradient(q_values)
    return networks.QRNetworkOutputs(q_dist=q_dist, q_values=q_values)


  return net_fn

def symm_qr_atari_network(num_actions: int, num_quantiles: int, 
                        n_layers: int, n_nodes: int ) -> networks.NetworkFn:
  """CR-DQN network, expects `uint8` input."""


  def net_fn(inputs):
    """Function representing CR-DQN Q-network."""
   
    torso_output = networks.dqn_torso()(inputs)

   
    N_1 = num_quantiles - 1
    layers_Qmed = []
    layers_Qinc = []

    layers = [layers_Qmed, layers_Qinc]

    for _ in range(n_layers):
      for l in layers:
        l += [networks.linear(n_nodes), jax.nn.relu]
        
    layers_Qmed += [networks.linear(num_actions)]
    layers_Qinc += [networks.linear(N_1 *num_actions)]

    network_output_Qmed = hk.Sequential(layers_Qmed)(torso_output)
    network_output_Qinc = hk.Sequential(layers_Qinc)(torso_output)
    
    Qmed =  jnp.reshape(network_output_Qmed[:,:num_actions], (-1, 1, num_actions))

    halfN = N_1//2 ##it's fine if N-1 is not not even
    halfNup = N_1 - halfN


    Qleft = jnp.reshape(network_output_Qinc[:,:halfN*num_actions], (-1, halfN, num_actions))
    Qright = jnp.reshape(network_output_Qinc[:,halfN*num_actions:], (-1, halfNup, num_actions))

    Qleft = jax.nn.relu(Qleft)
    Qright = jax.nn.relu(Qright)
    
    Qleft = Qmed - jnp.cumsum(Qleft,axis=1)
    Qright = Qmed + jnp.cumsum(Qright,axis=1)
    q_dist = jnp.concatenate([jnp.flip(Qleft,axis=1),Qmed,Qright], axis = 1 )
      
    q_values = jnp.mean(q_dist, axis=1)
    q_values = jax.lax.stop_gradient(q_values)
    return networks.QRNetworkOutputs(q_dist=q_dist, q_values=q_values)


  return net_fn

def cramer_dist(
    dist_src: Array,
    dist_target: Array,
) -> Numeric:
    
  chex.assert_rank([dist_src, dist_target], 1)
  chex.assert_type([dist_src, dist_target], float)
       
  #num_quantiles
  n = dist_src.shape[0]

  tau_inc = jnp.ones(n, dtype=FLOAT_TYPE)/n

  y_diff = jnp.concatenate([-tau_inc,tau_inc],axis=0)
  Qs = jnp.concatenate([dist_src,dist_target],axis=0)

  (sorted_Qs, y_diff) = jax.lax.sort((Qs,y_diff))    


  x_diff = sorted_Qs[1:] - sorted_Qs[:-1]
  y_diff = jnp.cumsum(y_diff)[:-1]

  integs = y_diff * x_diff * y_diff
  return jnp.sum(integs)



def quantile_q_learning(
    dist_q_tm1: Array,
    #tau_q_tm1: Array,
    a_tm1: Numeric,
    r_t: Numeric,
    discount_t: Numeric,
    dist_q_t_selector: Array,
    dist_q_t: Array,
    q_values: Array,
) -> Numeric:
  """Implements Q-learning for quantile-valued Q distributions.
  Args:
    dist_q_tm1: Q distribution at time t-1.
    tau_q_tm1: Q distribution probability thresholds.
    a_tm1: action index at time t-1.
    r_t: reward at time t.
    discount_t: discount at time t.
    dist_q_t_selector: Q distribution at time t for selecting greedy action in
      target policy. This is separate from dist_q_t as in Double Q-Learning, but
      can be computed with the target network and a separate set of samples.
    dist_q_t: target Q distribution at time t.
  Returns:
    Quantile regression Q learning loss.
  """
  chex.assert_rank([
      dist_q_tm1, a_tm1, r_t, discount_t, dist_q_t_selector, dist_q_t
  ], [2, 0, 0, 0, 2, 2])
  chex.assert_type([
      dist_q_tm1, a_tm1, r_t, discount_t, dist_q_t_selector, dist_q_t
  ], [float, int, float, float, float, float])

  # Only update the taken actions.
  dist_qa_tm1 = dist_q_tm1[:, a_tm1]

    
  # Select target action according to greedy policy w.r.t. dist_q_t_selector.
  q_t_selector = q_values
      
  
  a_t = jnp.argmax(q_t_selector)

  dist_qa_t = dist_q_t[:, a_t]

  # Compute target, do not backpropagate into it.
  dist_target = r_t + discount_t * dist_qa_t
  dist_target = jax.lax.stop_gradient(dist_target)

  return cramer_dist(dist_qa_tm1, dist_target)



# Batch variant of quantile_q_learning with fixed tau input across batch.
_batch_quantile_q_learning = jax.vmap(
    quantile_q_learning, in_axes=(0, 0, 0, 0, 0, 0, 0))
