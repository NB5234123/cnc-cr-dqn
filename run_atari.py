"""A CNC-CR-DQN agent training on Atari.
"""

# pylint: disable=g-bad-import-order

import collections
import itertools
import sys
import typing

from absl import app
from absl import flags
from absl import logging
import dm_env
import haiku as hk
import jax
from jax.config import config
import jax.numpy as jnp
import numpy as np
import optax

from dqn_zoo import atari_data
from dqn_zoo import gym_atari
from dqn_zoo import networks
from dqn_zoo import parts
from dqn_zoo import processors
from dqn_zoo import replay as replay_lib



from typing import Any, Callable, Mapping, Text

import rlax
from rlax._src.value_learning import _quantile_regression_loss as rlax_quantile_regression_loss

import chex
Array = chex.Array
Numeric = chex.Numeric

#from dqn_zoo.cnc_cr_dqn.cnc_cr_dqn_parts import qr_atari_network, nc_qr_atari_network, symm_qr_atari_network, _batch_quantile_q_learning
from cnc_cr_dqn_parts import qr_atari_network, nc_qr_atari_network, symm_qr_atari_network, _batch_quantile_q_learning


class CrDqn:
  """Quantile Regression DQN agent, using Cramér loss"""

  def __init__(
      self,
      preprocessor: processors.Processor,
      sample_network_input: jnp.ndarray,
      network: parts.Network,
      #quantiles: jnp.ndarray,
      optimizer: optax.GradientTransformation,
      transition_accumulator: Any,
      replay: replay_lib.TransitionReplay,
      batch_size: int,
      exploration_epsilon: Callable[[int], float],
      min_replay_capacity_fraction: float,
      learn_period: int,
      target_network_update_period: int,
      rng_key: parts.PRNGKey,
  ):
    self._preprocessor = preprocessor
    self._replay = replay
    self._transition_accumulator = transition_accumulator
    self._batch_size = batch_size
    self._exploration_epsilon = exploration_epsilon
    self._min_replay_capacity = min_replay_capacity_fraction * replay.capacity
    self._learn_period = learn_period
    self._target_network_update_period = target_network_update_period

    # Initialize network parameters and optimizer.
    self._rng_key, network_rng_key = jax.random.split(rng_key)
    self._online_params = network.init(network_rng_key,
                                       sample_network_input[None, ...])
    self._target_params = self._online_params
    self._opt_state = optimizer.init(self._online_params)

    # Other agent state: last action, frame count, etc.
    self._action = None
    self._frame_t = -1  # Current frame index.

    # Define jitted loss, update, and policy functions here instead of as
    # class methods, to emphasize that these are meant to be pure functions
    # and should not access the agent object's state via `self`.

    def loss_fn(online_params, target_params, transitions, rng_key):
      """Calculates loss given network parameters and transitions."""
      # Compute Q value distributions.
      _, online_key, target_key = jax.random.split(rng_key, 3)
      dist_q_tm1 = network.apply(online_params, online_key,
                                 transitions.s_tm1).q_dist
      
      q_target_t = network.apply(target_params, target_key,
                                      transitions.s_t)
      
      dist_q_target_t  = q_target_t.q_dist
      
      #this could be used instead of recomputing the mean
      q_values_target_t  = q_target_t.q_values


      losses = _batch_quantile_q_learning(
          dist_q_tm1,
          transitions.a_tm1,
          transitions.r_t,
          transitions.discount_t,
          dist_q_target_t,  # No double Q-learning here.
          dist_q_target_t,
          q_values_target_t,
      )
      assert losses.shape == (self._batch_size,)
      loss = jnp.mean(losses)
      return loss

    def update(rng_key, opt_state, online_params, target_params, transitions):
      """Computes learning update from batch of replay transitions."""
      rng_key, update_key = jax.random.split(rng_key)

      d_loss_d_params = jax.grad(loss_fn)(online_params, target_params,
                                          transitions, update_key)
      updates, new_opt_state = optimizer.update(d_loss_d_params, opt_state)
      new_online_params = optax.apply_updates(online_params, updates)
      return rng_key, new_opt_state, new_online_params


    self._update = jax.jit(update)

    def select_action(rng_key, network_params, s_t, exploration_epsilon):
      """Samples action from eps-greedy policy wrt Q-values at given state."""
      rng_key, apply_key, policy_key = jax.random.split(rng_key, 3)
      q_t = network.apply(network_params, apply_key, s_t[None, ...]).q_values[0]
      a_t = rlax.epsilon_greedy().sample(policy_key, q_t, exploration_epsilon)
      return rng_key, a_t

    self._select_action = jax.jit(select_action)
    #self._select_action = select_action

  def step(self, timestep: dm_env.TimeStep) -> parts.Action:
    """Selects action given timestep and potentially learns."""
    self._frame_t += 1

    timestep = self._preprocessor(timestep)

    if timestep is None:  # Repeat action.
      action = self._action
    else:
      action = self._action = self._act(timestep)

      for transition in self._transition_accumulator.step(timestep, action):
        self._replay.add(transition)

    if self._replay.size < self._min_replay_capacity:
      return action

    if self._frame_t % self._learn_period == 0:
      self._learn()

    if self._frame_t % self._target_network_update_period == 0:
      self._target_params = self._online_params

    return action

  def reset(self) -> None:
    """Resets the agent's episodic state such as frame stack and action repeat.

    This method should be called at the beginning of every episode.
    """
    self._transition_accumulator.reset()
    processors.reset(self._preprocessor)
    self._action = None

  def _act(self, timestep) -> parts.Action:
    """Selects action given timestep, according to epsilon-greedy policy."""
    s_t = timestep.observation
    self._rng_key, a_t = self._select_action(self._rng_key, self._online_params,
                                             s_t, self.exploration_epsilon)
    return parts.Action(jax.device_get(a_t))

  def _learn(self) -> None:
    """Samples a batch of transitions from replay and learns from it."""
    logging.log_first_n(logging.INFO, 'Begin learning', 1)
    transitions = self._replay.sample(self._batch_size)
    
    #print(frame_t)
    self._rng_key, self._opt_state, self._online_params = self._update(
        self._rng_key,
        self._opt_state,
        self._online_params,
        self._target_params,
        transitions
    )

  @property
  def online_params(self) -> parts.NetworkParams:
    """Returns current parameters of Q-network."""
    return self._online_params

  @property
  def exploration_epsilon(self) -> float:
    """Returns epsilon value currently used by (eps-greedy) behavior policy."""
    return self._exploration_epsilon(self._frame_t)

  def get_state(self) -> Mapping[Text, Any]:
    """Retrieves agent state as a dictionary (e.g. for serialization)."""
    state = {
        'rng_key': self._rng_key,
        'frame_t': self._frame_t,
        'opt_state': self._opt_state,
        'online_params': self._online_params,
        'target_params': self._target_params,
        'replay': self._replay.get_state(),
    }
    return state

  def set_state(self, state: Mapping[Text, Any]) -> None:
    """Sets agent state from a (potentially de-serialized) dictionary."""
    self._rng_key = state['rng_key']
    self._frame_t = state['frame_t']
    self._opt_state = jax.device_put(state['opt_state'])
    self._online_params = jax.device_put(state['online_params'])
    self._target_params = jax.device_put(state['target_params'])
    self._replay.set_state(state['replay'])
 
  @property
  def statistics(self) -> Mapping[Text, float]:
    """Returns current agent statistics as a dictionary."""
    return {}
########################################################################

# Relevant flag values are expressed in terms of environment frames.
FLAGS = flags.FLAGS
flags.DEFINE_string('environment_name', 'pong', '')
flags.DEFINE_integer('environment_height', 84, '')
flags.DEFINE_integer('environment_width', 84, '')
flags.DEFINE_bool('use_gym', False, '')
flags.DEFINE_integer('replay_capacity', int(1e6), '')
flags.DEFINE_bool('compress_state', True, '')
flags.DEFINE_float('min_replay_capacity_fraction', 0.05, '')
flags.DEFINE_integer('batch_size', 32, '')
flags.DEFINE_integer('max_frames_per_episode', 108000, '')  # 30 mins.
flags.DEFINE_integer('num_action_repeats', 4, '')
flags.DEFINE_integer('num_stacked_frames', 4, '')
flags.DEFINE_float('exploration_epsilon_begin_value', 1., '')
flags.DEFINE_float('exploration_epsilon_end_value', 0.01, '')
flags.DEFINE_float('exploration_epsilon_decay_frame_fraction', 0.02, '')
flags.DEFINE_float('eval_exploration_epsilon', 0.001, '')
flags.DEFINE_integer('target_network_update_period', int(4e4), '')
flags.DEFINE_float('learning_rate', 0.00005, '')
flags.DEFINE_float('optimizer_epsilon', 0.01 / 32, '')
flags.DEFINE_float('additional_discount', 0.99, '')
flags.DEFINE_float('max_abs_reward', 1., '')
flags.DEFINE_float('max_global_grad_norm', 10., '')
flags.DEFINE_integer('seed', 1, '')  # GPU may introduce nondeterminism.
flags.DEFINE_integer('num_iterations', 200, '')
flags.DEFINE_integer('num_train_frames', int(1e6), '')  # Per iteration.
flags.DEFINE_integer('num_eval_frames', int(5e5), '')  # Per iteration.
flags.DEFINE_integer('learn_period', 16, '')
flags.DEFINE_string('results_csv_path', '/tmp/results.csv', '')

flags.DEFINE_integer('num_quantiles', 201, '')
flags.DEFINE_integer('n_nodes', 512, '')
flags.DEFINE_integer('n_layers', 1, '')

flags.DEFINE_bool('nc', False, '')
flags.DEFINE_bool('symm', False, '')


def main(argv):
  """Trains CR-DQN agent on Atari."""
  logging.info(FLAGS.flags_into_string())

  del argv
  logging.info('CR-DQN with Cramer on Atari on %s.',
               jax.lib.xla_bridge.get_backend().platform)
  random_state = np.random.RandomState(FLAGS.seed)
  rng_key = jax.random.PRNGKey(
      random_state.randint(-sys.maxsize - 1, sys.maxsize + 1))

  if FLAGS.results_csv_path:
    writer = parts.CsvWriter(FLAGS.results_csv_path)
  else:
    writer = parts.NullWriter()

  def environment_builder():
    """Creates Atari environment."""
    env = gym_atari.GymAtari(
        FLAGS.environment_name, seed=random_state.randint(1, 2**32))
    return gym_atari.RandomNoopsEnvironmentWrapper(
        env,
        min_noop_steps=1,
        max_noop_steps=30,
        seed=random_state.randint(1, 2**32),
    )

  env = environment_builder()

  logging.info('Environment: %s', FLAGS.environment_name)
  logging.info('Action spec: %s', env.action_spec())
  logging.info('Observation spec: %s', env.observation_spec())
  num_actions = env.action_spec().num_values
  
  #if FLAGS.nc and FLAGS.num_quantiles%2 != 0: #must be even
  #    print("num_quantiles must be even for nc")
  #    exit(1)
  #num_quantiles = FLAGS.num_quantiles
  #quantiles = (jnp.arange(0, num_quantiles) + 0.5) / float(num_quantiles)

  if FLAGS.symm:
      logging.info('Symm network')
      network_fn = symm_qr_atari_network(num_actions, FLAGS.num_quantiles,
                                       FLAGS.n_layers, FLAGS.n_nodes)
  elif FLAGS.nc:
      logging.info('NC network')
      network_fn = nc_qr_atari_network(num_actions, FLAGS.num_quantiles,
                                       FLAGS.n_layers, FLAGS.n_nodes)
  else:
      logging.info('Standard QR network')
      network_fn = qr_atari_network(num_actions, FLAGS.num_quantiles)

  network = hk.transform(network_fn)


  def preprocessor_builder():
    return processors.atari(
        additional_discount=FLAGS.additional_discount,
        max_abs_reward=FLAGS.max_abs_reward,
        resize_shape=(FLAGS.environment_height, FLAGS.environment_width),
        num_action_repeats=FLAGS.num_action_repeats,
        num_pooled_frames=2,
        zero_discount_on_life_loss=True,
        num_stacked_frames=FLAGS.num_stacked_frames,
        grayscaling=True,
    )

  # Create sample network input from sample preprocessor output.
  sample_processed_timestep = preprocessor_builder()(env.reset())
  sample_processed_timestep = typing.cast(dm_env.TimeStep,
                                          sample_processed_timestep)
  sample_network_input = sample_processed_timestep.observation
  assert sample_network_input.shape == (FLAGS.environment_height,
                                        FLAGS.environment_width,
                                        FLAGS.num_stacked_frames)

  exploration_epsilon_schedule = parts.LinearSchedule(
      begin_t=int(FLAGS.min_replay_capacity_fraction * FLAGS.replay_capacity *
                  FLAGS.num_action_repeats),
      decay_steps=int(FLAGS.exploration_epsilon_decay_frame_fraction *
                      FLAGS.num_iterations * FLAGS.num_train_frames),
      begin_value=FLAGS.exploration_epsilon_begin_value,
      end_value=FLAGS.exploration_epsilon_end_value)

  if FLAGS.compress_state:

    def encoder(transition):
      return transition._replace(
          s_tm1=replay_lib.compress_array(transition.s_tm1),
          s_t=replay_lib.compress_array(transition.s_t))

    def decoder(transition):
      return transition._replace(
          s_tm1=replay_lib.uncompress_array(transition.s_tm1),
          s_t=replay_lib.uncompress_array(transition.s_t))
  else:
    encoder = None
    decoder = None

  replay_structure = replay_lib.Transition(
      s_tm1=None,
      a_tm1=None,
      r_t=None,
      discount_t=None,
      s_t=None,
  )

  replay = replay_lib.TransitionReplay(FLAGS.replay_capacity, replay_structure,
                                       random_state, encoder, decoder)

  optimizer = optax.adam(
      learning_rate=FLAGS.learning_rate, eps=FLAGS.optimizer_epsilon)
  if FLAGS.max_global_grad_norm > 0:
    optimizer = optax.chain(
        optax.clip_by_global_norm(FLAGS.max_global_grad_norm), optimizer)

  train_rng_key, eval_rng_key = jax.random.split(rng_key)

  train_agent = CrDqn(
      preprocessor=preprocessor_builder(),
      sample_network_input=sample_network_input,
      network=network,
      #quantiles=quantiles,
      optimizer=optimizer,
      transition_accumulator=replay_lib.TransitionAccumulator(),
      replay=replay,
      batch_size=FLAGS.batch_size,
      exploration_epsilon=exploration_epsilon_schedule,
      min_replay_capacity_fraction=FLAGS.min_replay_capacity_fraction,
      learn_period=FLAGS.learn_period,
      target_network_update_period=FLAGS.target_network_update_period,
      rng_key=train_rng_key,
  )
  eval_agent = parts.EpsilonGreedyActor(
      preprocessor=preprocessor_builder(),
      network=network,
      exploration_epsilon=FLAGS.eval_exploration_epsilon,
      rng_key=eval_rng_key,
  )

  # Set up checkpointing.
  checkpoint = parts.NullCheckpoint()

  state = checkpoint.state
  state.iteration = 0
  state.train_agent = train_agent
  state.eval_agent = eval_agent
  state.random_state = random_state
  state.writer = writer
  if checkpoint.can_be_restored():
    checkpoint.restore()

  while state.iteration <= FLAGS.num_iterations:
    # New environment for each iteration to allow for determinism if preempted.
    env = environment_builder()

    logging.info('Training iteration %d.', state.iteration)
    train_seq = parts.run_loop(train_agent, env, FLAGS.max_frames_per_episode)
    num_train_frames = 0 if state.iteration == 0 else FLAGS.num_train_frames
    train_seq_truncated = itertools.islice(train_seq, num_train_frames)
    
    train_trackers = parts.make_default_trackers(train_agent)
    train_stats = parts.generate_statistics(train_trackers,train_seq_truncated)

    logging.info('Evaluation iteration %d.', state.iteration)
    eval_agent.network_params = train_agent.online_params
    eval_seq = parts.run_loop(eval_agent, env, FLAGS.max_frames_per_episode)
    eval_seq_truncated = itertools.islice(eval_seq, FLAGS.num_eval_frames)
    eval_trackers = parts.make_default_trackers(eval_agent)
    eval_stats = parts.generate_statistics(eval_trackers,eval_seq_truncated)

    # Logging and checkpointing.
    human_normalized_score = atari_data.get_human_normalized_score(
        FLAGS.environment_name, eval_stats['episode_return'])
    capped_human_normalized_score = np.amin([1., human_normalized_score])
    log_output = [
        ('iteration', state.iteration, '%3d'),
        ('frame', state.iteration * FLAGS.num_train_frames, '%5d'),
        ('eval_episode_return', eval_stats['episode_return'], '% 2.2f'),
        ('train_episode_return', train_stats['episode_return'], '% 2.2f'),
        ('eval_num_episodes', eval_stats['num_episodes'], '%3d'),
        ('train_num_episodes', train_stats['num_episodes'], '%3d'),
        ('eval_frame_rate', eval_stats['step_rate'], '%4.0f'),
        ('train_frame_rate', train_stats['step_rate'], '%4.0f'),
        ('train_exploration_epsilon', train_agent.exploration_epsilon, '%.3f'),
        ('normalized_return', human_normalized_score, '%.3f'),
        ('capped_normalized_return', capped_human_normalized_score, '%.3f'),
        ('human_gap', 1. - capped_human_normalized_score, '%.3f'),
    ]
    log_output_str = ', '.join(('%s: ' + f) % (n, v) for n, v, f in log_output)
    logging.info(log_output_str)
    writer.write(collections.OrderedDict((n, v) for n, v, _ in log_output))
    state.iteration += 1
    checkpoint.save()

  writer.close()


if __name__ == '__main__':
  config.update('jax_platform_name', 'gpu')  # Default to GPU.
  config.update('jax_numpy_rank_promotion', 'raise')
  config.config_with_absl()
  app.run(main)
