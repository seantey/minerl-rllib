from minerl_rllib.envs.wrappers.action_repeat_wrapper import MineRLActionRepeat
from minerl_rllib.envs.wrappers.action_wrapper import MineRLActionWrapper
from minerl_rllib.envs.wrappers.deterministic_wrapper import MineRLDeterministic
from minerl_rllib.envs.wrappers.discrete_action_wrapper import MineRLDiscreteActionWrapper
from minerl_rllib.envs.wrappers.gray_scale_wrapper import MineRLGrayScale
from minerl_rllib.envs.wrappers.observation_stack_wrapper import MineRLObservationStack
from minerl_rllib.envs.wrappers.observation_wrapper import MineRLObservationWrapper
from minerl_rllib.envs.wrappers.time_limit_wrapper import MineRLTimeLimitWrapper


def wrap(env, discrete=False, num_actions=32, data_dir=None, num_stack=1, action_repeat=1,
         gray_scale=False, seed=None):
    env = MineRLTimeLimitWrapper(env)
    env = MineRLObservationWrapper(env)
    env = MineRLActionWrapper(env)
    if discrete:
        env = MineRLDiscreteActionWrapper(env, num_actions, data_dir=data_dir)
    if gray_scale:
        env = MineRLGrayScale(env)
    if num_stack > 1:
        env = MineRLObservationStack(env, num_stack)
    if action_repeat > 1:
        env = MineRLActionRepeat(env, action_repeat)
    if seed is not None:
        env = MineRLDeterministic(env, seed)
    return env

def gym_wrap(env_class, env_kwargs, discrete=False, num_actions=32, data_dir=None, num_stack=1, action_repeat=1,
            gray_scale=False, seed=None):
    def wrap():
        env = env_class(**env_kwargs)
        env = MineRLTimeLimitWrapper(env)
        env = MineRLObservationWrapper(env)
        env = MineRLActionWrapper(env)
        if discrete:
            env = MineRLDiscreteActionWrapper(env, num_actions, data_dir=data_dir)
        if gray_scale:
            env = MineRLGrayScale(env)
        if num_stack > 1:
            env = MineRLObservationStack(env, num_stack)
        if action_repeat > 1:
            env = MineRLActionRepeat(env, action_repeat)
        if seed is not None:
            env = MineRLDeterministic(env, seed)
        return env
    return wrap
