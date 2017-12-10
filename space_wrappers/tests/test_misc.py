import pytest
from gym import spaces
from space_wrappers.misc import *
try:
    from unittest import mock
except ImportError:
    import mock


def test_repeat_action_wrapper():
    with mock.patch("gym.wrappers.SkipWrapper") as skip_wrapper:
        env = mock.Mock()
        rep = RepeatActionWrapper(env, 5)

        skip_wrapper.assert_called_once_with(5)
        skip_wrapper.return_value.assert_called_once_with(env)

        assert rep == skip_wrapper.return_value.return_value


def test_scalar_action_wrapper():
    env = mock.Mock()
    wrapped = ToScalarActionWrapper(env)
    assert wrapped._action(np.array([1])) == 1
    # TODO do we want an error message here.
    assert wrapped._action([1, 2, 3]) == [1, 2, 3]
    assert list(wrapped._action(np.array([1, 2, 3]))) == [1, 2, 3]


def test_stack_observation():
    env = mock.Mock()
    env.observation_space = spaces.Box(0.0, 1.0, shape=(2, 3))
    env.reset.return_value = np.zeros((2, 3))
    env.step = lambda x: (x, None, None, None)

    wrapped = StackObservationWrapper(env, 5, axis=-1)
    assert wrapped.observation_space == spaces.Box(0.0, 1.0, shape=(2, 3, 5))

    assert wrapped.reset() == pytest.approx(np.zeros((2, 3, 5)))
    o, _, _, _ = wrapped.step(np.ones((2, 3)) * 0.5)
    assert o == pytest.approx(np.concatenate((np.zeros((2, 3, 4)), np.ones((2, 3, 1)) * 0.5), axis=2))
