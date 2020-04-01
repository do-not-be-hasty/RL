"""Environment steppers."""

import typing

import gin
import numpy as np

from alpacka.batch_steppers import core

# WA for: https://github.com/ray-project/ray/issues/5250
# One of later packages (e.g. gym_sokoban.envs) imports numba internally.
# This WA ensures its done before Ray to prevent llvm assertion error.
# TODO(pj): Delete the WA with new Ray release that updates pyarrow.
import numba  # pylint: disable=wrong-import-order
import ray  # pylint: disable=wrong-import-order
del numba


class RayObject(typing.NamedTuple):
    """Keeps value and id of an object in the Ray Object Store."""
    id: typing.Any
    value: typing.Any

    @classmethod
    def from_value(cls, value, weakref=False):
        return cls(ray.put(value, weakref=weakref), value)


class RayBatchStepper(core.BatchStepper):
    """Batch stepper running remotely using Ray.

    Runs predictions and steps environments for all Agents separately in their
    own workers.

    It's highly recommended to pass params to run_episode_batch as a numpy array
    or a collection of numpy arrays. Then each worker can retrieve params with
    zero-copy operation on each node.
    """

    class Worker:
        """Ray actor used to step agent-environment-network in own process."""

        def __init__(self, env_class, agent_class, network_fn, config):
            # Limit number of threads used between independent tf.op-s to 1.
            import tensorflow as tf  # pylint: disable=import-outside-toplevel
            tf.config.threading.set_inter_op_parallelism_threads(1)
            tf.config.threading.set_intra_op_parallelism_threads(1)

            # TODO(pj): Test that skip_unknown is required!
            gin.parse_config(config, skip_unknown=True)

            self.env = env_class()
            self.agent = agent_class()
            self._request_handler = core.RequestHandler(network_fn)

        def run(self, params, solve_kwargs):
            """Runs the episode using the given network parameters."""
            episode_cor = self.agent.solve(self.env, **solve_kwargs)
            return self._request_handler.run_coroutine(episode_cor, params)

        @property
        def network(self):
            return self._request_handler.network

    def __init__(self, env_class, agent_class, network_fn, n_envs, output_dir):
        super().__init__(env_class, agent_class, network_fn, n_envs, output_dir)

        config = RayBatchStepper._get_config(env_class, agent_class, network_fn)
        ray_worker_cls = ray.remote(RayBatchStepper.Worker)

        if not ray.is_initialized():
            kwargs = {
                # Size of the Plasma object store, hardcoded to 1GB for now.
                # TODO(koz4k): Gin-configure if we ever need to change it.
                'object_store_memory': int(1e9),
            }
            ray.init(**kwargs)
        self.workers = [ray_worker_cls.remote(  # pylint: disable=no-member
            env_class, agent_class, network_fn, config) for _ in range(n_envs)]

        self._params = RayObject(None, None)
        self._solve_kwargs = RayObject(None, None)

    def run_episode_batch(self, params, **solve_kwargs):
        """Runs a batch of episodes using the given network parameters.

        Args:
            params (list of np.ndarray): List of network parameters as
                numpy ndarray-s.
            **solve_kwargs (dict): Keyword arguments passed to Agent.solve().

        Returns:
            List of completed episodes (Agent/Trainer-dependent).
        """
        # Optimization, don't send the same parameters again.
        if self._params.value is None or not all(
            [np.array_equal(p1, p2)
             for p1, p2 in zip(params, self._params.value)]
        ):
            self._params = RayObject.from_value(params)

        # TODO(pj): Don't send the same solve kwargs again. This is more
        #           problematic than with params, as values may have very
        #           different types e.g. basic data types or np.ndarray or ???.
        self._solve_kwargs = RayObject.from_value(solve_kwargs)

        episodes = ray.get([w.run.remote(self._params.id, self._solve_kwargs.id)
                            for w in self.workers])
        return episodes

    @staticmethod
    def _get_config(env_class, agent_class, network_fn):
        """Returns gin operative config for (at least) env, agent and network.

        It creates env, agent and network to initialize operative gin-config.
        It deletes them afterwords.
        """

        env_class()
        agent_class()
        network_fn()
        return gin.operative_config_str()
