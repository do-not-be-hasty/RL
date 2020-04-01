"""Environment steppers."""

from alpacka import data
import numpy as np


class BatchStepper:
    """Base class for running a batch of steppers.

    Abstracts out local/remote prediction using a Network.
    """

    def __init__(
        self, env_class, agent_class, network_fn, n_envs, output_dir
    ):
        """No-op constructor just for documentation purposes.

        Args:
            env_class (type): Environment class.
            agent_class (type): Agent class.
            network_fn (callable): Function () -> Network. Note: we take this
                instead of an already-initialized Network, because some
                BatchSteppers will send it to remote workers and it makes no
                sense to force Networks to be picklable just for this purpose.
            n_envs (int): Number of parallel environments to run.
            output_dir (str or None): Experiment output dir if the BatchStepper
                is initialized from Runner, None otherwise.
        """
        del env_class
        del agent_class
        del network_fn
        del n_envs
        del output_dir

    def run_episode_batch(self, params, **solve_kwargs):  # pylint: disable=missing-param-doc
        """Runs a batch of episodes using the given network parameters.

        Args:
            params (Network-dependent): Network parameters.
            **solve_kwargs (dict): Keyword arguments passed to Agent.solve().

        Returns:
            List of completed episodes (Agent/Trainer-dependent).
        """
        raise NotImplementedError


class RequestHandler:
    """Handles requests from the agent coroutine to the network."""

    def __init__(self, network_fn):
        """Initializes RequestHandler.

        Args:
            network_fn (callable): Function () -> Network.
        """
        self.network_fn = network_fn

        self._network = None  # Lazy initialize if needed
        self._should_update_params = None

    def run_coroutine(self, episode_cor, params):  # pylint: disable=missing-param-doc
        """Runs an episode coroutine using the given network parameters.

        Args:
            episode_cor (coroutine): Agent.solve coroutine.
            params (Network-dependent): Network parameters.

        Returns:
            List of completed episodes (Agent/Trainer-dependent).
        """
        self._should_update_params = True

        try:
            request = next(episode_cor)
            # TODO hacky, input transformation should be handled in sender
            #print("request\n\n\n\n", episode_cor, request[0]['observation'].shape)
            #request = np.array([np.concatenate([request[i]['observation'], request[i]['desired_goal']], axis=-1) for i in range(len(request))])
            #print(request.shape)
            
            while True:
                if isinstance(request, data.NetworkRequest):
                    request_handler = self._handle_network_request
                else:
                    request_handler = self._handle_prediction_request

                response = request_handler(request, params)
                request = episode_cor.send(response)
        except StopIteration as e:
            return e.value  # episodes

    def _handle_network_request(self, request, params):
        del request
        return self.network_fn, params

    def _handle_prediction_request(self, request, params):
        return self.get_network(params).predict(request)

    def get_network(self, params=None):
        if self._network is None:
            self._network = self.network_fn()
        if params is not None and self._should_update_params:
            self.network.params = params
            self._should_update_params = False
        return self._network
    network = property(get_network)
