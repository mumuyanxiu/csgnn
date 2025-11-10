def get_network(network_name):
    network_name = network_name.lower()
    if network_name == 'hybrid' or network_name == 'serial':
        from .serial_model import HybridGraspNet
        return HybridGraspNet
    elif network_name == 'parallel':
        from .parallel_model import ParallelHybridGraspNet
        return ParallelHybridGraspNet
    else:
        raise NotImplementedError('Network {} is not implemented'.format(network_name))
