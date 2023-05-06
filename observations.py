def create_observation(observation_object):
    if not isinstance(observation_object, str):
        raise TypeError("Observation object must be a string.")
    return observation_object
