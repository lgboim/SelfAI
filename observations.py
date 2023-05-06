"""
Observations utility for SelfAI project.
This script contains a function to create an observation from a given string.
"""

def create_observation(observation_object):
    if not isinstance(observation_object, str):
        raise TypeError("Observation object must be a string.")
    return observation_object
