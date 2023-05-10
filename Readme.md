# Multi Agent Reinforcement Learning Simulation
main.py is the main file to run the simulation. 
Environment is created in the world directory.

Agent is present in the agent directory.
Helper functions like optimizers are present in the helper directory.

any agent should be represented as a class and should have the following functions:
1. __init__(self, env, agent_id, **kwargs)
2. forward(self, state, **kwargs)
3. save(self, path, **kwargs)
4. load(self, path, **kwargs)
5. select_action(self, output, **kwargs)

any environment should be represented as a class (inherited from gymnasium) and should have the following functions:
1. __init__(self, **kwargs)
2. reset(self, **kwargs)
3. step(self, action, **kwargs)
4. render(self, **kwargs), also it should have the option to choose from different render modes
5. close(self, **kwargs)


# State

[
    {
    "predator": [
        (3, VISION, VISION)nd array,
        ..... number of predators
        ],
    "prey": [
        [3, VISION, VISION],
        ..... number of preys
        ]
    }
......history times
]


# Reward
{
    "predator": [
            reward,
        ..... number of predators
    ],
    "prey": [
            reward,
        ..... number of preys
    ]
}
