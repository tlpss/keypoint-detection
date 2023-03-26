wishlist for the configs

- hierarchical configurations                                       -> just use python dataclasses with composition
- can validate input                                                -> use the pydantic dataclass instead
- all typed                                                         -> dataclasses
- minimal duplication to create CLI                                 -> like to avoid having to specify the names twice (and in a string format)
- can log configs to wandb                                          -> can convert config to a dict to log to wandb
- can use wandb sweeps with the config                              -> configurations can be updated through command line arugments to the python file
                                                                    or instantiated from a dict


- can have defaults, project defaults and then still override       -> defaults in dataclasses, can be overriden by config file defaults and those can
                                                                    be overruled by CLI



## Schemes

### plain python
### argparse CLI
### click CLI
### Hydra

#### init functions
#### Structured configs
#### Instantiation
####
example - https://github.com/ashleve/lightning-hydra-template


## Who does what?
