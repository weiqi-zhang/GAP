import config_diff
from cldm.hack import disable_verbosity, enable_sliced_attention


disable_verbosity()

if config_diff.save_memory:
    enable_sliced_attention()
