import os
import sys

sys.path.insert(0, os.getcwd())
from partgen.partfield_runner import PartFieldRunner

partfield_runner = PartFieldRunner()

model_path = "./thirdparty/PartField/data/objaverse_samples/00200996b8f34f55a2dd2f44d316d107.glb"
ans = partfield_runner.run_partfield(model_path)

import ipdb; ipdb.set_trace()
