"""Print the steps of a run's complete HF checkpoints (for shell loops in the SLURM scripts).

    python -m rrh.opd.list_checkpoints results/rp_opd/<run>   ->  "0 20 50 100 ..."
"""

import sys

from rrh.opd.checkpoint import list_hf_checkpoints
from rrh.results.layout import RunDir

if __name__ == "__main__":
    run = RunDir(sys.argv[1])
    print(" ".join(str(s) for s, _ in list_hf_checkpoints(run.checkpoint_dir())))
