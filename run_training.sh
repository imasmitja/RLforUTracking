#!/bin/bash

# run training with rendering enabled
# also performs cleanup
./clean.sh
xvfb-run -s "-screen 0 600x400x24" python main.py
echo "also gif files are saved in model_dir/*.gif"
echo "execute ./run_tensorboard.sh to view results"
