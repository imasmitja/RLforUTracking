#!/bin/bash
tensorboard --logdir=./log/ --host=0.0.0.0 --port=3000 &> /dev/null &
echo "Wait around 10 seconds and open this link at your browser (ignore other outputs):"
echo "https://$WORKSPACEID-3000.$WORKSPACEDOMAIN"
