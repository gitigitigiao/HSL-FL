#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.7
import os
cmds = [
"python3.7 comm_main_fed_async_roulette.py --dataset cifar --iid  --epochs 200 --times 2 --updateStrategy adaptive --alpha 0.01 ",
"python3.7 comm_main_fed_async_roulette.py --dataset cifar --iid  --epochs 100 --times 3 --updateStrategy fixed --T 100 --beta 1.0 ",

]
for cmd in cmds:
    os.system(cmd)

