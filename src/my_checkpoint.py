from mindspore.train.callback import ModelCheckpoint
import time

import src.config

class MyCheckpoint(ModelCheckpoint):
    def __init__(self, prefix='ms8p', directory=None, config=None):
        super(MyCheckpoint, self).__init__(prefix=prefix, directory=directory, config=config)
        self.last_step_time = time.time()
    def step_end(self, run_context):
        toc = time.time()
        step_cost = toc - self.last_step_time
        self.last_step_time = time.time()
        super(MyCheckpoint, self).step_end(run_context)
        cur_step_num = run_context.original_args()['cur_step_num']
        cur_epoch_num = run_context.original_args()['cur_epoch_num']
        net_outputs = run_context.original_args()['net_outputs']
        if cur_step_num//2 == 0:
            print(f'epoch: {cur_epoch_num}  step: {cur_step_num}  loss: {net_outputs.asnumpy():.5f}  step cost: {step_cost:.3f} ms')
    def epoch_end(self, run_context):
        super(MyCheckpoint, self).epoch_end(run_context)
        src.config.CUR_EPOCH = run_context.original_args()['cur_epoch_num']
