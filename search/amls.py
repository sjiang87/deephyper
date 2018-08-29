import json
import logging
import os
import pickle
import signal
import sys
import numpy as np
import time
from math import log
from numpy import integer, floating, ndarray

HERE = os.path.dirname(os.path.abspath(__file__)) # search dir
top  = os.path.dirname(os.path.dirname(HERE)) # directory containing deephyper
sys.path.append(top)

from deephyper.evaluators import evaluate
from deephyper.search import util

logger = util.conf_logger('deephyper.search.amls')

SERVICE_PERIOD = 2          # Delay (seconds) between main loop iterations
CHECKPOINT_INTERVAL = 1    # How many jobs to complete between optimizer checkpoints
SEED = 21278

profile_timer = util.Timer()

def save_checkpoint(opt_config, optimizer, evaluator):
    if evaluator.counter == 0: return
    data = {}
    data['opt_config'] = opt_config
    data['optimizer'] = optimizer
    data['evaluator'] = evaluator

    if evaluator.evals:
        best = min(evaluator.evals.items(), key=lambda x: x[1])
        data['best'] = best
        logger.info(f'best point: {best}')
    
    fname = f'{opt_config.benchmark}.pkl'
    with open(fname, 'wb') as fp:
        pickle.dump(data, fp)

    evaluator.dump_evals()
    logger.info(f"Checkpointed run in {os.path.abspath(fname)}")

def load_checkpoint(chk_path):
    chk_path = os.path.abspath(os.path.expanduser(chk_path))
    assert os.path.exists(chk_path), "No such checkpoint file"
    with open(chk_path, 'rb') as fp: data = pickle.load(fp)
    
    cfg, opt, evaluator = data['opt_config'], data['optimizer'], data['evaluator']

    cfg.num_workers = args.num_workers
    logger.info(f"Resuming from checkpoint in {chk_path}")
    logger.info(f"On eval {evaluator.counter}")
    return cfg, opt, evaluator

class Optimizer:
    class Encoder(json.JSONEncoder):
        '''JSON dump of numpy data'''
        def default(self, obj):
            if isinstance(obj, integer): return int(obj)
            elif isinstance(obj, floating): return float(obj)
            elif isinstance(obj, ndarray): return obj.tolist()
            else: return super(Encoder, self).default(obj)
    
    def _encode(self, x):
        return json.dumps(x, cls=self.Encoder)

    def _decode(self, key):
        return json.loads(key)

    def __init__(self, cfg):
        self._optimizer = util.sk_optimizer_from_config(cfg, SEED)
        assert cfg.amls_lie_strategy in "cl_min cl_mean cl_max".split()
        self.strategy = cfg.amls_lie_strategy
        self.evals = {}
        self.counter = 0

    def _get_lie(self):
        """
        if self.strategy == "cl_min":
            return min(self._optimizer.yi) if self._optimizer.yi else 0.0
        elif self.strategy == "cl_mean":
            return self._optimizer.yi.mean() if self._optimizer.yi else 0.0
        else:
            return  max(self._optimizer.yi) if self._optimizer.yi else 0.0
        """
        ti_available = "ps" in self._optimizer.acq_func and len(self._optimizer.yi) > 0
        ti = [t for (_, t) in self._optimizer.yi] if ti_available else None
        if self.strategy == "cl_min":
            y_lie = np.min(self._optimizer.yi) if self._optimizer.yi else 0.0  # CL-min lie
            t_lie = np.min(ti) if ti is not None else log(sys.float_info.max)
        elif self.strategy == "cl_mean":
            y_lie = np.mean(self._optimizer.yi) if self._optimizer.yi else 0.0  # CL-mean lie
            t_lie = np.mean(ti) if ti is not None else log(sys.float_info.max)
        else:
            y_lie = np.max(self._optimizer.yi) if self._optimizer.yi else 0.0  # CL-max lie
            t_lie = np.max(ti) if ti is not None else log(sys.float_info.max)

        # Lie to the optimizer.
        if "ps" in self._optimizer.acq_func:
            return    (y_lie, t_lie)
        else:
            return    y_lie
       
    def _xy_from_dict(self):
        keys = list(self.evals.keys())
        XX = [self._decode(x) for x in keys]
        YY = [self.evals[x] for x in keys]
        return XX, YY

    def _ask(self):
        x = self._optimizer.ask()
        y = self._get_lie()
        self._optimizer.tell(x,y)
        self.evals[self._encode(x)] = y
        return x

    def ask(self, n_points=None, batch_size=20):
        if n_points is None:
            self.counter += 1
            return self._ask()
        else:
            self.counter += n_points
            batch = []
            for i in range(n_points):
                batch.append(self._ask())
                if len(batch) == batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch

    def ask_initial(self, n_points):
        XX = self._optimizer.ask(n_points=n_points)
        for x in XX:
            key = self._encode(x)
            if "ps" in self._optimizer.acq_func:
              self.evals[key] = (0.0,800)
            else:
              self.evals[key] = 0.0  
        self.counter += n_points
        return XX
        
    def tell(self, xy_data):
        assert isinstance(xy_data, list)
        maxval = max(self._optimizer.yi) if self._optimizer.yi else 0.0
        for x,y in xy_data:
            key = self._encode(x)
            assert key in self.evals
            if "ps" in self._optimizer.acq_func:
              if isinstance(y, tuple):
                self.evals[key] = y
              else:
                self.evals[key] = (y, 36000)  
            else:   
               self.evals[key] = (y if y < sys.float_info.max else maxval)

        self._optimizer.Xi = []
        self._optimizer.yi = []
        XX, YY = self._xy_from_dict()
        assert len(XX) == len(YY) == self.counter
        #asen_code
        logger.info(f"XX:{XX}, YY:{YY}")
        #asen_code
        self._optimizer.tell(XX, YY)
        assert len(self._optimizer.Xi) == len(self._optimizer.yi) == self.counter

def main(args):
    '''Service loop: add jobs; read results; drive optimizer'''

    # Initialize optimizer
    if args.from_checkpoint:
        chk_path = args.from_checkpoint
        cfg, optimizer, evaluator = load_checkpoint(chk_path)
    else:
        cfg = util.OptConfig(args)
        print(cfg)
        optimizer = Optimizer(cfg)
        evaluator = evaluate.create_evaluator(cfg)
        logger.info(f"Starting new run with {cfg.benchmark_module_name}")

    timer = util.DelayTimer(max_minutes=None, period=SERVICE_PERIOD)
    chkpoint_counter = 0

    # Gracefully handle shutdown
    def handler(signum, stack):
        evaluator.stop()
        logger.info('Received SIGINT/SIGTERM')
        save_checkpoint(cfg, optimizer, evaluator)
        sys.exit(0)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    # INITIAL POINTS
    logger.info("AMLS-single server driver starting")
    logger.info(f"Generating {cfg.num_workers} initial points...")
    XX = optimizer.ask_initial(n_points=cfg.num_workers)
    for x in XX: evaluator.add_eval(x, re_evaluate=cfg.repeat_evals)
    #asen_code
    start_time = time.time()
    time_stamp = []
    maximum_seen = []
    #asen_code 
    
    # MAIN LOOP
    for elapsed_str in timer:
        logger.info(f"Elapsed time: {elapsed_str}")
        if len(evaluator.evals) == cfg.max_evals: break

        results = list(evaluator.get_finished_evals())
        
        if results:
            logger.info(f"Refitting model with batch of {len(results)} evals")
            #profile_timer.start('tell')
            optimizer.tell(results)
            #profile_timer.end('tell')
            #asen_code
            logger.info(f"Telling Finished")
            val = optimizer._optimizer.yi[0]
            logger.info(f"Val: {val}")
            if isinstance(val, list):
                current_max = -(min(optimizer._optimizer.yi, key = lambda x: x[0]))[0]
            else:
                current_max = -min(optimizer._optimizer.yi)                   
            time_stamp.append(time.time()-start_time)
            maximum_seen.append(current_max)
            logger.info(f"current_max: {current_max}")
            #asen_code
            logger.info(f"Drawing {len(results)} points with strategy {optimizer.strategy}")
            if len(results)== 1:
               optimizer.counter += 1
               x = optimizer._optimizer.ask()
               y = optimizer._get_lie()
               optimizer.evals[optimizer._encode(x)] = y
               evaluator.add_eval_batch([x], re_evaluate=cfg.repeat_evals)
            else:
               for batch in optimizer.ask(n_points=len(results)):
                  evaluator.add_eval_batch(batch, re_evaluate=cfg.repeat_evals)
            chkpoint_counter += len(results)
            

        if chkpoint_counter >= CHECKPOINT_INTERVAL:
            save_checkpoint(cfg, optimizer, evaluator)
            chkpoint_counter = 0
        sys.stdout.flush()
    #asen_code
    with open("current_max.txt", "wb") as fp:   
        pickle.dump(maximum_seen, fp)
    with open("time_stamp.txt","wb") as fp:
        pickle.dump(time_stamp,fp)
    #asen_code        
    # EXIT
    logger.info('Hyperopt driver finishing')
    save_checkpoint(cfg, optimizer, evaluator)

if __name__ == "__main__":
    parser = util.create_parser()
    args = parser.parse_args()
    print(args)
    main(args)
