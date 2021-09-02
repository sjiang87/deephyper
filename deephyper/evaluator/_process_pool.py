import logging
import asyncio

from deephyper.evaluator.evaluate import Evaluator

from concurrent.futures import ProcessPoolExecutor

logger = logging.getLogger(__name__)


class ProcessPoolEvaluator(Evaluator):

    def __init__(self, run_function, num_workers: int=1, callbacks=None):
        super().__init__(run_function, num_workers, callbacks)
        self.sem = asyncio.Semaphore(num_workers)
        logger.info(
            f"ProcessPool Evaluator will execute {self.run_function.__name__}() from module {self.run_function.__module__}"
        )

    async def execute(self, job):

        async with self.sem:
    
            executor = ProcessPoolExecutor(max_workers=1)
            sol = await self.loop.run_in_executor(executor, job.run_function, job.config)
    
            job.result = sol
    

        return job