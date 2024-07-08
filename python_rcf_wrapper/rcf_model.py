# Java imports
from typing import List, Optional, Tuple, Any

import numpy as np
import logging
from com.amazon.randomcutforest import RandomCutForest
import jpype

class RandomCutForestModel:
    """
    Random Cut Forest Python Binding around the AWS Random Cut Forest Official Java version:
    https://github.com/aws/random-cut-forest-by-aws
    """

    def __init__(self, forest: RandomCutForest = None, shingle_size: int = 8,
                 num_trees: int = 100, random_seed: int = None,
                 sample_size: int = 256, parallel_execution_enabled: bool = True,
                 thread_pool_size: Optional[int] = None, lam: float=0.0001,
                 output_after: int=256):
        if forest is not None:
            self.forest = forest
        else:
            builder = RandomCutForest.builder().numberOfTrees(num_trees). \
                sampleSize(sample_size). \
                dimensions(shingle_size). \
                storeSequenceIndexesEnabled(True). \
                centerOfMassEnabled(True). \
                parallelExecutionEnabled(parallel_execution_enabled). \
                timeDecay(lam). \
                outputAfter(output_after)
            if thread_pool_size is not None:
                builder.threadPoolSize(thread_pool_size)

            if random_seed is not None:
                builder = builder.randomSeed(random_seed)

            self.forest = builder.build()

    def score(self, point: List[float]) -> float:
        """
        Compute an anomaly score for the given point.

        Parameters
        ----------
        point: List[float]
            A data point with shingle size

        Returns
        -------
        float
            The anomaly score for the given point

        """
        return self.forest.getAnomalyScore(point)

    def update(self, point: List[float]):
        """
        Update the model with the data point.

        Parameters
        ----------
        point: List[float]
            Point with shingle size
        """
        self.forest.update(point)


    def impute(self, point: List[float]) -> List[float]:
        """
        Given a point with missing values, return a new point with the missing values imputed. Each tree in the forest
        individual produces an imputed value. For 1-dimensional points, the median imputed value is returned. For
        points with more than 1 dimension, the imputed point with the 25th percentile anomaly score is returned.

        Parameters
        ----------
        point: List[float]
            The point with shingle size

        Returns
        -------
        List[float]
            The imputed point.
        """

        num_missing = np.isnan(point).sum()
        if num_missing == 0:
            return point
        missing_index = np.argwhere(np.isnan(point)).flatten()
        imputed_shingle = list(self.forest.imputeMissingValues(point, num_missing, missing_index))
        return imputed_shingle

    def forecast(self, point: List[float]) -> float:
        """
        Given one shingled data point, return one step forecast containing the next value.

        Parameters
        ----------
        point: List[float]
            The point with shingle size

        Returns
        -------
        float
            Forecast value of next timestamp.

        """
        val = list(self.forest.extrapolateBasic(point, 1, 1, False, 0))[0]
        return val

    @property
    def shingle_size(self) -> int:
        """
        Returns
        -------
        int
            Shingle size of random cut trees.
        """
        return self.forest.getDimensions()

    def get_attribution(self, point: List[float]) -> Tuple[List[float], List[float]]:
        try:
            attribution_di_vec: Any = self.forest.getAnomalyAttribution(point)
            low: List[float] = list(attribution_di_vec.low)
            high: List[float] = list(attribution_di_vec.high)
            return low, high
        except jpype.JException as exception:
            logging.info("Error when loading the model: %s", exception.message())
            logging.info("Stack track: %s", exception.stacktrace())
            # Throw it back
            raise exception

