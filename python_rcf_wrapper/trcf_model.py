# Java imports
from typing import List, Optional, Tuple, Any

import numpy as np
import logging
from com.amazon.randomcutforest.parkservices import ThresholdedRandomCutForest
from com.amazon.randomcutforest.config import Precision
from com.amazon.randomcutforest.parkservices import AnomalyDescriptor
from com.amazon.randomcutforest.config import TransformMethod
import jpype

class TRandomCutForestModel:
    """
    Random Cut Forest Python Binding around the AWS Random Cut Forest Official Java version:
    https://github.com/aws/random-cut-forest-by-aws
    """

    def __init__(self, rcf_dimensions, shingle_size, num_trees: int = 30, output_after: int=256, anomaly_rate=0.005,
                 z_factor=2.5, score_differencing=0.5, ignore_delta_threshold=0, sample_size=256):
        self.forest = (ThresholdedRandomCutForest
        .builder()
        .dimensions(rcf_dimensions)
        .sampleSize(sample_size)
        .numberOfTrees(num_trees)
        .timeDecay(0.0001)
        .initialAcceptFraction(output_after*1.0/sample_size)
        .parallelExecutionEnabled(True)
        .compact(True)
        .precision(Precision.FLOAT_32)
        .boundingBoxCacheFraction(1)
        .shingleSize(shingle_size)
        .anomalyRate(anomaly_rate)
        .outputAfter(output_after)
        .internalShinglingEnabled(True)
        .transformMethod(TransformMethod.NORMALIZE)
        .alertOnce(True)
        .autoAdjust(True)
        .build())
        self.forest.setZfactor(z_factor)

    def process(self, point: List[float]) -> AnomalyDescriptor:
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
        return self.forest.process(point, 0)
