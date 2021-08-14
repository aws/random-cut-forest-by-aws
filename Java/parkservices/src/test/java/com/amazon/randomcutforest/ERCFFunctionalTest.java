/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * A copy of the License is located at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * or in the "license" file accompanying this file. This file is distributed
 * on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 * express or implied. See the License for the specific language governing
 * permissions and limitations under the License.
 */

package com.amazon.randomcutforest;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.Random;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import com.amazon.randomcutforest.config.Precision;
import com.amazon.randomcutforest.parkservices.AnomalyDescriptor;
import com.amazon.randomcutforest.parkservices.threshold.ThresholdedRandomCutForest;
import com.amazon.randomcutforest.parkservices.threshold.ThresholdedRandomCutForestMapper;

@Tag("functional")
public class ERCFFunctionalTest {

    @Test
    public void testRoundTrip() {
        int dimensions = 10;
        for (int trials = 0; trials < 10; trials++) {

            long seed = new Random().nextLong();
            RandomCutForest.Builder builder = RandomCutForest.builder().compact(true).dimensions(dimensions)
                    .precision(Precision.FLOAT_32).randomSeed(seed);

            ThresholdedRandomCutForest first = new ThresholdedRandomCutForest(builder, 0.01);
            ThresholdedRandomCutForest second = new ThresholdedRandomCutForest(builder, 0.01);
            RandomCutForest forest = builder.build();

            Random r = new Random();
            for (int i = 0; i < new Random().nextInt(1000); i++) {
                double[] point = r.ints(dimensions, 0, 50).asDoubleStream().toArray();
                AnomalyDescriptor firstResult = first.process(point);
                AnomalyDescriptor secondResult = second.process(point);

                assertEquals(firstResult.getRcfScore(), secondResult.getRcfScore(), 1e-10);
                assertEquals(firstResult.getRcfScore(), forest.getAnomalyScore(point), 1e-10);
                forest.update(point);
            }

            // serialize + deserialize
            ThresholdedRandomCutForestMapper mapper = new ThresholdedRandomCutForestMapper();
            ThresholdedRandomCutForest third = mapper.toModel(mapper.toState(second));

            // update re-instantiated forest
            for (int i = 0; i < 100; i++) {
                double[] point = r.ints(dimensions, 0, 50).asDoubleStream().toArray();
                AnomalyDescriptor firstResult = first.process(point);
                AnomalyDescriptor secondResult = second.process(point);
                AnomalyDescriptor thirdResult = third.process(point);
                double score = forest.getAnomalyScore(point);
                assertEquals(score, firstResult.getRcfScore(), 1e-10);
                assertEquals(score, secondResult.getRcfScore(), 1e-10);
                assertEquals(score, thirdResult.getRcfScore(), 1e-10);
                forest.update(point);
            }
        }
    }

}
