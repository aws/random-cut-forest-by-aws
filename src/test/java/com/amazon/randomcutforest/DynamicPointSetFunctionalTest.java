/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import java.util.List;

import com.amazon.randomcutforest.returntypes.DensityOutput;
import com.amazon.randomcutforest.returntypes.Neighbor;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import static com.amazon.randomcutforest.util.ExampleDataSets.generateFan;
import static java.lang.Math.PI;
import static java.lang.Math.cos;
import static java.lang.Math.sin;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

@Tag("functional")
public class DynamicPointSetFunctionalTest {

    private static int numberOfTrees;
    private static int sampleSize;
    private static int dimensions;
    private static int randomSeed;
    private static RandomCutForest parallelExecutionForest;
    private static RandomCutForest singleThreadedForest;
    private static RandomCutForest forestSpy;

    private static double baseMu;
    private static double baseSigma;
    private static double anomalyMu;
    private static double anomalySigma;
    private static double transitionToAnomalyProbability;
    private static double transitionToBaseProbability;
    private static int dataSize;

    public double [] rotateClockWise(double [] point, double theta){
        assertTrue(point.length==2);
        double [] result=new double [2];
        result[0] = cos(theta)*point[0]+sin(theta)*point[1];
        result[1] = -sin(theta)*point[0]+cos(theta)*point[1];
        return result;
    }
/*
    @Test
    public void movingDensity() {
        int newDimensions = 2;
        randomSeed = 123;

        RandomCutForest newForest = RandomCutForest.builder()
                .numberOfTrees(100)
                .sampleSize(256)
                .dimensions(newDimensions)
                .randomSeed(randomSeed)
                .windowSize(800)
                .centerOfMassEnabled(true)
                .storeSequenceIndexesEnabled(true)
                .build();

        double[][] data = generateFan(1000, 3);

        double[] queryPoint = new double[]{0.7, 0};
        for (int degree = 0; degree < 360; degree += 2) {
            for (int j = 0; j < data.length; j++) {
                newForest.update(rotateClockWise(data[j], 2 * PI * degree / 360));
            }
            DensityOutput density = newForest.getSimpleDensity(queryPoint);
            double value = density.getDensity(0.001, 2);
            if ((degree <= 60) || ((degree >= 120) && (degree <= 180))
                    || ((degree >= 240) && (degree <= 300)))
                assertTrue(value < 0.8);   // the fan is above at 90,210,330

            if (((degree >= 75) && (degree <= 105)) || ((degree >= 195) && (degree <= 225)) ||
                    ((degree >= 315) && (degree <= 345)))
                assertTrue(value > 0.5); // fan is close by
            //intentionally 0.5 is below 0.8 for a robust test

            // Testing for directionality
            //  There can be unclear directionality when the blades are right above

            double bladeAboveInY = density.getDirectionalDensity(0.001, 2).low[1];
            double bladeBelowInY = density.getDirectionalDensity(0.001, 2).high[1];
            double bladesToTheLeft = density.getDirectionalDensity(0.001, 2).high[0];
            double bladesToTheRight = density.getDirectionalDensity(0.001, 2).low[0];


            assertEquals(value, bladeAboveInY + bladeBelowInY + bladesToTheLeft + bladesToTheRight, 1E-6);

            // the tests below have a freedom of 10% of the total value
            if (((degree >= 75) && (degree <= 86)) || ((degree >= 195) && (degree <= 206)) ||
                    ((degree >= 315) && (degree <= 326))) {
                assertTrue(bladeAboveInY + 0.1 * value > bladeBelowInY);
                assertTrue(bladeAboveInY + 0.1 * value > bladesToTheRight);
            }

            if (((degree >= 94) && (degree <= 105)) || ((degree >= 214) && (degree <= 225)) ||
                    ((degree >= 334) && (degree <= 345))) {
                assertTrue(bladeBelowInY + 0.1 * value > bladeAboveInY);
                assertTrue(bladeBelowInY + 0.1 * value > bladesToTheRight);
            }

            if (((degree >= 60) && (degree <= 75)) || ((degree >= 180) && (degree <= 195)) ||
                    ((degree >= 300) && (degree <= 315))) {
                assertTrue(bladeAboveInY + 0.1 * value > bladesToTheLeft);
                assertTrue(bladeAboveInY + 0.1 * value > bladesToTheRight);
            }

            if (((degree >= 105) && (degree <= 120)) || ((degree >= 225) && (degree <= 240)) ||
                    (degree >= 345)) {
                assertTrue(bladeBelowInY + 0.1 * value > bladesToTheLeft);
                assertTrue(bladeBelowInY + 0.1 * value > bladesToTheRight);
            }

            // fans are farthest to the left at 30,150 and 270
            if (((degree >= 15) && (degree <= 45)) || ((degree >= 135) && (degree <= 165))
                    || ((degree >= 255) && (degree <= 285))) {
                assertTrue(bladesToTheLeft + 0.1 * value > bladeAboveInY + bladeBelowInY + bladesToTheRight);
                assertTrue(bladeAboveInY + bladeBelowInY + 0.1 * value > bladesToTheRight);
            }

        }

    }

    @Test
    public void movingNeighbors() {
        int newDimensions = 2;
        randomSeed = 123;

        RandomCutForest newForest = RandomCutForest.builder()
                .numberOfTrees(100)
                .sampleSize(256)
                .dimensions(newDimensions)
                .randomSeed(randomSeed)
                .windowSize(800)
                .centerOfMassEnabled(true)
                .storeSequenceIndexesEnabled(true)
                .build();

        double[][] data = generateFan(1000, 3);

        double[] queryPoint = new double[]{0.7, 0};
        for (int degree = 0; degree < 360; degree += 2) {
            for (int j = 0; j < data.length; j++) {
                newForest.update(rotateClockWise(data[j], 2 * PI * degree / 360));
            }
            List<Neighbor> ans=newForest.getNearNeighborsInSample(queryPoint,1);
            List<Neighbor> closeNeighBors=newForest.getNearNeighborsInSample(queryPoint,0.1);
            Neighbor best = null;
            if (ans!=null) {
                best = ans.get(0);
                for (int j = 1; j < ans.size(); j++) {
                    assert (ans.get(j).distance >= best.distance);
                }
            }

            // fan is away at 30, 150 and 270
            if (((degree>15) && (degree<45))|| ((degree >= 135) && (degree <= 165))
                    || ((degree >= 255) && (degree <= 285))) {
                assertTrue(closeNeighBors.size()==0); // no close neighbor
                assertTrue(best.distance>0.3);
            }

            // fan is overhead at 90, 210 and 330
            if (((degree>75) && (degree<105))|| ((degree >= 195) && (degree <= 225))
                    || ((degree >= 315) && (degree <= 345))) {
                assertTrue(closeNeighBors.size()>0);
                assertEquals(closeNeighBors.get(0).distance,best.distance,1E-10);
            }

        }

    }
*/
}
