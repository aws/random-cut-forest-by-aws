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

package com.amazon.randomcutforest.imputation;

import static com.amazon.randomcutforest.CommonUtils.checkArgument;
import static java.lang.Math.min;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import com.amazon.randomcutforest.returntypes.ConditionalTreeSample;
import com.amazon.randomcutforest.returntypes.SampleSummary;
import com.amazon.randomcutforest.summarization.Summarizer;
import com.amazon.randomcutforest.util.Weighted;

public class ConditionalSampleSummarizer {

    /**
     * this limits the number of valueswe would see per dimension; note that it may
     * be hard to interpret a larger list
     */
    public static int MAX_NUMBER_OF_TYPICAL_PER_DIMENSION = 2;

    /**
     * the maximum size of the typical points array, irrespective of the number of
     * missing dimensions
     */
    public static int MAX_NUMBER_OF_TYPICAL_ELEMENTS = 5;

    /**
     * the array of missing dimension indices
     */
    protected int[] missingDimensions;

    /**
     * the query point, where we are inferring the missing values indicated by
     * missingDimensions[0], missingDimensions[1], ... etc.
     */
    protected float[] queryPoint;

    /**
     * a control parameter; =0 corresponds to (near) random samples and =1
     * correponds to more central (low anomaly score) samples
     */
    protected double centrality;

    /**
     * a boolean that determines if the summarization should use the full dimensions
     * or only the missing dimensions.
     */
    protected boolean project = true;

    public ConditionalSampleSummarizer(int[] missingDimensions, float[] queryPoint, double centrality) {
        this.missingDimensions = Arrays.copyOf(missingDimensions, missingDimensions.length);
        this.queryPoint = Arrays.copyOf(queryPoint, queryPoint.length);
        this.centrality = centrality;
    }

    public ConditionalSampleSummarizer(int[] missingDimensions, float[] queryPoint, double centrality,
            boolean project) {
        this.missingDimensions = Arrays.copyOf(missingDimensions, missingDimensions.length);
        this.queryPoint = Arrays.copyOf(queryPoint, queryPoint.length);
        this.centrality = centrality;
        this.project = project;
    }

    public SampleSummary summarize(List<ConditionalTreeSample> alist) {
        checkArgument(alist.size() > 0, "incorrect call to summarize");
        return summarize(alist, true);
    }

    public SampleSummary summarize(List<ConditionalTreeSample> alist, boolean addTypical) {
        /**
         * first we dedup over the points in the pointStore -- it is likely, and
         * beneficial that different trees acting as different predictors in an ensemble
         * predict the same point that has been seen before. This would be specially
         * true if the time decay is large -- then the whole ensemble starts to behave
         * as a sliding window.
         *
         * note that it is possible that two different *points* predict the same missing
         * value especially when values are repeated in time. however that check of
         * equality of points would be expensive -- and one mechanism is to use a tree
         * (much like an RCT) to test for equality. We will try to not perform such a
         * test.
         */

        double totalWeight = alist.size();
        List<ConditionalTreeSample> newList = ConditionalTreeSample.dedup(alist);

        newList.sort((o1, o2) -> Double.compare(o1.distance, o2.distance));

        ArrayList<Weighted<float[]>> points = new ArrayList<>();
        newList.stream().forEach(e -> {
            if (!project) {
                points.add(new Weighted<>(e.leafPoint, (float) e.weight));
            } else {
                float[] values = new float[missingDimensions.length];
                for (int i = 0; i < missingDimensions.length; i++) {
                    values[i] = e.leafPoint[missingDimensions[i]];
                }
                points.add(new Weighted<>(values, (float) e.weight));
            }
        });

        if (!addTypical) {
            return new SampleSummary(points);
        }

        /**
         * for centrality = 0; there will be no filtration for centrality = 1; at least
         * half the values will be present -- the sum of distance(P33) + distance(P50)
         * appears to be slightly more reasonable than 2 * distance(P50) the distance 0
         * elements correspond to exact matches (on the available fields)
         *
         * it is an open question is the weight of such points should be higher. But if
         * one wants true dynamic adaptability then such a choice to increase weights of
         * exact matches would go against the dynamic sampling based use of RCF.
         **/

        int dimensions = queryPoint.length;

        double threshold = centrality * newList.get(0).distance;
        double currentWeight = 0;
        int alwaysInclude = 0;
        double remainderWeight = totalWeight;
        while (alwaysInclude < newList.size() && newList.get(alwaysInclude).distance == 0) {
            remainderWeight -= newList.get(alwaysInclude).weight;
            ++alwaysInclude;
        }
        for (int j = 1; j < newList.size(); j++) {
            if ((currentWeight < remainderWeight / 3 && currentWeight + newList.get(j).weight >= remainderWeight / 3)
                    || (currentWeight < remainderWeight / 2
                            && currentWeight + newList.get(j).weight >= remainderWeight / 2)) {
                threshold = centrality * newList.get(j).distance;
            }
            currentWeight += newList.get(j).weight;
        }
        // note that the threshold is currently centrality * (some distance in the list)
        // thus the sequel uses a convex combination; and setting centrality = 0 removes
        // the entire filtering based on distances
        threshold += (1 - centrality) * newList.get(newList.size() - 1).distance;
        int num = 0;
        while (num < newList.size() && newList.get(num).distance <= threshold) {
            ++num;
        }

        ArrayList<Weighted<float[]>> typicalPoints = new ArrayList<>();
        for (int j = 0; j < newList.size(); j++) {
            ConditionalTreeSample e = newList.get(j);

            float[] values;
            if (project) {
                values = new float[missingDimensions.length];
                for (int i = 0; i < missingDimensions.length; i++) {
                    values[i] = e.leafPoint[missingDimensions[i]];
                }
            } else {
                values = Arrays.copyOf(e.leafPoint, dimensions);
            }
            if (j < num) { // weight is changed for clustering,
                // based on the distance of the sample from the query point
                double weight = (e.distance <= threshold) ? e.weight : e.weight * threshold / e.distance;
                typicalPoints.add(new Weighted<>(values, (float) weight));
            }
        }
        int maxAllowed = min(queryPoint.length * MAX_NUMBER_OF_TYPICAL_PER_DIMENSION, MAX_NUMBER_OF_TYPICAL_ELEMENTS);
        maxAllowed = min(maxAllowed, num);
        SampleSummary projectedSummary = Summarizer.summarize(typicalPoints, maxAllowed, num, false);

        float[][] pointList = new float[projectedSummary.summaryPoints.length][];
        float[] likelihood = new float[projectedSummary.summaryPoints.length];

        for (int i = 0; i < projectedSummary.summaryPoints.length; i++) {
            pointList[i] = Arrays.copyOf(queryPoint, dimensions);
            for (int j = 0; j < missingDimensions.length; j++) {
                pointList[i][missingDimensions[j]] = projectedSummary.summaryPoints[i][j];
            }
            likelihood[i] = projectedSummary.relativeWeight[i];
        }

        return new SampleSummary(points, pointList, likelihood);
    }

}
