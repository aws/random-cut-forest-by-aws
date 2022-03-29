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
import static java.lang.Math.max;
import static java.lang.Math.min;
import static java.lang.Math.sqrt;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import com.amazon.randomcutforest.returntypes.ConditionalTreeSample;
import com.amazon.randomcutforest.returntypes.SampleSummary;

public class ConditionalSampleSummarizer {

    /**
     * the following determines the ratio between the sum of the (average) radius
     * and the sepration between centers for a merge; ratio greater than 1 means
     * significant overlap a ratio of 0 means merge closest pairs without
     * consideration of separartion
     **/
    public static double SEPARATION_RATIO_FOR_MERGE = 0.8;

    /**
     * a factor that controls weight assignment for soft clustering; this is the
     * multiple of the minimum distance and should be greater or equal 1.
     */
    public static double WEIGHT_ALLOCATION_THRESHOLD = 1.25;

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

    public ConditionalSampleSummarizer(int[] missingDimensions, float[] queryPoint, double centrality) {
        this.missingDimensions = Arrays.copyOf(missingDimensions, missingDimensions.length);
        this.queryPoint = Arrays.copyOf(queryPoint, queryPoint.length);
        this.centrality = centrality;
    }

    public SampleSummary summarize(List<ConditionalTreeSample> alist) {
        checkArgument(alist.size() > 0, "incorrect call to summarize");
        /**
         * first we dedupe over the points in the pointStore -- it is likely, and
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

        newList.sort((o1, o2) -> Double.compare(o1.distance, o2.distance));
        double threshold = 0;
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
                threshold += centrality * newList.get(j).distance;
            }
            currentWeight += newList.get(j).weight;
        }
        threshold += (1 - centrality) * newList.get(newList.size() - 1).distance;
        int num = 0;
        while (num < newList.size() && newList.get(num).distance <= threshold) {
            ++num;
        }

        /**
         * in the sequel we will create a global synopsis as well as a local one the
         * filtering based on thresholds will apply to the local one (points)
         */
        float[] coordMean = new float[queryPoint.length];
        double[] coordSqSum = new double[queryPoint.length];
        Center center = new Center(missingDimensions.length);

        ProjectedPoint[] points = new ProjectedPoint[num];
        for (int j = 0; j < newList.size(); j++) {
            ConditionalTreeSample e = newList.get(j);

            float[] values = new float[missingDimensions.length];
            for (int i = 0; i < missingDimensions.length; i++) {
                values[i] = e.leafPoint[missingDimensions[i]];
            }
            center.add(values, e.weight);
            for (int i = 0; i < coordMean.length; i++) {
                coordMean[i] += e.leafPoint[i] * e.weight; // weight unchanges
                coordSqSum[i] += e.leafPoint[i] * e.leafPoint[i] * e.weight;
            }
            if (j < num) { // weight is changed for clustering,
                // based on the distance of the sample from the query point
                double weight = (e.distance <= threshold) ? e.weight : e.weight * threshold / e.distance;
                points[j] = new ProjectedPoint(values, weight);
            }
        }
        // we compute p50 over the entire set
        float[] median = Arrays.copyOf(queryPoint, queryPoint.length);
        center.recompute();
        for (int y = 0; y < missingDimensions.length; y++) {
            median[missingDimensions[y]] = (float) center.coordinate[y];
        }
        // we compute deviation over the entire set, using original weights and no
        // filters
        float[] deviation = new float[queryPoint.length];
        for (int j = 0; j < coordMean.length; j++) {
            coordMean[j] = coordMean[j] / (float) totalWeight;
            deviation[j] = (float) sqrt(max(0, coordSqSum[j] / totalWeight - coordMean[j] * coordMean[j]));
        }

        /**
         * we now seed the centers according toa farthest point heuristic; such a
         * heuristic is used in clustering algorithms such as CURE:
         * https://en.wikipedia.org/wiki/CURE_algorithm to represent a cluster using
         * multiple points (multi-centroid approach) In that algorithm closest pairs are
         * merged -- the notion of closest will be altered here
         *
         * the first step is initialization to twice the final maximum number of
         * clusters
         */

        ArrayList<Center> centers = new ArrayList<>();
        centers.add(new Center(center.coordinate));
        int maxAllowed = min(center.coordinate.length * MAX_NUMBER_OF_TYPICAL_PER_DIMENSION,
                MAX_NUMBER_OF_TYPICAL_ELEMENTS);
        for (int k = 0; k < 2 * maxAllowed; k++) {
            double maxDist = 0;
            int maxIndex = -1;
            for (int j = 0; j < points.length; j++) {
                double minDist = Double.MAX_VALUE;
                for (int i = 0; i < centers.size(); i++) {
                    minDist = min(minDist, distance(points[j], centers.get(i)));
                }
                if (minDist > maxDist) {
                    maxDist = minDist;
                    maxIndex = j;
                }
            }
            if (maxDist == 0) {
                break;
            } else {
                centers.add(new Center(Arrays.copyOf(points[maxIndex].coordinate, points[maxIndex].coordinate.length)));
            }
        }

        /**
         * we will now prune the number of clusters iteratively; the first step will be
         * assignment of points the next step would be choosing the optimum centers
         * given the assignment
         */
        double measure = 10;
        do {
            for (int i = 0; i < centers.size(); i++) {
                centers.get(i).reset();
            }
            double maxDist = 0;
            for (int j = 0; j < points.length; j++) {
                double[] dist = new double[centers.size()];
                Arrays.fill(dist, Double.MAX_VALUE);
                double minDist = Double.MAX_VALUE;
                for (int i = 0; i < centers.size(); i++) {
                    dist[i] = distance(points[j], centers.get(i));
                    minDist = min(minDist, dist[i]);
                }
                if (minDist == 0) {
                    for (int i = 0; i < centers.size(); i++) {
                        if (dist[i] == 0) {
                            centers.get(i).add(points[j].coordinate, points[j].weight);
                        }
                    }
                } else {
                    maxDist = max(maxDist, minDist);
                    double sum = 0;
                    for (int i = 0; i < centers.size(); i++) {
                        if (dist[i] <= WEIGHT_ALLOCATION_THRESHOLD * minDist) {
                            sum += minDist / dist[i]; // setting up harmonic mean
                        }
                    }
                    for (int i = 0; i < centers.size(); i++) {
                        if (dist[i] == 0) {
                            centers.get(i).add(points[j].coordinate, points[j].weight);
                        } else if (dist[i] <= WEIGHT_ALLOCATION_THRESHOLD * minDist) {
                            // harmonic mean
                            centers.get(i).add(points[j].coordinate, points[j].weight * minDist / (dist[i] * sum));
                        }
                    }
                }
            }
            for (int i = 0; i < centers.size(); i++) {
                centers.get(i).recompute();
            }

            /**
             * we now find the "closest" pair and merge them; the smaller weight cluster is
             * merged into the larger weight cluster because of L1 errors
             */
            int first = -1;
            int second = -1;
            measure = 0;
            for (int i = 0; i < centers.size(); i++) {
                for (int j = i + 1; j < centers.size(); j++) {
                    double dist = distance(centers.get(i), centers.get(j));
                    double tempMeasure = (centers.get(i).radius() + centers.get(j).radius()) / dist;
                    if (measure < tempMeasure) {
                        first = i;
                        second = j;
                        measure = tempMeasure;
                    }
                }
            }
            if (measure >= SEPARATION_RATIO_FOR_MERGE) {
                if (centers.get(first).weight < centers.get(second).weight) {
                    centers.remove(first);
                } else {
                    centers.remove(second);
                }
            } else if (centers.size() > maxAllowed) {
                // not well separated, remove small weight cluster centers
                centers.sort((o1, o2) -> Double.compare(o1.weight, o2.weight));
                centers.remove(0);
            }
        } while (centers.size() > maxAllowed || measure >= SEPARATION_RATIO_FOR_MERGE);

        // sort in decreasing weight
        centers.sort((o1, o2) -> Double.compare(o2.weight, o1.weight));
        float[][] pointList = new float[centers.size()][];
        float[] likelihood = new float[centers.size()];

        for (int i = 0; i < centers.size(); i++) {
            pointList[i] = Arrays.copyOf(queryPoint, queryPoint.length);
            for (int j = 0; j < missingDimensions.length; j++) {
                pointList[i][missingDimensions[j]] = centers.get(i).coordinate[j];
            }
            likelihood[i] = (float) (centers.get(i).weight / totalWeight);
        }

        return new SampleSummary(totalWeight, pointList, likelihood, median, coordMean, deviation);
    }

    class ProjectedPoint {
        final float[] coordinate;
        double weight;

        ProjectedPoint(float[] coordinate, double weight) {
            // the following is to avoid copies of points as they are being moved
            // these values should NOT be altered
            this.coordinate = coordinate;
            this.weight = weight;
        }

    }

    static double distance(ProjectedPoint a, ProjectedPoint b) {
        double distance = 0;
        for (int i = 0; i < a.coordinate.length; i++) {
            distance += Math.abs(a.coordinate[i] - b.coordinate[i]);
        }
        return distance;
    }

    class Center extends ProjectedPoint {
        ArrayList<ProjectedPoint> points;
        double sumOfRadius;

        Center(int dimensions) {
            // explicitly copied because array elements will change
            super(new float[dimensions], 0);
            this.points = new ArrayList<>();
        }

        Center(float[] coordinate) {
            // explicitly copied because array elements will change
            super(Arrays.copyOf(coordinate, coordinate.length), 0);
            this.points = new ArrayList<>();
        }

        public void add(float[] coordinate, double weight) {
            points.add(new ProjectedPoint(coordinate, weight));
            this.weight += weight;
        }

        public void reset() {
            points = new ArrayList<>();
            weight = 0;
        }

        public double radius() {
            return (weight > 0) ? sumOfRadius / weight : 0;
        }

        public void recompute() {
            sumOfRadius = 0;
            if (weight == 0) {
                checkArgument(points.size() == 0, "adding 0 weight points?");
                Arrays.fill(coordinate, 0); // zero out values
                return;
            }
            for (int i = 0; i < coordinate.length; i++) {
                int index = i;
                points.sort((o1, o2) -> Double.compare(o1.coordinate[index], o2.coordinate[index]));
                double runningWeight = weight / 2;
                int position = 0;
                while (runningWeight >= 0 && position < points.size()) {
                    if (runningWeight >= points.get(position).weight) {
                        runningWeight -= points.get(position).weight;
                        ++position;
                    } else {
                        break;
                    }
                }
                coordinate[index] = points.get(position).coordinate[index];
                for (int j = 0; j < points.size(); j++) {
                    sumOfRadius += points.get(j).weight * Math.abs(coordinate[index] - points.get(j).coordinate[index]);
                }
            }
        }
    }

    public void assign(ProjectedPoint[] points, List<Center> centers) {
        centers.stream().forEach(x -> {
            x.weight = 0;
        });
        for (int i = 0; i < points.length; i++) {
            double[] distance = new double[centers.size()];
            double minDistance = Double.MAX_VALUE;
            for (int j = 0; j < centers.size(); j++) {
                distance[j] = distance(centers.get(j), points[i]);
                minDistance = min(minDistance, distance[j]);
            }
            double sum = 0;
            for (int j = 0; j < centers.size(); j++) {
                if (distance[j] <= 1.25 * minDistance) {
                    if (distance[j] > 0) {
                        sum += minDistance / distance[j];
                    } else {
                        sum += 1;
                    }
                }
            }
            for (int j = 0; j < centers.size(); j++) {
                if (distance[j] <= 1.25 * minDistance) {
                    if (distance[j] == 0) {
                        centers.get(j).add(points[i].coordinate, 1 / sum);
                    } else {
                        centers.get(j).add(points[i].coordinate, minDistance / (sum * distance[j]));
                    }
                }
            }
        }
    }

}
