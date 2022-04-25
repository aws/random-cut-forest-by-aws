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
import java.util.Random;
import java.util.function.BiFunction;

import com.amazon.randomcutforest.returntypes.SampleSummary;

public class Summarizer {

    /**
     * the following determines the ratio between the sum of the (average) radius
     * and the sepration between centers for a merge; ratio greater than 1 means
     * significant overlap a ratio of 0 means merge closest pairs without
     * consideration of separartion
     *
     **/
    public static double SEPARATION_RATIO_FOR_MERGE = 0.8;

    public static int PHASE2_THRESHOLD = 2;

    public static int LENGTH_BOUND = 1000;

    /**
     * adds a center corresponding to point number index in the list, provided there
     * is already no other center with the same coordinates (or rather, ruling out
     * distance 0 from other centers)
     * 
     * @param points  a list of points
     * @param centers a list of current centers
     * @param dist    distance function
     * @param index   the index of the point to be added as a center
     */
    public static void addCenter(ArrayList<ProjectedPoint> points, ArrayList<Center> centers,
            BiFunction<float[], float[], Double> dist, int index) {
        double minDist = Double.MAX_VALUE;
        for (int i = 0; i < centers.size(); i++) {
            double t = dist.apply(centers.get(i).coordinate, points.get(index).coordinate);
            minDist = min(t, minDist);
        }
        if (minDist > 0.0) {
            centers.add(new Center(points.get(index).coordinate, points.get(index).weight));
        }
    }

    /**
     * picks an item such that the prefix sum of the weight of the projectedPoints
     * is at most wt
     * 
     * @param points a list of projected points (points with weight)
     * @param wt     a selection parameter
     * @return the index of the point for which the condition is satisfied (or the
     *         last index)
     */
    public static int pick(ArrayList<ProjectedPoint> points, double wt) {
        int position = 0;
        double running = wt;
        for (int i = 0; i < points.size(); i++) {
            position = i;
            if (running - points.get(i).weight <= 0.0) {
                break;
            } else {
                running -= points.get(i).weight;
            }
        }
        return position;
    }

    /**
     * creates a sample of points based on an input list; it first adds all points
     * with large weights (0.1%, controlled by LENGTH_BOUND), and in the remainder
     * performs a standard MonteCarlo sampling (adjusting the weight of the sampled
     * elements keeping the total approximately the same) The numbers are chosen
     * such that the total number would be not much more than 10 * LENGTH_BOUND Note
     * that performing a sampling before grouping can be beneficial both in speed as
     * well as denoising see https://en.wikipedia.org/wiki/CURE_algorithm and
     * https://en.wikipedia.org/wiki/Data_stream_clustering the current sampling can
     * be used with intended spherical clusters as in here, or in future
     * non-spherical clusters as well
     *
     * @param points      list of points with weights
     * @param seed        a randomseed to control/reproduce the sampling
     * @param totalWeight the sum of weights of the points (this can be approximate)
     * @return a sample of points (preserving the same order)
     */

    static ArrayList<ProjectedPoint> createSample(ProjectedPoint[] points, long seed, double totalWeight) {
        ArrayList<ProjectedPoint> sampledPoints = new ArrayList<>();
        Random rng = new Random(seed);
        if (points.length < 5 * LENGTH_BOUND) {
            sampledPoints.addAll(Arrays.asList(points));
        } else {
            double remainder = 0.0;
            for (int j = 0; j < points.length; j++) {
                if (points[j].weight > totalWeight / LENGTH_BOUND) {
                    sampledPoints.add(new ProjectedPoint(points[j].coordinate, points[j].weight));
                } else {
                    remainder += points[j].weight;
                }
            }
            for (int j = 0; j < points.length; j++) {
                if ((points[j].weight <= totalWeight / LENGTH_BOUND)
                        && (rng.nextDouble() < 5.0 * LENGTH_BOUND / points.length)) {
                    double t = points[j].weight * (points.length / (5.0 * LENGTH_BOUND)) * (remainder / totalWeight);
                    sampledPoints.add(new ProjectedPoint(points[j].coordinate, (float) t));
                }
            }
        }
        return sampledPoints;
    }

    /**
     * the following function returns a summary of the input points
     * 
     * @param points          points with associated weights
     * @param maxAllowed      the maximum number of clusters/summary points
     * @param initial         the initial number of clusters/summary points, chosen
     *                        at random
     * @param reassignPerStep a boolean which controls if an EM like step is
     *                        performed every step; by default, such a step is not
     *                        performed until the number of clusters fall below
     *                        PHASE_THRESHOLD times the maximum number of clusters
     *                        allowed
     * @param distance        a distance function for the points, that determines
     *                        the order of the reverse delete however the EM like
     *                        step uses L1 measure (to be robust to noise)
     * @param seed            a random seed for controlling the randomness
     * @return a summary of the input points (Note: the median returned is an
     *         approximate median; exact computation is unlikely to be critical for
     *         true applications of summarization)
     */
    public static SampleSummary summarize(ProjectedPoint[] points, int maxAllowed, int initial, boolean reassignPerStep,
            BiFunction<float[], float[], Double> distance, long seed) {
        checkArgument(maxAllowed < 100, "are you sure you want more elements in the summary?");
        checkArgument(maxAllowed <= initial, "initial parameter should be at least maximum allowed in final result");

        double totalWeight = Arrays.stream(points).map(e -> (double) e.weight).reduce(0.0, Double::sum);
        checkArgument(!Double.isNaN(totalWeight) && Double.isFinite(totalWeight), " weights are incorrect");
        ArrayList<Center> centers = new ArrayList<>();
        Random rng = new Random(seed);
        ArrayList<ProjectedPoint> sampledPoints = createSample(points, rng.nextLong(), totalWeight);
        double sampledSum = Arrays.stream(points).map(e -> (double) e.weight).reduce(0.0, Double::sum);

        if (sampledPoints.size() < 10 * initial) {
            for (int k = 0; k < sampledPoints.size(); k++) {
                addCenter(sampledPoints, centers, distance, k);
            }
        } else {
            for (int k = 0; k < 2 * initial; k++) {
                double wt = rng.nextDouble() * sampledSum;
                addCenter(sampledPoints, centers, distance, pick(sampledPoints, wt));
            }
        }
        return iterateOverSamples(maxAllowed, reassignPerStep, sampledPoints, centers, distance, true,
                SEPARATION_RATIO_FOR_MERGE);
    }

    /**
     * Same as previous over a flat collection of unweighted float[]
     * 
     * @param points          points represented by float[][]
     * @param maxAllowed      maximum number of clusters in output
     * @param initial         initial number of points to seed; a control parameter
     *                        that serves both as a denoiser, as well as as a
     *                        facilitator of coninuity (large numbers would
     *                        correspond to MST like clustering)
     * @param reassignPerStep a boolean deciding if reassignment is performed each
     *                        step
     * @param distance        distance metric over float []
     * @param seed            random seed
     * @return a list of centers with weights
     */
    public static SampleSummary summarize(float[][] points, int maxAllowed, int initial, boolean reassignPerStep,
            BiFunction<float[], float[], Double> distance, long seed) {
        ProjectedPoint[] projectedPoints = new ProjectedPoint[points.length];
        for (int i = 0; i < points.length; i++) {
            // the following does not copy the vectors
            projectedPoints[i] = new ProjectedPoint(points[i], 1.0f);
        }
        return summarize(projectedPoints, maxAllowed, initial, reassignPerStep, distance, seed);
    }

    /**
     * same as before with common cases filled in, used in analysis of
     * ConditionalSamples
     * 
     * @param points          points in ProjectedPoint{}
     * @param maxAllowed      maximum number of groups/clusters
     * @param initial         a parameter controlling the initialization
     * @param reassignPerStep if reassignment is to be performed each step
     * @return a summarization
     */
    public static SampleSummary summarize(ProjectedPoint[] points, int maxAllowed, int initial,
            boolean reassignPerStep) {
        return summarize(points, maxAllowed, initial, reassignPerStep, Summarizer::L2distance, 42);
    }

    /**
     * Same as above, with the most common use cases filled in
     * 
     * @param points     points in float[][], each of weight 1.0
     * @param maxAllowed maximum number of clusters one is interested in
     * @return a summarization
     */
    public static SampleSummary summarize(float[][] points, int maxAllowed) {
        return summarize(points, maxAllowed, 4 * maxAllowed, false, Summarizer::L2distance, 42);
    }

    /**
     * the core subroutine that performs a potentially 3-phase (but configurable)
     * aggregation into clusters/groups In phase 1, after an initial reassignment
     * step, the number is reduced to 2 * maxAllowed; in phase II that number is
     * further reduced to maxAllowed (with reassignement at each step), and finally
     * a third phase that continues to reduce further (till 1, as long as the
     * overlap condition is met). Phase III can be turned off via the boolean, but
     * the overlapParameter would change the order of merges in phases I and II.
     * Setting this to DOUBLE.MAX_VALUE (not recommended) would merge the closest
     * pairs recursively.
     *
     * @param maxAllowed       the maximum number of target groups
     * @param reassignPerStep  if reassignment is performed at every step
     *                         (reassignment is performed by default when the number
     *                         of clusters are at or below 2 * maxAllowed)
     * @param sampledPoints    the list of sampled points (may or may not be the
     *                         original list of points)
     * @param centers          initial list of centers
     * @param distance         a distance metric
     * @param continuePast     a boolean that indicates if we should continue to
     *                         seek smaller grouping even when the number is below
     *                         maxAllowed; this is useful because true number of
     *                         groupings is seldom known
     * @param overlapParameter a parameter that controls merging ordering as well
     *                         merge at very low number of groups a larger value
     *                         would correspond to larger number of clusters each of
     *                         which is more compactly knit -- a lower value would
     *                         correspond to larger clusters.
     * @return
     */

    static SampleSummary iterateOverSamples(int maxAllowed, boolean reassignPerStep, List<ProjectedPoint> sampledPoints,
            ArrayList<Center> centers, BiFunction<float[], float[], Double> distance, boolean continuePast,
            double overlapParameter) {

        double totalWeight = sampledPoints.stream().map(e -> e.weight).reduce(Float::sum).get();
        int dimensions = sampledPoints.get(0).coordinate.length;
        if (reassignPerStep) {
            Center.assign(centers.size(), sampledPoints, centers, distance);
        }
        // assignment would change weights, sorting in non-decreasing order
        centers.sort((o1, o2) -> Double.compare(o1.weight, o2.weight));
        while (centers.get(0).weight == 0) {
            centers.remove(0);
        }

        double phase3Distance = 0;
        boolean keepReducingCenters = (centers.size() > maxAllowed);

        while (keepReducingCenters) {
            double measure = 0;
            double measureDist = Double.MAX_VALUE;
            int lower = 0;
            int firstOfMerge = lower;
            int secondOfMerge = lower + 1;// will be reset before exiting the loop
            boolean foundMerge = false;

            while (lower < centers.size() - 1 && !foundMerge) {
                // we will keep searching
                double minDist = Double.MAX_VALUE;
                int minNbr = -1;
                for (int j = lower + 1; j < centers.size(); j++) {
                    double dist = distance.apply(centers.get(lower).coordinate, centers.get(j).coordinate);
                    if (minDist > dist) {
                        minNbr = j;
                        minDist = dist;
                    }
                    double numerator = (centers.get(lower).radius() + centers.get(j).radius() + phase3Distance);
                    if (numerator >= overlapParameter * dist) { // note 0 >= 0
                        if (measure * dist < numerator) {
                            firstOfMerge = lower;
                            secondOfMerge = j;
                            if (dist == 0) {
                                foundMerge = true;
                            } else {
                                measure = numerator / dist;
                            }
                            measureDist = dist;
                        }
                    }
                }
                if (lower == 0 && !foundMerge) {
                    measureDist = minDist;
                    // this is set assuming we may be interested in merging the minimum weight
                    // cluster
                    // which corresponds to lower == 0
                    secondOfMerge = minNbr;
                }
                ++lower;
            }

            int inital = centers.size();
            if (inital > maxAllowed || foundMerge || (continuePast && measure > overlapParameter)) {
                centers.get(secondOfMerge).mergeInto(centers.get(firstOfMerge), distance);
                centers.remove(firstOfMerge);
                if (reassignPerStep || centers.size() <= PHASE2_THRESHOLD * maxAllowed) {
                    Center.assign(centers.size(), sampledPoints, centers, distance);
                }
                centers.sort((o1, o2) -> Double.compare(o1.weight, o2.weight));
                while (centers.get(0).weight == 0.0) {
                    centers.remove(0);
                }
                if (inital > maxAllowed && centers.size() <= maxAllowed) {
                    // phase 3 kicks in; but this will execute at most once
                    // note that measureDist can be 0 as well
                    phase3Distance = measureDist;
                }
            } else {
                keepReducingCenters = false;
            }
        }

        // sort in decreasing weight
        centers.sort((o1, o2) -> Double.compare(o2.weight, o1.weight));
        float[][] pointList = new float[centers.size()][];
        float[] likelihood = new float[centers.size()];

        for (int i = 0; i < centers.size(); i++) {
            pointList[i] = Arrays.copyOf(centers.get(i).coordinate, dimensions);
            likelihood[i] = (float) (centers.get(i).weight / totalWeight);
        }

        return new SampleSummary(totalWeight, pointList, likelihood, new float[dimensions], new float[dimensions],
                new float[dimensions]);
    }

    private static Double L1distance(float[] a, float[] b) {
        double dist = 0;
        for (int i = 0; i < a.length; i++) {
            dist += Math.abs(a[i] - b[i]);
        }
        return dist;
    }

    private static Double L2distance(float[] a, float[] b) {
        double dist = 0;
        for (int i = 0; i < a.length; i++) {
            double t = Math.abs(a[i] - b[i]);
            dist += t * t;
        }
        return Math.sqrt(dist);
    }
}
