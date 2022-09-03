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

package com.amazon.randomcutforest.summarization;

import static com.amazon.randomcutforest.CommonUtils.checkArgument;
import static com.amazon.randomcutforest.util.Weighted.createSample;
import static com.amazon.randomcutforest.util.Weighted.prefixPick;
import static java.lang.Math.max;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.function.BiFunction;
import java.util.function.Function;

import com.amazon.randomcutforest.returntypes.SampleSummary;
import com.amazon.randomcutforest.util.Weighted;

public class Summarizer {

    /**
     * a factor that controls weight assignment for soft clustering; this is the
     * multiple of the minimum distance and should be greater or equal 1.
     */
    public static double WEIGHT_ALLOCATION_THRESHOLD = 1.25;

    /**
     * the following determines the ratio between the sum of the (average) radius
     * and the separation between centers for a merge; ratio greater than 1 means
     * significant overlap a ratio of 0 means merge closest pairs without
     * consideration of separartion
     *
     **/
    public static double DEFAULT_SEPARATION_RATIO_FOR_MERGE = 0.8;

    public static int PHASE2_THRESHOLD = 2;

    public static int LENGTH_BOUND = 1000;

    public static Double L1distance(float[] a, float[] b) {
        double dist = 0;
        for (int i = 0; i < a.length; i++) {
            dist += Math.abs(a[i] - b[i]);
        }
        return dist;
    }

    public static Double L2distance(float[] a, float[] b) {
        double dist = 0;
        for (int i = 0; i < a.length; i++) {
            double t = Math.abs(a[i] - b[i]);
            dist += t * t;
        }
        return Math.sqrt(dist);
    }

    public static Double LInfinitydistance(float[] a, float[] b) {
        double dist = 0;
        for (int i = 0; i < a.length; i++) {
            dist = max(Math.abs(a[i] - b[i]), dist);
        }
        return dist;
    }

    /**
     * a function that reassigns points to clusters
     * 
     * @param sampledPoints   a list of sampled points with weights
     * @param clusters        a list of current clusters, because random access to
     *                        the elements is necessary
     * @param distance        a distance function
     * @param parallelEnabled a flag enabling limited parallelism; only during
     *                        cluster by cluster recomputation. Using parallel mode
     *                        during the assignment of points does not seem to help
     */
    public static <R> void assignAndRecompute(List<Weighted<Integer>> sampledPoints, Function<Integer, R> getPoint,
            List<ICluster<R>> clusters, BiFunction<R, R, Double> distance, boolean parallelEnabled) {
        checkArgument(clusters.size() > 0, " cannot be empty list of clusters");
        checkArgument(sampledPoints.size() > 0, " cannot be empty list of points");

        for (ICluster<R> cluster : clusters) {
            cluster.reset();
        }

        for (Weighted<Integer> point : sampledPoints) {
            if (point.weight > 0) {

                double[] dist = new double[clusters.size()];
                Arrays.fill(dist, Double.MAX_VALUE);
                double minDist = Double.MAX_VALUE;
                int minDistNbr = -1;
                for (int i = 0; i < clusters.size(); i++) {
                    dist[i] = clusters.get(i).distance(getPoint.apply(point.index), distance);
                    if (minDist > dist[i]) {
                        minDist = dist[i];
                        minDistNbr = i;
                    }
                    if (minDist == 0) {
                        break;
                    }
                }

                if (minDist == 0) {
                    clusters.get(minDistNbr).addPoint(point.index, point.weight, 0, getPoint.apply(point.index),
                            distance);
                } else {
                    double sum = 0;
                    for (int i = 0; i < clusters.size(); i++) {
                        if (dist[i] <= WEIGHT_ALLOCATION_THRESHOLD * minDist) {
                            sum += minDist / dist[i]; // setting up harmonic mean
                        }
                    }
                    for (int i = 0; i < clusters.size(); i++) {
                        if (dist[i] <= WEIGHT_ALLOCATION_THRESHOLD * minDist) {
                            // harmonic mean
                            clusters.get(i).addPoint(point.index, (float) (point.weight * minDist / (dist[i] * sum)),
                                    dist[i], getPoint.apply(point.index), distance);
                        }
                    }
                }
            }
        }

        if (parallelEnabled) {
            clusters.parallelStream().forEach(e -> e.recompute(getPoint, true, distance));
        } else {
            clusters.stream().forEach(e -> e.recompute(getPoint, true, distance));
        }
    }

    /**
     * The core subroutine for iterative clustering used herein. The clustering
     * algorithm borrows from CURE https://en.wikipedia.org/wiki/CURE_algorithm,
     * which used sampling as a tradeoff of representationa accuracy versus
     * algorithmic efficiency. Note however random sampling can also perform
     * denoising and reduce space as a filtering mechanism. Note that hierarchical
     * iterative merging strategies can be proven to not degrade clustering
     * https://en.wikipedia.org/wiki/Data_stream_clustering with the benefit of
     * small space. The algorithm herein proceeds in three phases, where the first
     * phase corresponds from the initial seeding to about twice the maximum number
     * of clusters one wishes to consider. The second phase corresponds to reducing
     * that number to the maximum allowable number. The third phase corresponds to
     * continuing the clustering as long as the conditions are similar to the end of
     * phase two, thereby enabling us to use a rough estimate for the maximum
     * allowed. By default, recomputation of the cluster makes sense in phases 2 and
     * 3 -- however can be enabled for phase 1 as well, thereby enabling the regular
     * K-Means algorithm to be expressed in the below algorithm as well. The
     * algorithm can also express Minimum Spanning Tree based clustering with
     * repeated merging of closest pair (which is a capability derived from CURE)
     *
     * The primary reason for the number of parameters is the ability to invoke this
     * function without creating a copy of the points (or equivalent large objects),
     * and hence the functions as parameters
     *
     * @param maxAllowed           number of maximum clusters one is interested in
     * @param initial              an initial number of sampled centers to start
     *                             from
     * @param stopAt               a hard lower bound on the number of clusters
     * @param refs                 a (possibly sampled) list of references with
     *                             weight
     * @param getPoint             a function which retrives the point/object given
     *                             an index in the refs
     * @param distance             a distance function
     * @param clusterInitializer   a function that creates a cluster given an object
     *                             aand a weight
     * @param seed                 a random seed
     * @param parallelEnabled      enabling parallel computation in the first phase
     *                             when points are assigned to different sampled
     *                             centers; and the centers are possibly adjusted
     * @param phase2GlobalReassign a flag that determines if the points would be
     *                             reassigned when the clusters fall below 1.2 *
     *                             maxAllowed -- this serves as a denoising.
     * @param overlapParameter     a parameter that controls the ordering of the
     *                             merges as well as the stopping condition of the
     *                             merges
     * @param previousClustering   a possibly null list of clusters seen previously,
     *                             used as zero weight seeds to smoothen the
     *                             continuous clustering
     * @param <R>                  type of object being clustered
     * @return a list of clusters
     */
    public static <R> List<ICluster<R>> iterativeClustering(int maxAllowed, int initial, int stopAt,
            List<Weighted<Integer>> refs, Function<Integer, R> getPoint, BiFunction<R, R, Double> distance,
            BiFunction<R, Float, ICluster<R>> clusterInitializer, long seed, boolean parallelEnabled,
            boolean phase2GlobalReassign, double overlapParameter, List<ICluster<R>> previousClustering) {

        checkArgument(refs.size() > 0, "empty list, nothing to do");
        checkArgument(maxAllowed >= stopAt && stopAt > 0, "incorrect bounds on number of clusters");

        Random rng = new Random(seed);
        double sampledSum = refs.stream().map(e -> (double) e.weight).reduce(Double::sum).get();
        ArrayList<ICluster<R>> centers = new ArrayList<>();
        if (refs.size() < 10 * (initial + 5)) {
            for (Weighted<Integer> point : refs) {
                centers.add(clusterInitializer.apply(getPoint.apply(point.index), 0f));
            }
        } else {
            for (int k = 0; k < 2 * (initial + 5); k++) {
                double wt = rng.nextDouble() * sampledSum;
                Weighted<Integer> picked = prefixPick(refs, wt);
                centers.add(clusterInitializer.apply(getPoint.apply(picked.index), 0f));
            }
        }
        if (previousClustering != null) {
            for (ICluster<R> previousCluster : previousClustering) {
                List<Weighted<R>> representatives = previousCluster.getRepresentatives();
                for (Weighted<R> representative : representatives) {
                    centers.add(clusterInitializer.apply(representative.index, 0f));
                }
            }
        }
        assignAndRecompute(refs, getPoint, centers, distance, parallelEnabled);
        // assignment would change weights, sorting in non-decreasing order
        centers.sort(Comparator.comparingDouble(ICluster::getWeight));
        while (centers.get(0).getWeight() == 0) {
            centers.remove(0);
        }

        double phase3Distance = 0;
        double runningPhase3Distance = 0;
        boolean keepReducingCenters = (centers.size() > maxAllowed);

        while (keepReducingCenters) {
            double measure = 0;
            double measureDist = Double.MAX_VALUE;
            int lower = 0;
            int firstOfMerge = lower;
            int secondOfMerge = lower + 1;// will be reset before exiting the loop
            boolean foundMerge = false;
            double minDist = Double.MAX_VALUE;

            while (lower < centers.size() - 1 && !foundMerge) {
                // we will keep searching
                int minNbr = -1;
                for (int j = lower + 1; j < centers.size(); j++) {
                    double dist = centers.get(lower).distance(centers.get(j), distance);
                    if (dist == 0) {
                        foundMerge = true;
                        firstOfMerge = lower;
                        secondOfMerge = minNbr = j;
                        minDist = measureDist = 0.0;
                        break;
                    } else {
                        if (minDist > dist) {
                            minNbr = j;
                            minDist = dist;
                        }

                        double temp = (centers.get(lower).extentMeasure() + centers.get(j).extentMeasure()
                                + phase3Distance) / dist;
                        if (temp > overlapParameter && measure < temp) {
                            firstOfMerge = lower;
                            secondOfMerge = j;
                            measure = temp;
                            measureDist = dist;
                        }
                    }
                }
                if (lower == 0 && !foundMerge) {
                    measureDist = minDist;
                    // this is set assuming we may be interested in merging the minimum weight
                    // cluster which corresponds to lower == 0
                    secondOfMerge = minNbr;
                }
                ++lower;
            }

            int inital = centers.size();

            if (inital > maxAllowed || foundMerge || (inital > stopAt && measure > overlapParameter)) {
                centers.get(secondOfMerge).absorb(centers.get(firstOfMerge), distance);
                if (phase2GlobalReassign && centers.size() <= PHASE2_THRESHOLD * maxAllowed + 1) {
                    centers.remove(firstOfMerge);
                    assignAndRecompute(refs, getPoint, centers, distance, parallelEnabled);
                } else {
                    centers.get(secondOfMerge).recompute(getPoint, false, distance);
                    centers.remove(firstOfMerge);
                }
                centers.sort(Comparator.comparingDouble(ICluster::getWeight));
                while (centers.get(0).getWeight() == 0.0) {
                    centers.remove(0);
                }
                if (inital < 1.2 * maxAllowed + 1) {
                    // phase 3 kicks in; but this will execute at most once
                    // note that measureDist can be 0 as well
                    runningPhase3Distance = max(runningPhase3Distance, measureDist);
                    if (inital > maxAllowed && centers.size() <= maxAllowed) {
                        phase3Distance = runningPhase3Distance;
                    }
                }
            } else {
                keepReducingCenters = false;
            }
        }
        // sort in decreasing weight
        centers.sort((o1, o2) -> Double.compare(o2.getWeight(), o1.getWeight()));
        return centers;
    }

    /**
     * the following function returns a summary of the input points
     *
     * @param points               points with associated weights
     * @param maxAllowed           the maximum number of clusters/summary points
     * @param initial              the initial number of clusters/summary points,
     *                             chosen at random
     * @param stopAt               a hard lower bound on the number of clusters
     * @param phase2GlobalReassign a flag that performs global reassignments when
     *                             the number of clusters is in the range
     *                             [maxAllowed, ceil(1.2*maxAllowed)]
     * @param overlapParameter     a control for merging clusters
     * @param distance             a distance function for the points, that
     *                             determines the order of the reverse delete
     *                             however the EM like step uses L1 measure (to be
     *                             robust to noise)
     * @param clusterInitializer   a function that creates the cluster type given a
     *                             single object and a weight
     * @param seed                 a random seed for controlling the randomness
     * @param parallelEnabled      flag enabling (limited) parallelism
     * @param previousClustering   any previous clustering that can be used as zero
     *                             weight seeds to ensure smoothness
     * @return a clustering of the input points (Note: the median returned is an
     *         approximate median; exact computation is unlikely to be critical for
     *         true applications of summarization)
     */
    public static <R> List<ICluster<R>> summarize(List<Weighted<R>> points, int maxAllowed, int initial, int stopAt,
            boolean phase2GlobalReassign, double overlapParameter, BiFunction<R, R, Double> distance,
            BiFunction<R, Float, ICluster<R>> clusterInitializer, long seed, boolean parallelEnabled,
            List<ICluster<R>> previousClustering) {
        checkArgument(maxAllowed < 100, "are you sure you want more elements in the summary?");
        checkArgument(maxAllowed <= initial, "initial parameter should be at least maximum allowed in final result");
        checkArgument(stopAt > 0 && stopAt <= maxAllowed, "lower bound set incorrectly");

        double totalWeight = points.stream().map(e -> (double) e.weight).reduce(0.0, Double::sum);
        checkArgument(!Double.isNaN(totalWeight) && Double.isFinite(totalWeight),
                " weights have to finite and non-NaN");
        Random rng = new Random(seed);
        // the following list is explicity copied and sorted for potential efficiency
        List<Weighted<R>> sampledPoints = createSample(points, rng.nextLong(), 5 * LENGTH_BOUND, 0.005, 1.0);

        ArrayList<Weighted<Integer>> refs = new ArrayList<>();
        for (int i = 0; i < sampledPoints.size(); i++) {
            refs.add(new Weighted<>(i, sampledPoints.get(i).weight));
        }

        Function<Integer, R> getPoint = (i) -> sampledPoints.get(i).index;
        checkArgument(sampledPoints.size() > 0, "empty list, nothing to do");
        double sampledSum = sampledPoints.stream().map(e -> (double) e.weight).reduce(Double::sum).get();

        return iterativeClustering(maxAllowed, initial, stopAt, refs, getPoint, distance, clusterInitializer,
                rng.nextLong(), parallelEnabled, phase2GlobalReassign, overlapParameter, previousClustering);
    }

    // same as above, specific for single centroid clustering of float[]
    // with an explicit stopping condition as well as a global reassignment option
    public static List<ICluster<float[]>> singleCentroidSummarize(List<Weighted<float[]>> points, int maxAllowed,
            int initial, int stopAt, boolean phase2GlobalReassign, BiFunction<float[], float[], Double> distance,
            long seed, boolean parallelEnabled, List<ICluster<float[]>> previousClustering) {
        return summarize(points, maxAllowed, initial, stopAt, phase2GlobalReassign, DEFAULT_SEPARATION_RATIO_FOR_MERGE,
                distance, Center::initialize, seed, parallelEnabled, previousClustering);
    }

    /**
     * the following function returns a summary of the input points
     *
     * @param points          points with associated weights
     * @param maxAllowed      the maximum number of clusters/summary points
     * @param initial         the initial number of clusters/summary points, chosen
     *                        at random
     * @param phase1reassign  (this parameter is ignored in the current version, but
     *                        the signature is unchanged for convenience)
     * @param distance        a distance function for the points, that determines
     *                        the order of the reverse delete however the EM like
     *                        step uses L1 measure (to be robust to noise)
     * @param seed            a random seed for controlling the randomness
     * @param parallelEnabled flag enabling (limited) parallelism
     * @return a summary of the input points (Note: the median returned is an
     *         approximate median; exact computation is unlikely to be critical for
     *         true applications of summarization)
     */
    public static SampleSummary summarize(List<Weighted<float[]>> points, int maxAllowed, int initial,
            boolean phase1reassign, BiFunction<float[], float[], Double> distance, long seed, boolean parallelEnabled) {
        checkArgument(maxAllowed < 100, "are you sure you want more elements in the summary?");
        checkArgument(maxAllowed <= initial, "initial parameter should be at least maximum allowed in final result");

        double totalWeight = points.stream().map(e -> (double) e.weight).reduce(0.0, Double::sum);
        checkArgument(!Double.isNaN(totalWeight) && Double.isFinite(totalWeight),
                " weights have to finite and non-NaN");
        Random rng = new Random(seed);
        // the following list is explicity copied and sorted for potential efficiency
        List<Weighted<float[]>> sampledPoints = createSample(points, rng.nextLong(), 5 * LENGTH_BOUND, 0.005, 1.0);
        List<ICluster<float[]>> centers = summarize(sampledPoints, maxAllowed, initial, 1, true,
                DEFAULT_SEPARATION_RATIO_FOR_MERGE, distance, Center::initialize, seed, parallelEnabled, null);

        float[][] pointList = new float[centers.size()][];
        float[] likelihood = new float[centers.size()];

        int dimensions = centers.get(0).primaryRepresentative(distance).length;
        for (int i = 0; i < centers.size(); i++) {
            pointList[i] = Arrays.copyOf(centers.get(i).primaryRepresentative(distance), dimensions);
            likelihood[i] = (float) (centers.get(i).getWeight() / totalWeight);
        }

        return new SampleSummary(sampledPoints, pointList, likelihood);
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
     * @param reassignPerStep unusued in current version
     * @param distance        distance metric over float []
     * @param seed            random seed
     * @param parallelEnabled flag enabling (limited) parallelism
     * @return a list of centers with weights
     */
    public static SampleSummary summarize(float[][] points, int maxAllowed, int initial, boolean reassignPerStep,
            BiFunction<float[], float[], Double> distance, long seed, Boolean parallelEnabled) {
        ArrayList<Weighted<float[]>> weighted = new ArrayList<>();
        for (float[] point : points) {
            weighted.add(new Weighted<>(point, 1.0f));
        }
        return summarize(weighted, maxAllowed, initial, reassignPerStep, distance, seed, parallelEnabled);
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
    public static SampleSummary summarize(List<Weighted<float[]>> points, int maxAllowed, int initial,
            boolean reassignPerStep) {
        return summarize(points, maxAllowed, initial, reassignPerStep, Summarizer::L2distance, new Random().nextLong(),
                false);
    }

    /**
     * Same as above, with the most common use cases filled in
     *
     * @param points     points in float[][], each of weight 1.0
     * @param maxAllowed maximum number of clusters one is interested in
     * @return a summarization
     */
    public static SampleSummary summarize(float[][] points, int maxAllowed) {
        return summarize(points, maxAllowed, 4 * maxAllowed, false, Summarizer::L2distance, new Random().nextLong(),
                false);
    }

    /**
     *
     * @param points                  points represented by R[]
     * @param maxAllowed              maximum number of clusters in output
     * @param initial                 initial number of points to seed; a control
     *                                parameter that serves both as a denoiser, as
     *                                well as as a facilitator of coninuity (large
     *                                numbers would correspond to MST like
     *                                clustering)
     * @param phase2GlobalReassign    a boolean determining global reassignment
     * @param overlapParameter        a parameter controlling order of mergers
     * @param distance                distance metric over float []
     * @param seed                    random seed
     * @param parallelEnabled         flag enabling (limited) parallelism
     * @param shrinkage               a parameter that morphs from centroidal
     *                                behavior (=1) to robust Minimum Spanning Tree
     *                                (=0)
     * @param numberOfRepresentatives the number of representatives ina multicentrod
     *                                representation used to cluster potentially
     *                                non-spherical shapes
     * @return a list of centers with weights
     */
    public static <R> List<ICluster<R>> multiSummarize(List<R> points, int maxAllowed, int initial, int stopAt,
            boolean phase2GlobalReassign, double overlapParameter, BiFunction<R, R, Double> distance, long seed,
            Boolean parallelEnabled, double shrinkage, int numberOfRepresentatives) {

        ArrayList<Weighted<R>> weighted = new ArrayList<>();
        for (R point : points) {
            weighted.add(new Weighted<>(point, 1.0f));
        }
        BiFunction<R, Float, ICluster<R>> clusterInitializer = (a, b) -> GenericMultiCenter.initialize(a, b, shrinkage,
                numberOfRepresentatives);
        return summarize(weighted, maxAllowed, initial, stopAt, phase2GlobalReassign, overlapParameter, distance,
                clusterInitializer, seed, parallelEnabled, null);
    }

    // same as above, different input
    public static <R> List<ICluster<R>> multiSummarize(R[] points, int maxAllowed, int initial, int stopAt,
            boolean phase2GlobalReassign, double overlapParameter, BiFunction<R, R, Double> distance, long seed,
            Boolean parallelEnabled, double shrinkage, int numberOfRepresentatives) {

        ArrayList<Weighted<R>> weighted = new ArrayList<>();
        for (R point : points) {
            weighted.add(new Weighted<>(point, 1.0f));
        }
        BiFunction<R, Float, ICluster<R>> clusterInitializer = (a, b) -> GenericMultiCenter.initialize(a, b, shrinkage,
                numberOfRepresentatives);
        return summarize(weighted, maxAllowed, initial, stopAt, phase2GlobalReassign, overlapParameter, distance,
                clusterInitializer, seed, parallelEnabled, null);
    }

    // same as above, with defaults
    public static List<ICluster<float[]>> multiSummarize(float[][] points, int maxAllowed, double shrinkage,
            int numberOfRepresentatives) {

        ArrayList<Weighted<float[]>> weighted = new ArrayList<>();
        for (float[] point : points) {
            weighted.add(new Weighted<>(point, 1.0f));
        }
        BiFunction<float[], Float, ICluster<float[]>> clusterInitializer = (a, b) -> MultiCenter.initialize(a, b,
                shrinkage, numberOfRepresentatives);
        return summarize(weighted, maxAllowed, 4 * maxAllowed, 1, true, DEFAULT_SEPARATION_RATIO_FOR_MERGE,
                Summarizer::L2distance, clusterInitializer, new Random().nextLong(), true, null);
    }

}
