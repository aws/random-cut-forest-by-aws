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

import com.amazon.randomcutforest.returntypes.SampleSummary;
import com.amazon.randomcutforest.util.Weighted;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.function.BiFunction;
import java.util.function.Function;

import static com.amazon.randomcutforest.CommonUtils.checkArgument;
import static com.amazon.randomcutforest.util.Weighted.createSample;
import static com.amazon.randomcutforest.util.Weighted.prefixPick;
import static java.lang.Math.max;

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
     * @param clusters        an array list of current clusters, because random
     *                        access to the elements is necessary
     * @param distance        a distance function
     * @param parallelEnabled a flag enabling limited parallelism; only during
     *                        cluster by cluster recomputation. Using parallel mode
     *                        during the assignment of points does not seem to help
     */
    public static <R> void assignAndRecompute(List<Weighted<Integer>> sampledPoints, Function<Integer,R> getPoint, ArrayList<ICluster<R>> clusters,
            BiFunction<R, R, Double> distance, boolean parallelEnabled) {
        checkArgument(clusters.size() > 0, " cannot be empty list of clusters");
        checkArgument(sampledPoints.size() > 0, " cannot be empty list of points");

        for (ICluster cluster : clusters) {
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
                    clusters.get(minDistNbr).addPoint(point.index, point.weight, 0,getPoint.apply(point.index),distance);
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
            clusters.parallelStream().forEach(e -> e.recompute(getPoint, distance));
        } else {
            clusters.stream().forEach(e -> e.recompute(getPoint, distance));
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
     * @param maxAllowed         maximum number of clusters
     * @param initial            the initial number of "seeded" clusters, this
     *                           parameter controls both denoising (lower is better)
     *                           as well as ensuring clusters remain connected
     *                           (higher is better)
     * @param sampledPoints      the list of sampled points to consider; again this
     *                           sample performs denoising (lower is better) as well
     *                           as provides representation (higher is better in
     *                           information content, but slower)
     * @param distance           the distance function desired
     * @param clusterInitializer a factory method that initializes a cluster from a
     *                           single weighted point
     * @param seed               random seed for choosing initial centers
     * @param parallelEnabled    a flag controlling (limited) parallelism
     * @param phase1reassign     should the centers be optimized during phase 1?
     *                           (default suggestion is false)
     * @param phase2reassign same for phase 2, when the number of clusters is close to maxallowed
     *                           (default suggestion is true)
     * @param enablePhase3       should phase 3 take place?
     * @param overlapParameter   a parameter that controls the ordering of cluster
     *                           merges
     * @param previousClustering a list of previous clusters which can be used to seed the clusters (with 0 weight;
     *                           only the most recent data in SampledPoints have non-zero weight). This may be useful for
     *                           maintaining smoothness in dynamic clustering environments.
     * @param <Q>                type of the cluster, so that this same framework
     *                           can be used for multiple clustering algorithms that
     *                           correspond to different implementations
     * @return a list of cluster centers
     */

    public static <R,Q extends ICluster<R>> List<ICluster<R>>
    iterativeClustering(int maxAllowed, int initial, List<Weighted<Integer>> sampledPoints, Function<Integer,R> getPoint, BiFunction<R, R, Double> distance,
                        BiFunction<R, Float, Q> clusterInitializer, long seed, boolean parallelEnabled,
                        boolean phase1reassign, boolean phase2reassign, boolean enablePhase3, double overlapParameter, List<ICluster<R>> previousClustering) {

        checkArgument(sampledPoints.size() > 0, "empty list, nothing to do");
        double sampledSum = sampledPoints.stream().map(e -> (double) e.weight).reduce(Double::sum).get();
        Random rng = new Random(seed);
        ArrayList<ICluster<R>> centers = new ArrayList<>();
        if (sampledPoints.size() < 10 * initial) {
            for (Weighted<Integer> point : sampledPoints) {
                centers.add(clusterInitializer.apply(getPoint.apply(point.index), point.weight));
            }
        } else {
            for (int k = 0; k < 2 * initial; k++) {
                double wt = rng.nextDouble() * sampledSum;
                Weighted<Integer> picked = prefixPick(sampledPoints, wt);
                centers.add(clusterInitializer.apply(getPoint.apply(picked.index), picked.weight));
            }
        }
        if (previousClustering != null){
            for(ICluster<R> previousCluster : previousClustering) {
                List<Weighted<R>> representatives = previousCluster.getRepresentatives();
                for( Weighted<R> representative: representatives){
                    centers.add(clusterInitializer.apply(representative.index,0f));
                }
            }
        }

        assignAndRecompute(sampledPoints, getPoint, centers, distance, parallelEnabled);

        // assignment would change weights, sorting in non-decreasing order
        centers.sort(Comparator.comparingDouble(ICluster::getWeight));
        while (centers.get(0).getWeight() == 0) {
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
                        double temp = (centers.get(lower).averageRadius() + centers.get(j).averageRadius()
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
            if (inital > maxAllowed || foundMerge || (enablePhase3 && measure > overlapParameter)) {
                centers.get(secondOfMerge).absorb(centers.get(firstOfMerge), distance);
                if (phase1reassign || phase2reassign && centers.size() <= PHASE2_THRESHOLD * maxAllowed) {
                    centers.remove(firstOfMerge);
                    assignAndRecompute(sampledPoints, getPoint, centers, distance, parallelEnabled);
                } else {
                    centers.get(secondOfMerge).recompute(getPoint,distance);
                    centers.remove(firstOfMerge);
                }
                centers.sort(Comparator.comparingDouble(ICluster::getWeight));
                while (centers.get(0).getWeight() == 0.0) {
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
        return centers;
    }

    /**
     * the following function returns a summary of the input points
     * 
     * @param points          points with associated weights
     * @param maxAllowed      the maximum number of clusters/summary points
     * @param initial         the initial number of clusters/summary points, chosen
     *                        at random
     * @param phase1reassign  a boolean which controls if an EM like step is
     *                        performed every step; by default, such a step is not
     *                        performed until the number of clusters fall below
     *                        PHASE_THRESHOLD times the maximum number of clusters
     *                        allowed
     * @param phase2Reassign same, for phase 2, when the number of clusters are close to
     *                       the maximum allowed
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
            boolean phase1reassign, boolean phase2Reassign, BiFunction<float[], float[], Double> distance, long seed, boolean parallelEnabled, List<ICluster<float[]>> previousClustering) {
        checkArgument(maxAllowed < 100, "are you sure you want more elements in the summary?");
        checkArgument(maxAllowed <= initial, "initial parameter should be at least maximum allowed in final result");

        double totalWeight = points.stream().map(e -> (double) e.weight).reduce(0.0, Double::sum);
        checkArgument(!Double.isNaN(totalWeight) && Double.isFinite(totalWeight),
                " weights have to finite and non-NaN");
        Random rng = new Random(seed);
        List<Weighted<float[]>> sampledPoints = createSample(points, rng.nextLong(), 5 * LENGTH_BOUND, 0.005, 1.0);

        ArrayList<Weighted<Integer>> refs = new ArrayList<>();
        for(int i=0;i<sampledPoints.size();i++){
            refs.add(new Weighted<>(i,sampledPoints.get(i).weight));
        }

        Function<Integer,float[]> getPoint = (i) -> sampledPoints.get(i).index;
        List<ICluster<float[]>> centers = iterativeClustering(maxAllowed, initial, refs, getPoint, distance,
                Center::initialize, rng.nextLong(), parallelEnabled, phase1reassign, phase2Reassign, true, DEFAULT_SEPARATION_RATIO_FOR_MERGE,previousClustering);

        // sort in decreasing weight
        centers.sort((o1, o2) -> Double.compare(o2.getWeight(), o1.getWeight()));
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
     * @param reassignPerStep a boolean deciding if reassignment is performed each
     *                        step, corresponding to phase 1. Phase 2 default is true.
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
        return summarize(weighted, maxAllowed, initial, reassignPerStep, true, distance, seed, parallelEnabled,null);
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
        return summarize(points, maxAllowed, initial, reassignPerStep, false, Summarizer::L2distance, new Random().nextLong(),
                false,null);
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
     * the following function returns a summary of the input points
     *
     * @param points          points with associated weights
     * @param maxAllowed      the maximum number of clusters/summary points
     * @param initial         the initial number of clusters/summary points, chosen
     *                        at random
     * @param phase1reassign  a boolean which controls if an EM like step is
     *                        performed every step; by default, such a step is not
     *                        performed until the number of clusters fall below
     *                        PHASE_THRESHOLD times the maximum number of clusters
     *                        allowed which defines phase 2
     * @param phase2reassign same as above, till the nunber of clusters fall to maximum allowed
     * @param distance        a distance function for the points, that determines
     *                        the order of the reverse delete however the EM like
     *                        step uses L1 measure (to be robust to noise)
     * @param seed            a random seed for controlling the randomness
     * @param parallelEnabled flag enabling (limited) parallelism
     * @return a summary of the input points (Note: the median returned is an
     *         approximate median; exact computation is unlikely to be critical for
     *         true applications of summarization)
     */
    public static <R>  List<ICluster<R>> multiSummarize(List<Weighted<R>> points, int maxAllowed, int initial,
                                                       boolean phase1reassign, boolean phase2reassign, BiFunction<R, R, Double> distance, long seed, boolean parallelEnabled,
                                                       double shrinkage, int numberOfRepresentatives, List<ICluster<R>> previousClustering) {
        checkArgument(maxAllowed < 100, "are you sure you want more elements in the summary?");
        checkArgument(maxAllowed <= initial, "initial parameter should be at least maximum allowed in final result");
        checkArgument(shrinkage>=0 && shrinkage <= 1.0, " parameter has to be in [0,1]");
        checkArgument(numberOfRepresentatives>0 && numberOfRepresentatives <= 100, " the number of representatives has to be in (0,100]");

        double totalWeight = points.stream().map(e -> (double) e.weight).reduce(0.0, Double::sum);
        checkArgument(!Double.isNaN(totalWeight) && Double.isFinite(totalWeight),
                " weights have to finite and non-NaN");
        Random rng = new Random(seed);
        List<Weighted<R>> sampledPoints = createSample(points, rng.nextLong(), 5 * LENGTH_BOUND, 0.005, 1.0);

        ArrayList<Weighted<Integer>> refs = new ArrayList<>();
        for(int i=0;i<sampledPoints.size();i++){
            refs.add(new Weighted<>(i,sampledPoints.get(i).weight));
        }

        Function<Integer,R> getPoint = (i) -> sampledPoints.get(i).index;
        BiFunction<R, Float, ICluster<R>> clusterInitializer = (a,b) -> MultiCenter.initialize(a,b,shrinkage,numberOfRepresentatives);
        List<ICluster<R>> centers = iterativeClustering(maxAllowed, initial, refs,getPoint, distance,
                clusterInitializer, rng.nextLong(), parallelEnabled, phase1reassign, phase2reassign,true, DEFAULT_SEPARATION_RATIO_FOR_MERGE,previousClustering);

        // sort in decreasing weight
        centers.sort((o1, o2) -> Double.compare(o2.getWeight(), o1.getWeight()));
        return centers;
    }

    /**
     *
     * @param points          points represented by R[]
     * @param maxAllowed      maximum number of clusters in output
     * @param initial         initial number of points to seed; a control parameter
     *                        that serves both as a denoiser, as well as as a
     *                        facilitator of coninuity (large numbers would
     *                        correspond to MST like clustering)
     * @param phase1Reassign a boolean deciding if reassignment is performed each
     *                        step in phase 1 (default false)
     * @param phase2reassign same, for phase 2 (default true)
     * @param distance        distance metric over float []
     * @param seed            random seed
     * @param parallelEnabled flag enabling (limited) parallelism
     * @param shrinkage a parameter that morphs from centroidal behavior (=1) to robust Minimum Spanning Tree (=0)
     * @param numberOfRepresentatives the number of representatives ina multicentrod representation used to cluster
     *                                potentially non-spherical shapes
     * @return a list of centers with weights
     */
    public static <R> List<ICluster<R>> multiSummarize(R[] points, int maxAllowed, int initial, boolean phase1Reassign, boolean phase2reassign,
                                                       BiFunction<R, R, Double> distance, long seed, Boolean parallelEnabled, double shrinkage, int numberOfRepresentatives) {

        ArrayList<Weighted<R>> weighted = new ArrayList<>();
        for (R point : points) {
            weighted.add(new Weighted<>(point, 1.0f));
        }
        return multiSummarize(weighted, maxAllowed, initial, phase1Reassign,phase2reassign, distance, seed, parallelEnabled,shrinkage,numberOfRepresentatives,null);
    }

    // same as above, without worrying about any previous clusters
    public static <R> List<ICluster<R>> multiSummarize(List<R> points, int maxAllowed, int initial, boolean reassignPerStep, boolean phase2reassign,
                                                       BiFunction<R, R, Double> distance, long seed, Boolean parallelEnabled, double shrinkage, int numberOfRepresentatives) {

        ArrayList<Weighted<R>> weighted = new ArrayList<>();
        for (R point : points) {
            weighted.add(new Weighted<>(point, 1.0f));
        }
        return multiSummarize(weighted, maxAllowed, initial, reassignPerStep, phase2reassign, distance, seed, parallelEnabled,shrinkage,numberOfRepresentatives,null);
    }

    /**
     * Same as above, with the most common use cases filled in
     *
     * @param points     points in float[][], each of weight 1.0
     * @param maxAllowed maximum number of clusters one is interested in
     * @param shringkage the parameter determing centroidal behavior (=1) or MST (=0)
     * @param numberOfRepresentatives number of representives for multicentorid cluster
     * @return a summarization based on L2distance
     */
    public static List<ICluster<float[]>> multiSummarize(float[][] points, int maxAllowed, double shringkage, int numberOfRepresentatives) {
        return multiSummarize(points, maxAllowed, 4 * maxAllowed, false, false,Summarizer::L2distance, new Random().nextLong(),
                false,shringkage,numberOfRepresentatives);
    }

}
