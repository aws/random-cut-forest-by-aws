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

package com.amazon.randomcutforest.parkservices;

import static com.amazon.randomcutforest.CommonUtils.checkArgument;
import static com.amazon.randomcutforest.summarization.GenericMultiCenter.DEFAULT_NUMBER_OF_REPRESENTATIVES;
import static com.amazon.randomcutforest.summarization.GenericMultiCenter.DEFAULT_SHRINKAGE;
import static java.lang.Math.abs;
import static java.lang.Math.exp;
import static java.lang.Math.min;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.function.BiFunction;

import com.amazon.randomcutforest.parkservices.returntypes.GenericAnomalyDescriptor;
import com.amazon.randomcutforest.parkservices.threshold.BasicThresholder;
import com.amazon.randomcutforest.summarization.GenericMultiCenter;
import com.amazon.randomcutforest.summarization.ICluster;
import com.amazon.randomcutforest.summarization.Summarizer;
import com.amazon.randomcutforest.util.Weighted;

public class GlobalLocalAnomalyDetector<P> extends StreamSampler<P> {

    // default maximum number of clusters to consider
    public static int DEFAULT_MAX = 10;

    // an upper bound on the score
    public static float FLOAT_MAX = 10;

    // the relative weight of small clusters which should not be used in anomaly
    // detection
    // this controls masking effects
    public static double DEFAULT_IGNORE_SMALL_CLUSTER_REPRESENTATIVE = 0.005;

    // the number of steps we have to wait before reclustering; in principle this
    // can be 1, but that would be
    // neither be meaningful nor efficient; it is set to a default of the capacity/2
    protected int doNotreclusterWithin;

    // a thresholder for flagging anomalies (same thresholder as in TRCF)
    protected final BasicThresholder thresholder;

    // remembering when the last clustering was performed
    protected long lastCluster = 0L;

    // remembers when the mean of the scores were just above a certain threshold
    // acts as a calibration mechanism
    protected double lastMean = 1;

    // actual list of clusters
    List<ICluster<P>> clusters;

    // the number of maximum clusters to be considered; this is configurable and can
    // be chaned dynamically
    protected int maxAllowed;

    // the shrinkage parameter in multi-centroid clustering such as CURE. Shrinkage
    // of 0 provides
    // non-spherical shapes, whereas shrinkage of 1 corresponds to choosing single
    // centroid (not recommended)
    protected double shrinkage;

    // number of representatives used in multi-centroidal clustering
    protected int numberOfRepresentatives;

    // threshold of weight for small clusters so that masking can be averted, can be
    // changed dynamically
    protected double ignoreBelow;

    // the global function used in clustering, can be changed dynamically (but
    // clustering would be controlled
    // automatically due to efficiency reasons)
    protected BiFunction<P, P, Double> globalDistance;

    public static Builder<?> builder() {
        return new Builder<>();
    }

    protected GlobalLocalAnomalyDetector(Builder<?> builder) {
        super(builder);
        thresholder = new BasicThresholder(builder.timeDecay);
        thresholder.setAbsoluteThreshold(1.2);
        doNotreclusterWithin = builder.doNotReclusterWithin.orElse(builder.capacity / 2);
        shrinkage = builder.shrinkage;
        maxAllowed = builder.maxAllowed;
        numberOfRepresentatives = builder.numberOfRepresentatives;
    }

    protected GlobalLocalAnomalyDetector(Builder<?> builder, BiFunction<P, P, Double> distance) {
        this(builder);
        globalDistance = distance;
    }

    public void setGlobalDistance(BiFunction<P, P, Double> dist) {
        globalDistance = dist;
    }

    // sets the zFactor; increasing this number should increase precision (and will
    // likely lower recall)
    // this is the same as in BasicThresholder class
    public void setZfactor(double factor) {
        thresholder.setZfactor(factor);
    }

    public double getZfactor() {
        return thresholder.getZFactor();
    }

    // as in BasicThresholder class, useful in tuning
    public void setLowerThreshold(double lowerThreshold) {
        thresholder.setLowerThreshold(lowerThreshold);
    }

    public double getLowerThreshold() {
        return thresholder.getLowerThreshold();
    }

    public int getDoNotreclusterWithin() {
        return doNotreclusterWithin;
    }

    public void setDoNotreclusterWithin(int value) {
        checkArgument(value > 0, " has to be positive, recommended as 1/2 the capacity");
        doNotreclusterWithin = value;
    }

    public int getNumberOfRepresentatives() {
        return numberOfRepresentatives;
    }

    public void setNumberOfRepresentatives(int reps) {
        checkArgument(reps > 0, " has to be positive");
    }

    public double getShrinkage() {
        return shrinkage;
    }

    public void setShrinkage(double value) {
        checkArgument(value >= 0 && value <= 1, " has to be in [0,1]");
        shrinkage = value;
    }

    public double getIgnoreBelow() {
        return ignoreBelow;
    }

    public void setIgnoreBelow(double value) {
        checkArgument(value >= 0 && value < 0.1, " relative weight has to be in range [0,0.1] ");
        ignoreBelow = value;
    }

    public int getMaxAllowed() {
        return maxAllowed;
    }

    public void setMaxAllowed(int value) {
        checkArgument(value >= 5 && value < 100,
                " too few or too many clusters are not " + "meaningful to this algorithm");
        maxAllowed = value;
    }

    /**
     * The following provides a single invocation for scoring and updating.
     * Semantics of the recency biased sampling (sequentiality in decision making)
     * and efficient automatic reclustering demand that scoring and updating be
     * simultaneous. While scoring is provided as a separate function to let future
     * preditor-corrector methods reuse this code, it is strongly recommneded that
     * only the process() function be invoked.
     * 
     * @param object            current object being considered
     * @param weight            weight of the object (for clustering purposes as
     *                          well as recency biased sampling)
     * @param localDistance     a local distance metric that determines the order in
     *                          which different clusters are considered; can be
     *                          null, in which case the global distance would be
     *                          used
     * @param considerOcclusion consider occlusion by smaller dense clusters, when
     *                          adjacent to larger and more spread out clusters
     * @return a generic descriptor with score, threshold, anomaly grade (anomaly
     *         grade greater than zero is likely anomalous; anomaly grade can be -ve
     *         to allow down stream correction using semi-supervision or other
     *         means) and a list of cluster representatives (sorted by distance)
     *         with corresponding scores (lowest score may not correspond to lowest
     *         distance) which can be used to investigate anomalous points further
     */
    public GenericAnomalyDescriptor<P> process(P object, float weight, BiFunction<P, P, Double> localDistance,
            boolean considerOcclusion) {
        checkArgument(weight >= 0, "weight cannot be negative");
        // recompute clusters first; this enables easier merges and deserialization
        if (sequenceNumber > lastCluster + doNotreclusterWithin) {
            checkArgument(globalDistance != null, "set global distance function");
            double currentMean = thresholder.getPrimaryDeviation().getMean();
            if (abs(currentMean - lastMean) > 0.1 || currentMean > 1.7
                    || sequenceNumber > lastCluster + 20 * doNotreclusterWithin) {
                lastCluster = sequenceNumber;
                lastMean = currentMean;
                clusters = getClusters(maxAllowed, 4 * maxAllowed, 1, numberOfRepresentatives, shrinkage,
                        globalDistance, null);
            }
        }
        List<Weighted<P>> result = score(object, localDistance, considerOcclusion);
        double threshold = thresholder.threshold();
        double grade = 0;
        float score = 0;
        if (result != null) {
            score = result.stream().map(a -> a.weight).reduce(FLOAT_MAX, Float::min);
            if (score < FLOAT_MAX) {
                // an exponential attribution
                double sum = result.stream()
                        .map(a -> (double) ((a.weight == FLOAT_MAX) ? 0 : exp(-a.weight * a.weight)))
                        .reduce(0.0, Double::sum);
                for (Weighted<P> item : result) {
                    item.weight = (item.weight == FLOAT_MAX) ? 0.0f
                            : (float) min(1.0f, (float) exp(-item.weight * item.weight) / sum);
                }
            } else {
                // uniform attribution
                for (Weighted<P> item : result) {
                    item.weight = (float) 1.0 / (result.size());
                }
            }
            grade = thresholder.getAnomalyGrade(score, false);

        }
        // note average score would be 1
        thresholder.update(score, min(score, thresholder.getZFactor()));
        sample(object, weight);

        return new GenericAnomalyDescriptor<>(result, score, threshold, grade);
    }

    /**
     * The following function scores a point -- it considers an ordering of the
     * representatives based on the local distance; and considers occlusion --
     * namely, should an asteroid between moon and the earth be considered to be a
     * part of a cluster around the moon or the earth? The below provides some
     * initial geometric take on the three objects. We deliberately avoid explicit
     * density computation since it would be difficult to define uniform definition
     * of density.
     * 
     * @param current           the object being scored
     * @param localDistance     a distance function that we wish to use for this
     *                          specific score. This can be null, and in that case
     *                          the global distance would be used
     * @param considerOcclusion a boolean that determines if closeby dense clusters
     *                          can occlude membership in further away "less dense
     *                          cluster"
     * @return A list of weighted type where the index is a representative (based on
     *         local distance) and the weight is the score corresponding to that
     *         representative. The scores are sorted from least anomalous to most
     *         anomalous.
     */
    public List<Weighted<P>> score(P current, BiFunction<P, P, Double> localDistance, boolean considerOcclusion) {
        if (clusters == null) {
            return null;
        } else {
            BiFunction<P, P, Double> local = (localDistance != null) ? localDistance : globalDistance;
            double totalWeight = clusters.stream().map(e -> e.getWeight()).reduce(0.0, Double::sum);
            ArrayList<Candidate> candidateList = new ArrayList<>();
            for (ICluster<P> cluster : clusters) {
                double wt = cluster.averageRadius();
                double tempMinimum = Double.MAX_VALUE;
                P closestInCluster = null;
                for (Weighted<P> rep : cluster.getRepresentatives()) {
                    if (rep.weight > ignoreBelow * totalWeight) {
                        double tempDist = local.apply(current, rep.index);
                        if (tempDist < 0) {
                            throw new IllegalArgumentException(" distance cannot be negative ");
                        }
                        if (tempMinimum > tempDist) {
                            tempMinimum = tempDist;
                            closestInCluster = rep.index;
                        }
                    }
                }
                if (closestInCluster != null) {
                    candidateList.add(new Candidate(closestInCluster, wt, tempMinimum));
                }
            }
            candidateList.sort((o1, o2) -> Double.compare(o1.distance, o2.distance));
            checkArgument(candidateList.size() > 0, "empty candidate list, should not happen");
            ArrayList<Weighted<P>> answer = new ArrayList<>();
            if (candidateList.get(0).distance == 0.0) {
                answer.add(new Weighted<P>(candidateList.get(0).representative, 0.0f));
                return answer;
            }
            int index = 0;
            while (index < candidateList.size()) {
                Candidate head = candidateList.get(index);
                double dist = (localDistance == null) ? head.distance
                        : globalDistance.apply(current, head.representative);
                float tempMeasure = (head.averageRadiusOfCluster > 0.0)
                        ? min(FLOAT_MAX, (float) (dist / head.averageRadiusOfCluster))
                        : FLOAT_MAX;
                answer.add(new Weighted<P>(head.representative, tempMeasure));
                if (considerOcclusion) {
                    for (int j = index + 1; j < candidateList.size(); j++) {
                        double occludeDistance = local.apply(head.representative, candidateList.get(j).representative);
                        double candidateDistance = candidateList.get(j).distance;
                        if (occludeDistance < candidateDistance && head.distance > Math
                                .sqrt(head.distance * head.distance + candidateDistance * candidateDistance)) {
                            // delete element
                            candidateList.remove(j);
                        }
                    }
                }
                ++index;
            }
            // we will not resort answer; the scores will be in order of distance
            // we note that score() should be invoked with care and likely postprocessing
            return answer;
        }
    }

    /**
     * a merging routine for the mopdels which would be used in the future for
     * distributed analysis. Note that there is no point of storing sequence indices
     * explicitly in case of a merge.
     * 
     * @param first     the first model
     * @param second    the second model
     * @param builder   the parameters of the new clustering
     * @param recluster a boolean that determines immediate reclustering
     * @param distance  the distance function of the new clustering
     */
    public GlobalLocalAnomalyDetector(GlobalLocalAnomalyDetector first, GlobalLocalAnomalyDetector second,
            Builder<?> builder, boolean recluster, BiFunction<P, P, Double> distance) {
        super(first, second, builder.capacity, builder.timeDecay, builder.randomSeed);
        thresholder = new BasicThresholder(builder.timeDecay);
        thresholder.setAbsoluteThreshold(1.2);
        doNotreclusterWithin = builder.doNotReclusterWithin.orElse(builder.capacity / 2);
        shrinkage = builder.shrinkage;
        maxAllowed = builder.maxAllowed;
        numberOfRepresentatives = builder.numberOfRepresentatives;
        globalDistance = distance;
        if (recluster) {
            lastCluster = sequenceNumber;
            clusters = getClusters(maxAllowed, 4 * maxAllowed, 1, numberOfRepresentatives, shrinkage, globalDistance,
                    null);
        }
    }

    /**
     * an inner class that is useful for the scoring procedure to avoid
     * recomputation of fields.
     */
    class Candidate {
        P representative;
        double averageRadiusOfCluster;
        double distance;

        Candidate(P representative, double averageRadiusOfCluster, double distance) {
            this.representative = representative;
            this.averageRadiusOfCluster = averageRadiusOfCluster;
            this.distance = distance;
        }
    }

    public ArrayList<Weighted<P>> getObjectList() {
        return objectList;
    }

    public List<ICluster<P>> getClusters() {
        return clusters;
    }

    public List<ICluster<P>> getClusters(int maxAllowed, int initial, int stopAt, int representatives, double shrink,
            BiFunction<P, P, Double> distance, List<ICluster<P>> previousClusters) {
        BiFunction<P, Float, ICluster<P>> clusterInitializer = (a, b) -> GenericMultiCenter.initialize(a, b, shrink,
                representatives);
        return Summarizer.summarize(objectList, maxAllowed, initial, stopAt, false, 0.8, distance, clusterInitializer,
                0L, false, previousClusters);
    }

    /**
     * a builder
     */
    public static class Builder<T extends Builder<T>> extends StreamSampler.Builder<T> {
        protected double shrinkage = DEFAULT_SHRINKAGE;
        protected double ignoreBelow = DEFAULT_IGNORE_SMALL_CLUSTER_REPRESENTATIVE;
        protected int numberOfRepresentatives = DEFAULT_NUMBER_OF_REPRESENTATIVES;
        protected Optional<Integer> doNotReclusterWithin = Optional.empty();
        protected int maxAllowed = DEFAULT_MAX;

        // ignores small clusters with population weight below this threshold
        public T ignoreBelow(double ignoreBelow) {
            this.ignoreBelow = ignoreBelow;
            return (T) this;
        }

        // parameters of the multi-representative CURE algorithm
        public T shrinkage(double shrinkage) {
            this.shrinkage = shrinkage;
            return (T) this;
        }

        // a parameter that ensures that clustering is not recomputed too frequently,
        // which can be both inefficient as well as jittery
        public T doNotReclusterWithin(int refresh) {
            this.doNotReclusterWithin = Optional.of(refresh);
            return (T) this;
        }

        // maximum number of clusters to consider
        public T maxAllowed(int maxAllowed) {
            this.maxAllowed = maxAllowed;
            return (T) this;
        }

        // parameters of the multi-representative CURE algorithm
        public T numberOfRepresentatives(int number) {
            this.numberOfRepresentatives = number;
            return (T) this;
        }

        @Override
        public GlobalLocalAnomalyDetector build() {
            return new GlobalLocalAnomalyDetector<>(this);
        }
    }
}
