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

package com.amazon.randomcutforest.extendedrandomcutforest.threshold;

import static com.amazon.randomcutforest.CommonUtils.checkArgument;

import java.util.Arrays;

import com.amazon.randomcutforest.RandomCutForest;
import com.amazon.randomcutforest.extendedrandomcutforest.AnomalyDescriptor;
import com.amazon.randomcutforest.returntypes.DiVector;

public class ThresholdedRandomCutForest{

    public static int MINIMUM_OBSERVATIONS_FOR_EXPECTED = 100;

    // a parameter that determines if the current potential anomaly is describing the same anomaly
    // within the same shingle or across different time points
    protected double ignoreSimilarFactor = 0.3;
    // uses attribution by default; can be useful without attribution as a general featurization
    protected double triggerFactor = 3.0;

    // saved attribution of the last seen anomaly
    protected long lastAnomalyTimeStamp;
    protected double lastAnomalyScore;
    protected DiVector lastAnomalyAttribution;
    protected double lastScore;
    double[] lastAnomalyPoint;
    double[] lastExpectedPoint;

    boolean previousIsPotentialAnomaly;
    boolean inHighScoreRegion;

    // flag that determines if we should dedup similar anomalies not in the same shingle, for example an
    // anomaly, with the same pattern is repeated across more than a shingle
    protected boolean ignoreSimilar;

    // for anomaly description we would only look at these may top attributors
    // note that expected value is not well defined when this number is greater than 1
    int numberOfAttributors = 2;

    protected RandomCutForest forest;
    protected BasicThresholder thresholder;



    public ThresholdedRandomCutForest(RandomCutForest.Builder builder, double anomalyRate){
        forest = builder.build();
        checkArgument(!forest.isInternalShinglingEnabled(),"Incorrect setting, not supported");
        thresholder = new BasicThresholder(anomalyRate);
        if (forest.getDimensions()/forest.getShingleSize() == 1){
            thresholder.setLowerThreshold(1.1);
        }
    }

    public ThresholdedRandomCutForest(RandomCutForest forest, BasicThresholder thresholder, ThresholdedRandomCutForestState state){
        this.forest = forest;
        this.thresholder = thresholder;
        this.inHighScoreRegion = state.isInAnomaly();
        this.lastAnomalyPoint = (state.getLastAnomalyPoint() == null)?null:Arrays.copyOf(state.getLastAnomalyPoint(),state.getLastAnomalyPoint().length);
        this.lastExpectedPoint = (state.getLastExpectedPoint() == null)?null:Arrays.copyOf(state.getLastExpectedPoint(),state.getLastExpectedPoint().length);
        this.lastAnomalyAttribution = (state.getLastAnomalyAttribution() == null)?null:
                new DiVector(state.getLastAnomalyAttribution());
        this.lastAnomalyTimeStamp = state.getLastAnomalyTimeStamp();
        this.lastAnomalyScore = state.getLastAnomalyScore();
        this.lastScore = state.getLastScore();
        this.ignoreSimilar = state.isIgnoreSimilar();
        this.ignoreSimilarFactor = state.getIgnoreSimilarFactor();
        this.previousIsPotentialAnomaly = state.isPreviousIsPotentialAnomaly();
        this.triggerFactor = state.getTriggerFactor();
        this.numberOfAttributors = state.getNumberOfAttributors();
    }

    public AnomalyDescriptor process(double[] point) {
       AnomalyDescriptor result = getAnomalyDescription(point);
       forest.update(point);
       return result;
    }


    public RandomCutForest getForest() {
        return forest;
    }

    public IThresholder getThresholder() {
        return thresholder;
    }


    protected boolean useLastScore() {
        return lastScore > 0 && !previousIsPotentialAnomaly;
    }

    protected void update(double score, double secondScore, boolean flag) {
        if (useLastScore()) {
            thresholder.update(score, secondScore - lastScore);
        }
        lastScore = score;
        previousIsPotentialAnomaly = flag;
    }

    private int maxContribution(DiVector diVector, int baseDimension, int startIndex) {
        double val = 0;
        int index = startIndex;
        int position = diVector.getDimensions() + startIndex * baseDimension;
        for (int i = 0; i < baseDimension; i++) {
            val += diVector.getHighLowSum(i + position);
        }
        for (int i = position + baseDimension; i < diVector.getDimensions(); i += baseDimension) {
            double sum = 0;
            for (int j = 0; j < baseDimension; j++) {
                sum += diVector.getHighLowSum(i + j);
            }
            if (sum > val) {
                val = sum;
                index = (i - diVector.getDimensions()) / baseDimension;
            }
        }
        return index;
    }

    private int[] largestFeatures(DiVector diVector, int position, int baseDimension, int max_number) {
        if (baseDimension == 1) {
            return new int[]{position};
        }
        double sum = 0;
        double[] values = new double[baseDimension];
        for (int i = 0; i < baseDimension; i++) {
            sum += values[i] = diVector.getHighLowSum(i + position);
        }
        Arrays.sort(values);
        double cutoff = values[baseDimension - Math.min(max_number, baseDimension)];
        int[] answer = new int[Math.min(max_number, baseDimension)];
        int count = 0;
        for (int i = 0; i < baseDimension; i++) {
            if (diVector.getHighLowSum(i + position) >= cutoff &&
                    diVector.getHighLowSum(i + position) > sum * 0.1) {
                answer[count++] = position + i;
            }
        }
        return Arrays.copyOf(answer, count);
    }

    protected boolean trigger(DiVector candidate, int gap, int baseDimension, DiVector ideal) {
        if (lastAnomalyAttribution == null) {
            return true;
        }
        checkArgument(lastAnomalyAttribution.getDimensions() == candidate.getDimensions(), " error in DiVectors");
        int dimensions = candidate.getDimensions();

        int difference = baseDimension * gap;

        if (difference < dimensions) {
            if (ideal == null) {
                double remainder = 0;
                for (int i = dimensions - difference; i < dimensions; i++) {
                    remainder += candidate.getHighLowSum(i);
                }
                return thresholder.getAnomalyGrade(remainder * dimensions / difference, triggerFactor) > 0;
            } else {
                double differentialRemainder = 0;
                for (int i = dimensions - difference; i < dimensions; i++) {
                    differentialRemainder += Math.abs(candidate.low[i] - ideal.low[i]) + Math.abs(candidate.high[i] - ideal.high[i]);
                }
                return (differentialRemainder > ignoreSimilarFactor * lastAnomalyScore) &&
                        thresholder.getAnomalyGrade(differentialRemainder * dimensions / difference, triggerFactor) > 0;
            }
        } else {
            if (!ignoreSimilar) {
                return true;
            }
            double sum = 0;
            for (int i = 0; i < dimensions; i++) {
                sum += Math.abs(lastAnomalyAttribution.high[i] - candidate.high[i]) +
                        Math.abs(lastAnomalyAttribution.low[i] - candidate.low[i]);
            }
            return (sum > ignoreSimilarFactor * lastScore);
        }
    }

    protected AnomalyDescriptor getAnomalyDescription(double[] point) {
        AnomalyDescriptor result = new AnomalyDescriptor();
        result.rcfScore = forest.getAnomalyScore(point);
        result.timeStamp = forest.getTotalUpdates();
        result.forestSize = forest.getNumberOfTrees();
        result.attribution = forest.getAnomalyAttribution(point);
        int shingleSize = forest.getShingleSize();
        int baseDimensions = forest.getDimensions() / shingleSize;
        result.currentValues = new double[baseDimensions];
        int startPosition = (shingleSize - 1) * baseDimensions;

        // the forecast may not be reasonable with less data
        boolean reasonableForecast = (result.timeStamp > MINIMUM_OBSERVATIONS_FOR_EXPECTED) && (shingleSize * baseDimensions >= 4);

        for (int i = 0; i < baseDimensions; i++) {
            result.currentValues[i] = point[startPosition + i];
        }

        if (result.timeStamp == 864){
            System.out.println("HA");
        }
        /**
         * We first check if the score is high enough to be considered as a candidate
         * anomaly. If not, which is hopefully 99% of the data, the computation is short
         */
        if (thresholder.getAnomalyGrade(result.rcfScore) == 0) {
            result.anomalyGrade = 0;
            inHighScoreRegion = false;
            update(result.rcfScore, result.rcfScore,false);
            return result;
        }

        // the score is now high enough to be considered an anomaly
        result.inHighScoreRegion = true;

        /**
         * We now check if (1) we have another anomaly in the current shingle (2) have predictions
         * about what the values should have been and (3) replacing by those "should have been" makes the
         * anomaly score of the new shingled point low enough to not be an anomaly. In this case we can
         * "explain" the high score is due to the past and do not need to vend anomaly -- because the most
         * recent point, on their own would not produce an anomalous shingle.
         *
         * However the strategy is only executable if there are (A) sufficiently many observations and (B) enough
         * data in each time point such that the forecast is reasonable. While forecasts can be corrected for very low
         * shingle sizes and say 1d input, the allure of RCF is in the multivariate case. Even for 1d, a shiglesize of
         * 4 or larger would produce reasonable forecast for the purposes of anomaly detection.
         */

        int gap = (int) (result.timeStamp - lastAnomalyTimeStamp);

        if (reasonableForecast && lastAnomalyPoint != null && lastExpectedPoint != null && gap < shingleSize) {
            double[] correctedPoint = Arrays.copyOf(point, point.length);
            for (int i = 0; i < point.length - gap * baseDimensions; i++) {
                correctedPoint[i] = lastExpectedPoint[i + gap * baseDimensions];
            }
            double correctedScore = forest.getAnomalyScore(correctedPoint);
            if (thresholder.getAnomalyGrade(correctedScore) == 0) {
                // fixing the past makes this anomaly go away; nothing to do but process the score
                // we will not change inAnomaly however, because the score has been larger
                update(result.rcfScore, correctedScore,false);
                result.anomalyGrade = 0;
                return result;
            }
        }

        /**
         * We now check the most egregeous values seen in the current timestamp, as determined by attribution.
         * Those locations provide information about (a) which attributes and (b) what the values should have
         * been. However those calculations of imputation only make sense when sufficient observations are available.
         */

        double[] newPoint = null;
        double newScore = 0;
        DiVector newAttribution = null;
        if (reasonableForecast) {
            int[] likelyMissingIndices = largestFeatures(result.attribution, startPosition, baseDimensions, numberOfAttributors);
            newPoint = forest.imputeMissingValues(point, likelyMissingIndices.length, likelyMissingIndices);
            newAttribution = forest.getAnomalyAttribution(newPoint);
            newScore = forest.getAnomalyScore(newPoint);
        }

        /**
         * we now find the time slice, relative to the current time, which is indicative of the high
         * score. relativeIndex = 0 is current time. It is negative if the most egregeous attribution
         * was due to the past values in the shingle
         */

        result.relativeIndex = maxContribution(result.attribution, baseDimensions, -shingleSize) + 1;

        /**
         * if we are transitioning from low score to high score range (given by inAnomaly) then
         * we check if the new part of the input could have triggered anomaly on its own
         * That decision is vended by trigger() which extrapolates a partial shingle
         */
        if (!inHighScoreRegion && trigger(result.attribution, gap, baseDimensions, null)) {
            result.anomalyGrade = thresholder.getAnomalyGrade(result.rcfScore);
            lastAnomalyScore = newScore;
            inHighScoreRegion = true;
            result.startOfAnomaly = true;
            lastAnomalyAttribution = new DiVector(result.attribution);
            lastAnomalyTimeStamp = result.timeStamp;
            lastAnomalyPoint = Arrays.copyOf(point,point.length);
            update(result.rcfScore, result.rcfScore, true);
        } else {
            /**
             * we again check if the new input produces an anomaly/not on its own
             */
            if (trigger(result.attribution, gap, baseDimensions, newAttribution) && result.rcfScore > newScore) {
                result.anomalyGrade = thresholder.getAnomalyGrade(result.rcfScore);
                lastAnomalyScore = result.rcfScore;
                lastAnomalyAttribution = new DiVector(result.attribution);
                lastAnomalyTimeStamp = result.timeStamp;
                lastAnomalyPoint = Arrays.copyOf(point,point.length);
                update(result.rcfScore, result.rcfScore, true);
            } else {
                // not changing inAnomaly
                result.anomalyGrade = 0;
                update(result.rcfScore, result.rcfScore,false);
            }
        }

        if (result.anomalyGrade>0){
            result.expectedValuesPresent = reasonableForecast;
            if (result.relativeIndex < 0 && result.startOfAnomaly) {
                // anomaly in the past and detected late; repositioning the computation
                startPosition = result.attribution.getDimensions() + (result.relativeIndex - 1) * baseDimensions;
                if (result.expectedValuesPresent) {
                    int[] missingIndices = largestFeatures(result.attribution, startPosition, baseDimensions, numberOfAttributors);
                    newPoint = forest.imputeMissingValues(point, missingIndices.length, missingIndices);
                    result.oldValues = new double[baseDimensions];
                    for (int i = 0; i < baseDimensions; i++) {
                        result.oldValues[i] = point[startPosition + i];
                    }
                }
            }
            if (result.expectedValuesPresent) {
                result.expectedValuesList = new double[1][];
                result.expectedValuesList[0] = new double[baseDimensions];
                for (int i = 0; i < baseDimensions; i++) {
                    result.expectedValuesList[0][i] = newPoint[startPosition + i];
                }
                result.likelihoodOfValues = new double[]{1.0};
                lastExpectedPoint = Arrays.copyOf(newPoint,newPoint.length);
            } else {
                lastExpectedPoint = null;
            }

            result.flattenedAttribution = new double[baseDimensions];
            for (int i = 0; i < baseDimensions; i++) {
                result.flattenedAttribution[i] = result.attribution.getHighLowSum(startPosition + i);
            }
        }
        return result;
    }

    public boolean isPreviousIsPotentialAnomaly() {
        return previousIsPotentialAnomaly;
    }

    public boolean isIgnoreSimilar() {
        return ignoreSimilar;
    }

    public DiVector getLastAnomalyAttribution() {
        return lastAnomalyAttribution;
    }


    public double getIgnoreSimilarFactor() {
        return ignoreSimilarFactor;
    }

    public double getLastScore() {
        return lastScore;
    }

    public double[] getLastAnomalyPoint() {
        return (lastAnomalyPoint == null)? null:Arrays.copyOf(lastAnomalyPoint,lastAnomalyPoint.length);
    }

    public double[] getLastExpectedPoint() {
        return (lastExpectedPoint == null)? null:Arrays.copyOf(lastExpectedPoint,lastExpectedPoint.length);
    }

    public boolean isInHighScoreRegion() {
        return inHighScoreRegion;
    }

    public double getTriggerFactor() {
        return triggerFactor;
    }

    public long getLastAnomalyTimeStamp() {
        return lastAnomalyTimeStamp;
    }

    public double getLastAnomalyScore() {
        return lastAnomalyScore;
    }

    public void setIgnoreSimilarFactor(double factor){
        ignoreSimilarFactor = factor;
    }

    public int getNumberOfAttributors() {
        return numberOfAttributors;
    }

    public void setLowerThreshold(double threshold){
        thresholder.setLowerThreshold(threshold);
    }

}
