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

package com.amazon.randomcutforest.anomalydetection;

import com.amazon.randomcutforest.CommonUtils;

/**
 * Attribution exposes the attribution of scores produced by ScalarScoreVisitor
 * corresponding to different attributes. It allows a boolean
 * ignoreClosestCandidate; which when true will compute the attribution as it
 * that near neighbor was not present in RCF. This is turned on by default for
 * duplicate points seen by the forest, so that the attribution does not change
 * is a sequence of duplicate points are seen. For non-duplicate points, if the
 * boolean turned on, reduces effects of masking (when anomalous points are
 * included in the forest (which will be true with a few samples or when the
 * samples are not refreshed appropriately). It is worth remembering that
 * disallowing anomalous points from being included in the forest forest
 * explicitly will render the algorithm incapable of adjusting to a new normal
 * -- which is a strength of this algorithm.
 **/
public class AnomalyAttributionVisitor extends AbstractAttributionVisitor {

	public AnomalyAttributionVisitor(double[] pointToScore, int treeMass, int ignoreThreshold) {
		super(pointToScore, treeMass, ignoreThreshold);
	}

	public AnomalyAttributionVisitor(double[] pointToScore, int treeMass) {
		super(pointToScore, treeMass);
	}

	@Override
	protected double scoreSeen(int depth, int mass) {
		return CommonUtils.defaultScoreSeenFunction(depth, mass);
	}

	@Override
	protected double scoreUnseen(int depth, int mass) {
		return CommonUtils.defaultScoreUnseenFunction(depth, mass);
	}

	@Override
	protected double damp(int leafMass, int treeMass) {
		return CommonUtils.defaultDampFunction(leafMass, treeMass);
	}

}
