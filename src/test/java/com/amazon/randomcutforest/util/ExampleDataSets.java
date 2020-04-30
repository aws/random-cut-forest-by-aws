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

package com.amazon.randomcutforest.util;

import java.util.Random;

import static java.lang.Math.PI;
import static java.lang.Math.cos;
import static java.lang.Math.sin;

/**
 * This class samples point from a mixture of 2 multi-variate normal
 * distribution with covariance matrices of the form sigma * I. One of the
 * normal distributions is considered the base distribution, the second is
 * considered the anomaly distribution, and there are random transitions between
 * the two.
 */
public class ExampleDataSets {

	public static double[][] generateFan(int numberPerBlade, int numberOfBlades) {
		if ((numberOfBlades > 12) || (numberPerBlade <= 0))
			return null;
		int newDimensions = 2;
		int dataSize = numberOfBlades * numberPerBlade;

		Random prg = new Random(0);
		NormalMixtureTestData generator = new NormalMixtureTestData(0.0, 1.0, 0.0, 1.0, 0, 1);
		double[][] data = generator.generateTestData(dataSize, newDimensions, 100);

		double[][] transformedData = new double[data.length][newDimensions];
		for (int j = 0; j < data.length; j++) {

			// shrink

			transformedData[j][0] = 0.05 * data[j][0];
			transformedData[j][1] = 0.2 * data[j][1];
			double toss = prg.nextDouble();

			// rotate
			int i = 0;
			while (i < numberOfBlades + 1) {
				if (toss < i * 1.0 / numberOfBlades) {
					double[] vec = rotateClockWise(transformedData[j], 2 * PI * i / numberOfBlades);
					transformedData[j][0] = vec[0] + 0.6 * sin(2 * PI * i / numberOfBlades);
					transformedData[j][1] = vec[1] + 0.6 * cos(2 * PI * i / numberOfBlades);
					break;
				} else
					++i;
			}
		}
		return transformedData;

	}

	static double[] rotateClockWise(double[] point, double theta) {
		double[] result = new double[2];
		result[0] = cos(theta) * point[0] + sin(theta) * point[1];
		result[1] = -sin(theta) * point[0] + cos(theta) * point[1];
		return result;
	}

}
