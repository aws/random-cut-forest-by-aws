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

package com.amazon.randomcutforest.runner;

import java.util.List;

import com.amazon.randomcutforest.RandomCutForest;

/**
 * This interface is used by SimpleRunner to transform input lines into output
 * lines.
 */
public interface LineTransformer {

	/**
	 * For the given parsed input point, return a list of string values that should
	 * be written as output. The list of strings will be joined together using the
	 * user-specified delimiter.
	 * 
	 * @param point
	 *            A point value that was parsed from the input stream.
	 * @return a list of string values that should be written as output.
	 */
	List<String> getResultValues(double[] point);

	/**
	 * @return a list of string values that should be written to the output when
	 *         processing a line if there is no input point available. This method
	 *         is invoked when shingling is enabled before the first shingle is
	 *         full.
	 */
	List<String> getEmptyResultValue();

	/**
	 * @return a list of column names to write to the output if headers are enabled.
	 */
	List<String> getResultColumnNames();

	/**
	 * @return the RandomCutForest instance which is being used internally to
	 *         process lines.
	 */
	RandomCutForest getForest();
}
