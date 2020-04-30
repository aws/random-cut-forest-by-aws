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

package com.amazon.randomcutforest.imputation;

import com.amazon.randomcutforest.CommonUtils;
import com.amazon.randomcutforest.tree.BoundingBox;
import com.amazon.randomcutforest.tree.Cut;
import com.amazon.randomcutforest.tree.Node;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static com.amazon.randomcutforest.CommonUtils.defaultScoreSeenFunction;
import static com.amazon.randomcutforest.CommonUtils.defaultScoreUnseenFunction;
import static com.amazon.randomcutforest.TestUtils.EPSILON;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.when;

public class ImputeVisitorTest {

	private double[] queryPoint;
	private int numberOfMissingValues;
	private int[] missingIndexes;
	private ImputeVisitor visitor;

	@BeforeEach
	public void setUp() {
		// create a point where the 2nd value is missing
		// The second value of queryPoint and the 2nd and 3rd values of missingIndexes
		// should be ignored in all tests

		queryPoint = new double[]{-1.0, 1000.0, 3.0};
		numberOfMissingValues = 1;
		missingIndexes = new int[]{1, 99, -888};

		visitor = new ImputeVisitor(queryPoint, numberOfMissingValues, missingIndexes);
	}

	@Test
	public void testNew() {
		assertArrayEquals(queryPoint, visitor.getResult());
		assertNotSame(queryPoint, visitor.getResult());
		assertEquals(10.0, visitor.getRank());
	}

	@Test
	public void testCopyConstructor() {
		ImputeVisitor copy = new ImputeVisitor(visitor);
		assertArrayEquals(queryPoint, copy.getResult());
		assertNotSame(copy.getResult(), visitor.getResult());
		assertEquals(10.0, visitor.getRank());
	}

	@Test
	public void testAcceptLeafEquals() {
		double[] point = {queryPoint[0], 2.0, queryPoint[2]};
		Node leafNode = spy(new Node(point));

		int leafDepth = 100;
		int leafMass = 10;
		when(leafNode.getMass()).thenReturn(leafMass);

		visitor.acceptLeaf(leafNode, leafDepth);

		double[] expected = new double[]{-1.0, 2.0, 3.0};
		assertArrayEquals(expected, visitor.getResult());

		assertEquals(defaultScoreSeenFunction(leafDepth, leafMass), visitor.getRank());
	}

	@Test
	public void testAcceptLeafEqualsZeroDepth() {
		double[] point = {queryPoint[0], 2.0, queryPoint[2]};
		Node leafNode = spy(new Node(point));

		int leafDepth = 0;
		int leafMass = 10;
		when(leafNode.getMass()).thenReturn(leafMass);

		visitor.acceptLeaf(leafNode, leafDepth);

		double[] expected = new double[]{-1.0, 2.0, 3.0};
		assertArrayEquals(expected, visitor.getResult());

		assertEquals(0.0, visitor.getRank());
	}

	@Test
	public void testAcceptLeafNotEquals() {
		double[] point = {queryPoint[0], 2.0, -111.11};
		Node leafNode = spy(new Node(point));

		int leafDepth = 100;
		int leafMass = 10;
		when(leafNode.getMass()).thenReturn(leafMass);

		visitor.acceptLeaf(leafNode, leafDepth);

		double[] expected = new double[]{-1.0, 2.0, 3.0};
		assertArrayEquals(expected, visitor.getResult());

		assertEquals(defaultScoreUnseenFunction(leafDepth, leafMass), visitor.getRank());
	}

	@Test
	public void testAccept() {

		double[] point = {queryPoint[0], 2.0, -111.11};
		Node node = spy(new Node(point));

		int depth = 100;
		int leafMass = 10;
		when(node.getMass()).thenReturn(leafMass);

		visitor.acceptLeaf(node, depth);

		double[] expected = new double[]{-1.0, 2.0, 3.0};
		assertArrayEquals(expected, visitor.getResult());

		assertEquals(defaultScoreUnseenFunction(depth, leafMass), visitor.getRank());

		depth--;
		BoundingBox boundingBox = node.getBoundingBox().getMergedBox(new double[]{99.0, 4.0, -19.0});
		node = spy(new Node(null, null, null, boundingBox));
		when(node.getMass()).thenReturn(leafMass + 2);

		double oldRank = visitor.getRank();
		visitor.accept(node, depth);
		assertArrayEquals(expected, visitor.getResult());

		double p = CommonUtils.getProbabilityOfSeparation(boundingBox, expected);
		double expectedRank = p * defaultScoreUnseenFunction(depth, node.getMass()) + (1 - p) * oldRank;
		assertEquals(expectedRank, visitor.getRank(), EPSILON);
	}

	@Test
	public void testTrigger() {
		Cut cut = mock(Cut.class);
		when(cut.getDimension()).thenReturn(1).thenReturn(0);

		Node node = mock(Node.class);
		when(node.getCut()).thenReturn(cut);

		assertTrue(visitor.trigger(node));
		assertFalse(visitor.trigger(node));
	}

	@Test
	public void testNewCopy() {
		ImputeVisitor copy = (ImputeVisitor) visitor.newCopy();
		assertArrayEquals(queryPoint, copy.getResult());
		assertNotSame(copy.getResult(), visitor.getResult());
		assertEquals(10.0, visitor.getRank());
	}

	@Test
	public void testMerge() {
		double[] otherPoint = new double[]{99, 100, 101};
		ImputeVisitor other = new ImputeVisitor(otherPoint, 0, null);

		// set other.rank to a small value
		Node node = new Node(new double[]{0, 0, 0});
		other.acceptLeaf(node, 99);

		assertTrue(other.getRank() < visitor.getRank());

		other.combine(visitor);
		assertArrayEquals(otherPoint, other.getResult());

		visitor.combine(other);
		assertArrayEquals(otherPoint, visitor.getResult());
	}
}
