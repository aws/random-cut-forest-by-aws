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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import com.amazon.randomcutforest.returntypes.ConditionalTreeSample;
import com.amazon.randomcutforest.returntypes.SampleSummary;
import com.amazon.randomcutforest.summarization.Summarizer;

public class ConditionalSampleSummarizerTest {

    private float[] queryPoint;
    private int[] missingIndexes;
    ConditionalSampleSummarizer summarizer;
    ConditionalSampleSummarizer projectedSummarizer;

    @BeforeEach
    public void setUp() {
        queryPoint = new float[] { 50, 70, 90, 100 };
        missingIndexes = new int[] { 2, 3 };
        summarizer = new ConditionalSampleSummarizer(missingIndexes, queryPoint, 0.2, false, 1, 0, 1);
        projectedSummarizer = new ConditionalSampleSummarizer(missingIndexes, queryPoint, 0.2, true, 5, 0.3, 1);
    }

    @Test
    public void testSummarize() {
        assertThrows(IllegalArgumentException.class, () -> summarizer.summarize(Collections.emptyList()));

        Random random = new Random(42);
        ArrayList<ConditionalTreeSample> list = new ArrayList<>();
        for (int i = 0; i < 999; i++) {
            float[] point = new float[] { 50, 70, 90, 100 + 2 * random.nextFloat() };
            list.add(new ConditionalTreeSample(i, null, Summarizer.L1distance(point, queryPoint), point));
        }
        list.add(new ConditionalTreeSample(999, null, 100, new float[] { 50, 70, 90, 200 }));

        SampleSummary summary = summarizer.summarize(list, false);
        assertNull(summary.summaryPoints);
        SampleSummary summaryTwo = summarizer.summarize(list, true);
        assertNotNull(summaryTwo.summaryPoints);
        for (float[] element : summaryTwo.summaryPoints) {
            assertEquals(element.length, 4);
            assertEquals(element[0], 50);
            assertEquals(element[1], 70);
            assertEquals(element[2], 90);
            assertTrue(100 < element[3] && element[3] < 102);
        }
        assertEquals(4, summaryTwo.mean.length);
        assertEquals(0, summaryTwo.deviation[0]);
        SampleSummary summaryThree = projectedSummarizer.summarize(list);
        assertNotNull(summaryThree.summaryPoints);
        for (float[] element : summaryThree.summaryPoints) {
            assertEquals(element.length, missingIndexes.length);
        }
    }

    @Test
    public void testZero() {
        ArrayList<ConditionalTreeSample> list = new ArrayList<>();
        for (int i = 0; i < 1000; i++) {
            list.add(new ConditionalTreeSample(i, null, 0, queryPoint));
        }
        assert (summarizer.summarize(list, true).summaryPoints.length == 1);
    }

}
