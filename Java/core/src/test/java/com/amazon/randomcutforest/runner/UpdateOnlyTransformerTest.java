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


package com.amazon.randomcutforest.runner;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;

import com.amazon.randomcutforest.RandomCutForest;
import java.util.List;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

public class UpdateOnlyTransformerTest {

    private RandomCutForest forest;
    private UpdateOnlyTransformer transformer;

    @BeforeEach
    public void setUp() {
        forest = mock(RandomCutForest.class);
        transformer = new UpdateOnlyTransformer(forest);
    }

    @Test
    public void testGetResultValues() {
        List<String> result = transformer.getResultValues(1.0, 2.0, 3.0);
        assertTrue(result.isEmpty());
        verify(forest).update(new double[] {1.0, 2.0, 3.0});
    }

    @Test
    public void testGetEmptyResultValue() {
        assertTrue(transformer.getEmptyResultValue().isEmpty());
    }

    @Test
    public void testGetResultColumnNames() {
        assertTrue(transformer.getResultColumnNames().isEmpty());
    }
}
