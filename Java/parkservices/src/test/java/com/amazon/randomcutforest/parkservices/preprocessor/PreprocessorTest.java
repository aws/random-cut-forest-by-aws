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

package com.amazon.randomcutforest.parkservices.preprocessor;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

import com.amazon.randomcutforest.config.ForestMode;
import com.amazon.randomcutforest.config.TransformMethod;

public class PreprocessorTest {

    @Test
    void constructorTest() {
        assertThrows(IllegalArgumentException.class, () -> {
            new Preprocessor.Builder<>().transformMethod(null).build();
        });
        assertThrows(IllegalArgumentException.class, () -> {
            new Preprocessor.Builder<>().transformMethod(TransformMethod.NONE).forestMode(null).build();
        });
        assertThrows(IllegalArgumentException.class, () -> {
            new Preprocessor.Builder<>().transformMethod(TransformMethod.NONE).forestMode(ForestMode.STANDARD).build();
        });
        assertThrows(IllegalArgumentException.class, () -> {
            new Preprocessor.Builder<>().transformMethod(TransformMethod.NONE).forestMode(ForestMode.STANDARD)
                    .inputLength(10).build();
        });
        assertThrows(IllegalArgumentException.class, () -> {
            new Preprocessor.Builder<>().transformMethod(TransformMethod.NONE).forestMode(ForestMode.STANDARD)
                    .inputLength(10).dimensions(12).build();
        });
        assertDoesNotThrow(() -> {
            new Preprocessor.Builder<>().transformMethod(TransformMethod.NONE).forestMode(ForestMode.STANDARD)
                    .inputLength(12).dimensions(12).build();
        });
        assertThrows(IllegalArgumentException.class, () -> {
            new Preprocessor.Builder<>().transformMethod(TransformMethod.NONE).forestMode(ForestMode.TIME_AUGMENTED)
                    .inputLength(12).dimensions(12).build();
        });
        assertDoesNotThrow(() -> {
            new Preprocessor.Builder<>().transformMethod(TransformMethod.NONE).forestMode(ForestMode.TIME_AUGMENTED)
                    .inputLength(12).dimensions(13).build();
        });
        assertThrows(IllegalArgumentException.class, () -> {
            new Preprocessor.Builder<>().transformMethod(TransformMethod.NONE).shingleSize(2)
                    .forestMode(ForestMode.TIME_AUGMENTED).inputLength(12).dimensions(13).build();
        });
        assertThrows(IllegalArgumentException.class, () -> {
            new Preprocessor.Builder<>().transformMethod(TransformMethod.NONE).shingleSize(2)
                    .forestMode(ForestMode.TIME_AUGMENTED).inputLength(12).dimensions(14).build();
        });
        assertDoesNotThrow(() -> {
            new Preprocessor.Builder<>().transformMethod(TransformMethod.NONE).shingleSize(2)
                    .forestMode(ForestMode.TIME_AUGMENTED).inputLength(6).dimensions(14).build();
        });
        assertDoesNotThrow(() -> {
            new Preprocessor.Builder<>().transformMethod(TransformMethod.NONE).shingleSize(2).normalizeTime(true)
                    .forestMode(ForestMode.TIME_AUGMENTED).inputLength(6).dimensions(14).build();
        });

        // external shingling in STANDARD mode
        assertDoesNotThrow(() -> {
            new Preprocessor.Builder<>().transformMethod(TransformMethod.NONE).shingleSize(2)
                    .forestMode(ForestMode.TIME_AUGMENTED).inputLength(6).dimensions(14).build();
        });

        // internal shingling
        assertDoesNotThrow(() -> {
            new Preprocessor.Builder<>().transformMethod(TransformMethod.NONE).shingleSize(2)
                    .forestMode(ForestMode.STANDARD).inputLength(6).dimensions(12).build();
        });
        assertThrows(IllegalArgumentException.class, () -> {
            new Preprocessor.Builder<>().transformMethod(TransformMethod.NONE).forestMode(ForestMode.STANDARD)
                    .inputLength(6).dimensions(12).build();
        });
        assertDoesNotThrow(() -> {
            new Preprocessor.Builder<>().transformMethod(TransformMethod.NONE).shingleSize(2).normalizeTime(true)
                    .forestMode(ForestMode.STANDARD).inputLength(6).dimensions(12).build();
        });
        assertThrows(IllegalArgumentException.class, () -> {
            new Preprocessor.Builder<>().transformMethod(TransformMethod.NONE).shingleSize(2)
                    .forestMode(ForestMode.STANDARD).inputLength(5).dimensions(12).build();
        });
    }

}
