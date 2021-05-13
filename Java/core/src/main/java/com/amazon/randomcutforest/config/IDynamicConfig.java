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


package com.amazon.randomcutforest.config;

/**
 * This interface is used by model classes to configure model parameters by name. This is intended
 * primarily for settings that a user may want to change at runtime.
 */
public interface IDynamicConfig {

    <T> void setConfig(String name, T value, Class<T> clazz);

    default void setConfig(String name, short value) {
        setConfig(name, value, Short.class);
    }

    default void setConfig(String name, int value) {
        setConfig(name, value, Integer.class);
    }

    default void setConfig(String name, long value) {
        setConfig(name, value, Long.class);
    }

    default void setConfig(String name, float value) {
        setConfig(name, value, Float.class);
    }

    default void setConfig(String name, double value) {
        setConfig(name, value, Double.class);
    }

    default void setConfig(String name, boolean value) {
        setConfig(name, value, Boolean.class);
    }

    <T> T getConfig(String name, Class<T> clazz);

    default Object getConfig(String name) {
        return getConfig(name, Object.class);
    }
}
