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

package com.amazon.randomcutforest;

import java.util.ArrayList;
import java.util.Collection;

/**
 * A ComponentList is an ArrayList specialized to contain IComponentModel
 * instances. Executor classes operate on ComponentLists.
 *
 * @param <P> The internal point representation expected by the component models
 *            in this list.
 */
public class ComponentList<P> extends ArrayList<IComponentModel<P>> {
    public ComponentList() {
        super();
    }

    public ComponentList(Collection<? extends IComponentModel<P>> collection) {
        super(collection);
    }

    public ComponentList(int initialCapacity) {
        super(initialCapacity);
    }
}
