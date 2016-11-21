/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.cf.taste.similarity.precompute;

import java.io.Closeable;
import java.io.IOException;

/**
 * Used to persist the results of a batch item similarity computation
 * conducted with a {@link BatchItemSimilarities} implementation
 * 表示如何将item1-time2-value的信息写入成文件
 */
public interface SimilarItemsWriter extends Closeable {

  void open() throws IOException;

  //添加一个item与item-value集合
  void add(SimilarItems similarItems) throws IOException;

}
