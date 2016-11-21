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

import org.apache.mahout.cf.taste.recommender.ItemBasedRecommender;

import java.io.IOException;

public abstract class BatchItemSimilarities {

  private final ItemBasedRecommender recommender;//真正去计算两个item如何相似的算法接口
  private final int similarItemsPerItem;//每一个item找到多少个与其相似的item

  /**
   * @param recommender recommender to use
   * @param similarItemsPerItem number of similar items to compute per item
   */
  protected BatchItemSimilarities(ItemBasedRecommender recommender, int similarItemsPerItem) {
    this.recommender = recommender;
    this.similarItemsPerItem = similarItemsPerItem;
  }

  protected ItemBasedRecommender getRecommender() {
    return recommender;
  }

  protected int getSimilarItemsPerItem() {
    return similarItemsPerItem;
  }

  /**
   * @param degreeOfParallelism number of threads to use for the computation 多少个线程处理
   * @param maxDurationInHours  maximum duration of the computation 计算的最大时间
   * @param writer  {@link SimilarItemsWriter} used to persist the results 最终数据写入到哪里
   * @return  the number of similarities precomputed 返回值是一共多少个item-item-value数据
   * @throws IOException
   * @throws RuntimeException if the computation takes longer than maxDurationInHours
   */
  public abstract int computeItemSimilarities(int degreeOfParallelism, int maxDurationInHours,
      SimilarItemsWriter writer) throws IOException;
}
