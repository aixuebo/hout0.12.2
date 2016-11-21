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

import com.google.common.collect.UnmodifiableIterator;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;

import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;

/**
 * Compact representation of all similar items for an item
 * 代表一个item和他所有关联的item的相似度集合
 */
public class SimilarItems {

  private final long itemID;
  private final long[] similarItemIDs;//跟itemID相关联的item集合
  private final double[] similarities;//每一个关联的相似度

  /**
   * @param similarItems 该item拥有相似的item集合
   */
  public SimilarItems(long itemID, List<RecommendedItem> similarItems) {
    this.itemID = itemID;

    int numSimilarItems = similarItems.size();
    similarItemIDs = new long[numSimilarItems];
    similarities = new double[numSimilarItems];

    for (int n = 0; n < numSimilarItems; n++) {
      similarItemIDs[n] = similarItems.get(n).getItemID();
      similarities[n] = similarItems.get(n).getValue();
    }
  }

  public long getItemID() {
    return itemID;
  }

  //返回跟item相关联的有多少个item
  public int numSimilarItems() {
    return similarItemIDs.length;
  }

  //迭代每一个相关联的item--返回的是item-value
  public Iterable<SimilarItem> getSimilarItems() {
    return new Iterable<SimilarItem>() {
      @Override
      public Iterator<SimilarItem> iterator() {
        return new SimilarItemsIterator();
      }
    };
  }

  //迭代每一个相关联的item--返回的是item-value
  private class SimilarItemsIterator extends UnmodifiableIterator<SimilarItem> {

    private int index = -1;

    @Override
    public boolean hasNext() {
      return index < (similarItemIDs.length - 1);
    }

    @Override
    public SimilarItem next() {
      if (!hasNext()) {
        throw new NoSuchElementException();
      }
      index++;
      return new SimilarItem(similarItemIDs[index], similarities[index]);
    }
  }
}
