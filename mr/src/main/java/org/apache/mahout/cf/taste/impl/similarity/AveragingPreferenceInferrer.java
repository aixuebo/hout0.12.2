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

package org.apache.mahout.cf.taste.impl.similarity;

import java.util.Collection;

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.Cache;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;
import org.apache.mahout.cf.taste.impl.common.Retriever;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.similarity.PreferenceInferrer;

/**
 * <p>
 * Implementations of this interface compute an inferred preference for a user and an item that the user has
 * not expressed any preference for. This might be an average of other preferences scores from that user, for
 * example. This technique is sometimes called "default voting".
 * </p>
 * 为每一个user打偏爱度分,该分数是该用户所有item对应value的平均值
 */
public final class AveragingPreferenceInferrer implements PreferenceInferrer {
  
  private static final Float ZERO = 0.0f;
  
  private final DataModel dataModel;
  private final Cache<Long,Float> averagePreferenceValue;
  
  public AveragingPreferenceInferrer(DataModel dataModel) throws TasteException {
    this.dataModel = dataModel;
    Retriever<Long,Float> retriever = new PrefRetriever();
    averagePreferenceValue = new Cache<>(retriever, dataModel.getNumUsers());
    refresh(null);
  }
  
  //返回该user对应的平均值,跟该item无关系
  @Override
  public float inferPreference(long userID, long itemID) throws TasteException {
    return averagePreferenceValue.get(userID);
  }
  
  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    averagePreferenceValue.clear();
  }
  
  private final class PrefRetriever implements Retriever<Long,Float> {
    
    @Override
    public Float get(Long key) throws TasteException {
      PreferenceArray prefs = dataModel.getPreferencesFromUser(key);//返回该userid对应的一组user-item-value集合,并且按照item排序好了
      int size = prefs.length();//该用户所有的item数量
      if (size == 0) {//说明该用户没有关注item
        return ZERO;
      }
      
      //获取该user所有item的平均值
      RunningAverage average = new FullRunningAverage();
      for (int i = 0; i < size; i++) {
        average.addDatum(prefs.getValue(i));
      }
      return (float) average.getAverage();
    }
  }
  
  @Override
  public String toString() {
    return "AveragingPreferenceInferrer";
  }
  
}
