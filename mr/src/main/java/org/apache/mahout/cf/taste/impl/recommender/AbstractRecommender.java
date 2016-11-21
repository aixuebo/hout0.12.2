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

package org.apache.mahout.cf.taste.impl.recommender;

import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.recommender.CandidateItemsStrategy;

import java.util.List;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.recommender.IDRescorer;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Preconditions;

public abstract class AbstractRecommender implements Recommender {
  
  private static final Logger log = LoggerFactory.getLogger(AbstractRecommender.class);
  
  private final DataModel dataModel;//数据源对象
  private final CandidateItemsStrategy candidateItemsStrategy;//选举item的策略
  
  protected AbstractRecommender(DataModel dataModel, CandidateItemsStrategy candidateItemsStrategy) {
    this.dataModel = Preconditions.checkNotNull(dataModel);
    this.candidateItemsStrategy = Preconditions.checkNotNull(candidateItemsStrategy);
  }

  protected AbstractRecommender(DataModel dataModel) {
    this(dataModel, getDefaultCandidateItemsStrategy());
  }

  //默认选举策略
  protected static CandidateItemsStrategy getDefaultCandidateItemsStrategy() {
    return new PreferredItemsNeighborhoodCandidateItemsStrategy();
  }


  /**
   * <p>
   * Default implementation which just calls
   * {@link Recommender#recommend(long, int, org.apache.mahout.cf.taste.recommender.IDRescorer)}, with a
   * {@link org.apache.mahout.cf.taste.recommender.Rescorer} that does nothing.
   * </p>
   * 给userid推荐商品item,返回最有可能的howMany个item
   */
  @Override
  public List<RecommendedItem> recommend(long userID, int howMany) throws TasteException {
    return recommend(userID, howMany, null, false);
  }

  /**
   * <p>
   * Default implementation which just calls
   * {@link Recommender#recommend(long, int, org.apache.mahout.cf.taste.recommender.IDRescorer)}, with a
   * {@link org.apache.mahout.cf.taste.recommender.Rescorer} that does nothing.
   * </p>
   * 给userid推荐商品item,返回最有可能的howMany个item
   * 
   * 参数includeKnownItems true表示推荐的商品中不包含userid本来有兴趣的商品
   */
  @Override
  public List<RecommendedItem> recommend(long userID, int howMany, boolean includeKnownItems) throws TasteException {
    return recommend(userID, howMany, null, includeKnownItems);
  }
  
  /**
   * <p> Delegates to {@link Recommender#recommend(long, int, IDRescorer, boolean)}
   * 给userid推荐商品item,返回最有可能的howMany个item
   * 
   * 参数rescorer 说明在最终推荐什么产品之前,可以重新给用户打分,用于自定义扩展,有时候业务需求,有时候根据某种业务可能会大更多的分数
   */
  @Override
  public List<RecommendedItem> recommend(long userID, int howMany, IDRescorer rescorer) throws TasteException{
    return recommend(userID, howMany,rescorer, false);  
  }
  
  /**
   * <p>
   * Default implementation which just calls {@link DataModel#setPreference(long, long, float)}.
   * </p>
   *
   * @throws IllegalArgumentException
   *           if userID or itemID is {@code null}, or if value is {@link Double#NaN}
   * 添加一个user-item-偏好分数数据
   */
  @Override
  public void setPreference(long userID, long itemID, float value) throws TasteException {
    Preconditions.checkArgument(!Float.isNaN(value), "NaN value");
    log.debug("Setting preference for user {}, item {}", userID, itemID);
    dataModel.setPreference(userID, itemID, value);
  }
  
  /**
   * <p>
   * Default implementation which just calls {@link DataModel#removePreference(long, long)} (Object, Object)}.
   * </p>
   *
   * @throws IllegalArgumentException
   *           if userID or itemID is {@code null}
   * 移除一个user-item的样本
   */
  @Override
  public void removePreference(long userID, long itemID) throws TasteException {
    log.debug("Remove preference for user '{}', item '{}'", userID, itemID);
    dataModel.removePreference(userID, itemID);
  }
  
  @Override
  public DataModel getDataModel() {
    return dataModel;
  }

  /**
   * @param userID
   *          ID of user being evaluated 准备推荐的用户
   * @param preferencesFromUser
   *          the preferences from the user 与userID相似的用户集合
   * @param includeKnownItems
   *          whether to include items already known by the user in recommendations  true表示该userID已经有的商品是不被再次推荐的
   * @return all items in the {@link DataModel} for which the user has not expressed a preference and could
   *         possibly be recommended to the user
   * @throws TasteException
   *           if an error occurs while listing items
   * 返回用户可能要买的商品item集合
   */
  protected FastIDSet getAllOtherItems(long userID, PreferenceArray preferencesFromUser, boolean includeKnownItems)
      throws TasteException {
    return candidateItemsStrategy.getCandidateItems(userID, preferencesFromUser, dataModel, includeKnownItems);
  }
  
}
