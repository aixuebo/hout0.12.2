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

import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.Callable;

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.RefreshHelper;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.IDRescorer;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Rescorer;
import org.apache.mahout.cf.taste.recommender.UserBasedRecommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;
import org.apache.mahout.common.LongPair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.base.Preconditions;

/**
 * <p>
 * A simple {@link org.apache.mahout.cf.taste.recommender.Recommender}
 * which uses a given {@link DataModel} and {@link UserNeighborhood} to produce recommendations.
 * </p>
 * 为一个用户推荐相关联的用户，以及为一个用户推荐可能他需要的商品集合
 * GenericUserBasedRecommender逻辑是 userid--找到相似的userid集合--然后找到相似userid购买了哪些商品--进行推荐
 * 
 * 而GenericItemBasedRecommender 类逻辑 根据推荐的用户---找到用户喜欢的itemid集合---找到这些itemid相似的哪些itemid进行推荐
 */
public class GenericUserBasedRecommender extends AbstractRecommender implements UserBasedRecommender {
  
  private static final Logger log = LoggerFactory.getLogger(GenericUserBasedRecommender.class);
  
  private final UserNeighborhood neighborhood;
  private final UserSimilarity similarity;
  private final RefreshHelper refreshHelper;
  private EstimatedPreferenceCapper capper;
  
  public GenericUserBasedRecommender(DataModel dataModel,
                                     UserNeighborhood neighborhood,
                                     UserSimilarity similarity) {
    super(dataModel);
    Preconditions.checkArgument(neighborhood != null, "neighborhood is null");
    this.neighborhood = neighborhood;
    this.similarity = similarity;
    this.refreshHelper = new RefreshHelper(new Callable<Void>() {
      @Override
      public Void call() {
        capper = buildCapper();
        return null;
      }
    });
    refreshHelper.addDependency(dataModel);
    refreshHelper.addDependency(similarity);
    refreshHelper.addDependency(neighborhood);
    capper = buildCapper();
  }
  
  public UserSimilarity getSimilarity() {
    return similarity;
  }
  
  /**
   * 给userid推荐商品item,返回最有可能的howMany个item
   * 原理:
   * 找到user的相似邻居,---得到用户可能要的商品集合---选择howMany个item
   */
  @Override
  public List<RecommendedItem> recommend(long userID, int howMany, IDRescorer rescorer, boolean includeKnownItems)
    throws TasteException {
    Preconditions.checkArgument(howMany >= 1, "howMany must be at least 1");

    log.debug("Recommending items for user ID '{}'", userID);

    long[] theNeighborhood = neighborhood.getUserNeighborhood(userID);//返回该user的邻居,即与该用户相似的用户集合.属于 user推荐user

    if (theNeighborhood.length == 0) {//没有邻居.因此没有邀请记录
      return Collections.emptyList();
    }

    //返回用户可能要买的商品item集合--即来自相似的邻居都买什么了,减去 用户自己拥有的,剩余就是用户可能购买的
    FastIDSet allItemIDs = getAllOtherItems(theNeighborhood, userID, includeKnownItems);

    TopItems.Estimator<Long> estimator = new Estimator(userID, theNeighborhood);

    //循环用户可能会买的商品,选择最可能买的howMany个
    List<RecommendedItem> topItems = TopItems
        .getTopItems(howMany, allItemIDs.iterator(), rescorer, estimator);

    log.debug("Recommendations are: {}", topItems);
    return topItems;
  }
  
  //获取user对item的偏好度,如果已经设置了,则直接返回,如果没有,则要预估
  @Override
  public float estimatePreference(long userID, long itemID) throws TasteException {
    DataModel model = getDataModel();
    Float actualPref = model.getPreferenceValue(userID, itemID);//返回userid-itemid对应的偏好度value,如果不存在,则返回null
    if (actualPref != null) {//说明已经设置了,因此就直接返回偏好度即可
      return actualPref;
    }
    long[] theNeighborhood = neighborhood.getUserNeighborhood(userID);//找到该user的相似user集合
    return doEstimatePreference(userID, theNeighborhood, itemID);//看相似的user是否买该商品,然后计算该用户买该商品的概率
  }
  
  //计算最有可能与该用户相似的用户集合
  @Override
  public long[] mostSimilarUserIDs(long userID, int howMany) throws TasteException {
    return mostSimilarUserIDs(userID, howMany, null);
  }
  
  //计算最有可能与该用户相似的用户集合
  @Override
  public long[] mostSimilarUserIDs(long userID, int howMany, Rescorer<LongPair> rescorer) throws TasteException {
    TopItems.Estimator<Long> estimator = new MostSimilarEstimator(userID, similarity, rescorer);
    return doMostSimilarUsers(howMany, estimator);
  }
  
  //计算最有可能与该用户相似的用户集合
  private long[] doMostSimilarUsers(int howMany, TopItems.Estimator<Long> estimator) throws TasteException {
    DataModel model = getDataModel();
    return TopItems.getTopUsers(howMany, model.getUserIDs(), null, estimator);
  }
  
  /**
   * 
   * @param theUserID 指定userid
   * @param theNeighborhood 对应theUserID相似的邻居集合
   * @param itemID 为theUserID推荐这个商品
   * @return 返回用户可能买该商品的概率
   * @throws TasteException
   */
  protected float doEstimatePreference(long theUserID, long[] theNeighborhood, long itemID) throws TasteException {
    if (theNeighborhood.length == 0) {//没有相似邻居,推荐就是0
      return Float.NaN;
    }
    DataModel dataModel = getDataModel();
    double preference = 0.0;//所有相似的user度*对应item的偏好之和
    double totalSimilarity = 0.0;//所有user相似度之和
    int count = 0;//多少个邻居中购买了该商品
    for (long userID : theNeighborhood) {
      if (userID != theUserID) {
        // See GenericItemBasedRecommender.doEstimatePreference() too
        Float pref = dataModel.getPreferenceValue(userID, itemID);//返回userid-itemid对应的偏好度value,如果不存在,则返回null
        if (pref != null) {
          double theSimilarity = similarity.userSimilarity(theUserID, userID);//计算user之间的相似度
          if (!Double.isNaN(theSimilarity)) {
            preference += theSimilarity * pref;//所有相似的user度*对应item的偏好之和
            totalSimilarity += theSimilarity;
            count++;
          }
        }
      }
    }
    // Throw out the estimate if it was based on no data points, of course, but also if based on
    // just one. This is a bit of a band-aid on the 'stock' item-based algorithm for the moment.
    // The reason is that in this case the estimate is, simply, the user's rating for one item
    // that happened to have a defined similarity. The similarity score doesn't matter, and that
    // seems like a bad situation.
    if (count <= 1) {//说明该商品可能就一个用户买过,因此不推荐
      return Float.NaN;
    }
    float estimate = (float) (preference / totalSimilarity);//得到购买该商品的概率,即 所有相似的user度*对应item的偏好之和 / 所有user相似度之和,即每一个相似度下,可能买的情况
    if (capper != null) {
      estimate = capper.capEstimate(estimate);
    }
    return estimate;
  }
  
  /**
   * 
   * @param theNeighborhood 与要推荐的用户(theUserID)相似的用户集合
   * @param theUserID 要推荐的用户
   * @param includeKnownItems true表示包含已经知道的item,false表示不包含已知的item
   * @return 返回用户可能要买的商品item集合
   * @throws TasteException
   */
  protected FastIDSet getAllOtherItems(long[] theNeighborhood, long theUserID, boolean includeKnownItems)
    throws TasteException {
    DataModel dataModel = getDataModel();
    FastIDSet possibleItemIDs = new FastIDSet();//可能要推荐的商品集合
    for (long userID : theNeighborhood) {//循环该用户相似的用户集合
      possibleItemIDs.addAll(dataModel.getItemIDsFromUser(userID));//添加相似用户对应的一组itemid集合
    }
    if (!includeKnownItems) {//表示不包含已知的item,因此要移除属于user的item
      possibleItemIDs.removeAll(dataModel.getItemIDsFromUser(theUserID));
    }
    return possibleItemIDs;
  }
  
  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    refreshHelper.refresh(alreadyRefreshed);
  }
  
  @Override
  public String toString() {
    return "GenericUserBasedRecommender[neighborhood:" + neighborhood + ']';
  }

  private EstimatedPreferenceCapper buildCapper() {
    DataModel dataModel = getDataModel();
    if (Float.isNaN(dataModel.getMinPreference()) && Float.isNaN(dataModel.getMaxPreference())) {
      return null;
    } else {
      return new EstimatedPreferenceCapper(dataModel);//约束.让最终用户的偏好度一定在max和min之间
    }
  }
  
  //估计两个user之间的相似度
  private static final class MostSimilarEstimator implements TopItems.Estimator<Long> {
    
    private final long toUserID;
    private final UserSimilarity similarity;
    private final Rescorer<LongPair> rescorer;
    
    private MostSimilarEstimator(long toUserID, UserSimilarity similarity, Rescorer<LongPair> rescorer) {
      this.toUserID = toUserID;
      this.similarity = similarity;
      this.rescorer = rescorer;
    }
    
    @Override
    public double estimate(Long userID) throws TasteException {
      // Don't consider the user itself as a possible most similar user
      if (userID == toUserID) {//不考虑用户自己跟自己比较相似度
        return Double.NaN;
      }
      if (rescorer == null) {
        return similarity.userSimilarity(toUserID, userID);//判断两个user的相似度
      } else {
        LongPair pair = new LongPair(toUserID, userID);
        if (rescorer.isFiltered(pair)) {
          return Double.NaN;
        }
        double originalEstimate = similarity.userSimilarity(toUserID, userID);
        return rescorer.rescore(pair, originalEstimate);//重新计算一下该用户的相似度
      }
    }
  }
  
  private final class Estimator implements TopItems.Estimator<Long> {
    
    private final long theUserID;//给定用户
    private final long[] theNeighborhood;//用户的邻居,相似用户集合
    
    Estimator(long theUserID, long[] theNeighborhood) {
      this.theUserID = theUserID;
      this.theNeighborhood = theNeighborhood;
    }
    
    //计算用户可能购买item的可能性
    @Override
    public double estimate(Long itemID) throws TasteException {
      return doEstimatePreference(theUserID, theNeighborhood, itemID);
    }
  }
}
