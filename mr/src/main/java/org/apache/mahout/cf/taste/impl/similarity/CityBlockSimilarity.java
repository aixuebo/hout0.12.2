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
import org.apache.mahout.cf.taste.impl.common.FastIDSet;
import org.apache.mahout.cf.taste.impl.common.RefreshHelper;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.similarity.PreferenceInferrer;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

/**
 * Implementation of City Block distance (also known as Manhattan distance) - the absolute value of the difference of
 * each direction is summed.  The resulting unbounded distance is then mapped between 0 and 1.
 * 城市街区距离(City Block distance),也称 曼哈顿距离,即每一个位置的绝对值之和
 */
public final class CityBlockSimilarity extends AbstractItemSimilarity implements UserSimilarity {

  public CityBlockSimilarity(DataModel dataModel) {
    super(dataModel);
  }

  /**
   * @throws UnsupportedOperationException
   */
  @Override
  public void setPreferenceInferrer(PreferenceInferrer inferrer) {
    throw new UnsupportedOperationException();
  }

  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
    Collection<Refreshable> refreshed = RefreshHelper.buildRefreshed(alreadyRefreshed);
    RefreshHelper.maybeRefresh(refreshed, getDataModel());
  }

  //item和item之间的相似度
  @Override
  public double itemSimilarity(long itemID1, long itemID2) throws TasteException {
    DataModel dataModel = getDataModel();
    int preferring1 = dataModel.getNumUsersWithPreferenceFor(itemID1);//返回该item对应多少人持有该用户的偏好度
    int preferring2 = dataModel.getNumUsersWithPreferenceFor(itemID2);//返回该item对应多少人持有该用户的偏好度
    int intersection = dataModel.getNumUsersWithPreferenceFor(itemID1, itemID2);//返回同时包含这两个item的用户数量
    return doSimilarity(preferring1, preferring2, intersection);
  }

  @Override
  public double[] itemSimilarities(long itemID1, long[] itemID2s) throws TasteException {
    DataModel dataModel = getDataModel();
    int preferring1 = dataModel.getNumUsersWithPreferenceFor(itemID1);
    double[] distance = new double[itemID2s.length];
    for (int i = 0; i < itemID2s.length; ++i) {
      int preferring2 = dataModel.getNumUsersWithPreferenceFor(itemID2s[i]);
      int intersection = dataModel.getNumUsersWithPreferenceFor(itemID1, itemID2s[i]);
      distance[i] = doSimilarity(preferring1, preferring2, intersection);
    }
    return distance;
  }

  //user-user之间的相似度
  @Override
  public double userSimilarity(long userID1, long userID2) throws TasteException {
    DataModel dataModel = getDataModel();
    FastIDSet prefs1 = dataModel.getItemIDsFromUser(userID1);//返回该userid对应的一组itemid集合
    FastIDSet prefs2 = dataModel.getItemIDsFromUser(userID2);//返回该userid对应的一组itemid集合
    int prefs1Size = prefs1.size();
    int prefs2Size = prefs2.size();
    //获取两个user对应的itemid的交集数量
    int intersectionSize = prefs1Size < prefs2Size ? prefs2.intersectionSize(prefs1) : prefs1.intersectionSize(prefs2);
    return doSimilarity(prefs1Size, prefs2Size, intersectionSize);
  }

  /**
   * Calculate City Block Distance from total non-zero values and intersections and map to a similarity value.
   *
   * @param pref1        number of non-zero values in left vector 集合数量
   * @param pref2        number of non-zero values in right vector 集合数量
   * @param intersection number of overlapping non-zero values 两个集合交集数量
   * 
   * 确保结果是-1到1之间的数据,因此 1/大于1的数
   */
  private static double doSimilarity(int pref1, int pref2, int intersection) {
	  /**
	   * 这个应该算法应该不是曼哈顿距离算法,但是我这边也给一下算法逻辑
	   * 假设两个集合完全相似,则说明距离是0,比如a b集合都含有10个元素,但是相同的元素是10个,因此是10+10-20=0,说明距离就是0
	   * 
	   * 假设a有10个元素，b=6个元素  两者交集是6,因此就是10+6-12 = 4,说明也很近
	   * 假设a有10个元素，b=6个元素  两者交集是3,因此就是10+6-6 = 10,说明就远了
	   * 
	   * 所以公式是说 两者一共理论上能有多少集合数量 - 相似的数量*2,因为相似的是说明两个都相同,因此要*2,剩下多少就说明有多少是没匹配上的
	   * 
	   * 从这理论上来说也是曼哈顿距离,因为就做了减法,没做什么其他复杂的运算
	   */
    int distance = pref1 + pref2 - 2 * intersection;
    return 1.0 / (1.0 + distance);
  }

}
