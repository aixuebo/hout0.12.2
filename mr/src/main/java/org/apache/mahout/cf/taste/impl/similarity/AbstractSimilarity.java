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
import java.util.concurrent.Callable;

import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.common.Weighting;
import org.apache.mahout.cf.taste.impl.common.RefreshHelper;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.similarity.PreferenceInferrer;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;

import com.google.common.base.Preconditions;

/** Abstract superclass encapsulating functionality that is common to most implementations in this package. */
abstract class AbstractSimilarity extends AbstractItemSimilarity implements UserSimilarity {

  private PreferenceInferrer inferrer;//是否可以对user--item进行推断分数
  private final boolean weighted;//是否是Weighting.WEIGHTED枚举类
  private final boolean centerData;
  private int cachedNumItems;//一共多少个item
  private int cachedNumUsers;//一共多少个user
  private final RefreshHelper refreshHelper;

  /**
   * <p>
   * Creates a possibly weighted {@link AbstractSimilarity}.
   * </p>
   */
  AbstractSimilarity(final DataModel dataModel, Weighting weighting, boolean centerData) throws TasteException {
    super(dataModel);
    this.weighted = weighting == Weighting.WEIGHTED;
    this.centerData = centerData;
    this.cachedNumItems = dataModel.getNumItems();
    this.cachedNumUsers = dataModel.getNumUsers();
    this.refreshHelper = new RefreshHelper(new Callable<Object>() {
      @Override
      public Object call() throws TasteException {
        cachedNumItems = dataModel.getNumItems();
        cachedNumUsers = dataModel.getNumUsers();
        return null;
      }
    });
  }

  final PreferenceInferrer getPreferenceInferrer() {
    return inferrer;
  }
  
  @Override
  public final void setPreferenceInferrer(PreferenceInferrer inferrer) {
    Preconditions.checkArgument(inferrer != null, "inferrer is null");
    refreshHelper.addDependency(inferrer);
    refreshHelper.removeDependency(this.inferrer);
    this.inferrer = inferrer;
  }
  
  final boolean isWeighted() {
    return weighted;
  }
  
  /**
   * <p>
   * Several subclasses in this package implement this method to actually compute the similarity from figures
   * computed over users or items. Note that the computations in this class "center" the data, such that X and
   * Y's mean are 0.
   * </p>
   * 
   * <p>
   * Note that the sum of all X and Y values must then be 0. This value isn't passed down into the standard
   * similarity computations as a result.
   * </p>
   * 
   * @param n
   *          total number of users or items
   * @param sumXY
   *          sum of product of user/item preference values, over all items/users preferred by both
   *          users/items
   * @param sumX2
   *          sum of the square of user/item preference values, over the first item/user
   * @param sumY2
   *          sum of the square of the user/item preference values, over the second item/user
   * @param sumXYdiff2
   *          sum of squares of differences in X and Y values
   * @return similarity value between -1.0 and 1.0, inclusive, or {@link Double#NaN} if no similarity can be
   *         computed (e.g. when no items have been rated by both users
   */
  abstract double computeResult(int n, double sumXY, double sumX2, double sumY2, double sumXYdiff2);
  
  /**
   * 计算两个user对应相同的item进行打分
   */
  @Override
  public double userSimilarity(long userID1, long userID2) throws TasteException {
    DataModel dataModel = getDataModel();
    PreferenceArray xPrefs = dataModel.getPreferencesFromUser(userID1);//返回该userid对应的一组user-item-value集合,并且按照item排序好了
    PreferenceArray yPrefs = dataModel.getPreferencesFromUser(userID2);//返回该userid对应的一组user-item-value集合,并且按照item排序好了
    int xLength = xPrefs.length();//user1一共对应多少个商品
    int yLength = yPrefs.length();//user2一共对应多少个商品
    
    if (xLength == 0 || yLength == 0) {
      return Double.NaN;
    }
    
    long xIndex = xPrefs.getItemID(0);//user1对应的item
    long yIndex = yPrefs.getItemID(0);//user2对应的item
    int xPrefIndex = 0;//移动user1的item集合到哪个位置了
    int yPrefIndex = 0;
    
    /**
     * eg:
     * user1 item1 3
     * user1 item2 4
     * user1 item3 5
     * 
     * user2 item1 6
     * user2 item2 7
     * user3 item3 8
     */
    double sumX = 0.0;//user1所有item的打分之和 , 即3+4+5
    double sumX2 = 0.0;//user1每一个item打分的平方和,即 9+16+25
    
    double sumY = 0.0;//user2所有item的打分之和 , 即6+7+8
    double sumY2 = 0.0;//user2每一个item打分的平方和,即 36+49+64
    
    double sumXY = 0.0;//user1、user2对于同一个商品的打分乘积之和,即3*6+4*7+5*8
    
    double sumXYdiff2 = 0.0;//user1与user2对于同一个商品的差的平方和,即 (3-6)(3-6)+(4-7)(4-7)+(5-8)(5-8)
    
    int count = 0;//一共做了多少次计算
    
    boolean hasInferrer = inferrer != null;//true表示是否可以推测 两个item之间的相似度
    
    while (true) {
      int compare = xIndex < yIndex ? -1 : xIndex > yIndex ? 1 : 0;
      //只是对两个用户都包含的item进行计算
      //如果可以推断,则将user1、user2所有商品进行两两打分
      if (hasInferrer || compare == 0) {//compare == 0 说明两个item相同,即两个userid关注相同的item  hasInferrer 表示可以对user-item进行推测偏爱度
        double x;//user1对该item的分数
        double y;//user2对该item的分数
        if (xIndex == yIndex) {//说明两个item相同,即两个userid关注相同的item
          // Both users expressed a preference for the item
          //分别获取两个用户对同一商品的分数
          x = xPrefs.getValue(xPrefIndex);
          y = yPrefs.getValue(yPrefIndex);
        } else {
          // Only one user expressed a preference, but infer the other one's preference and tally
          // as if the other user expressed that preference
          //仅仅在一个user表达了对该商品的偏爱度,但是推测另一个user的偏爱度,并且符合另外一个user要表达的偏爱度
          if (compare < 0) {//说明yIndex>xIndex,即y没有x这个item的偏爱记录,因此说明user2没有该商品偏爱程度值
            // X has a value; infer Y's即user1有一个偏爱度,user2要去推测
            x = xPrefs.getValue(xPrefIndex);//x对应得知
            y = inferrer.inferPreference(userID2, xIndex);//y是推测出来的,即推测user2对user1对应的itemid进行推测
          } else {
            // compare > 0
            // Y has a value; infer X's 说明user1没有item,user2有,因此要推测user1
            x = inferrer.inferPreference(userID1, yIndex);
            y = yPrefs.getValue(yPrefIndex);
          }
        }
        sumXY += x * y;
        sumX += x;
        sumX2 += x * x;
        sumY += y;
        sumY2 += y * y;
        double diff = x - y;
        sumXYdiff2 += diff * diff;
        
        count++;
      }
      if (compare <= 0) {//说明user2对item有缺失,则移动x位置
        if (++xPrefIndex >= xLength) {//移动user对应的item位置,说明已经是最后一个item了
          if (hasInferrer) {//如果要推断,则说明要把未来没匹配上的item都要进行匹配
            // Must count other Ys; pretend next X is far away
            if (yIndex == Long.MAX_VALUE) {//如果y的位置也是最大值,说明y早就到最后了,因此跳出循环
              // ... but stop if both are done!
              break;
            }
            xIndex = Long.MAX_VALUE;//设置x的位置是最大值
          } else {
            break;
          }
        } else {//切换下一个item的id
          xIndex = xPrefs.getItemID(xPrefIndex);
        }
      }
      if (compare >= 0) {//说明user1对item缺失
        if (++yPrefIndex >= yLength) {
          if (hasInferrer) {
            // Must count other Xs; pretend next Y is far away
            if (xIndex == Long.MAX_VALUE) {
              // ... but stop if both are done!
              break;
            }
            yIndex = Long.MAX_VALUE;
          } else {
            break;
          }
        } else {
          yIndex = yPrefs.getItemID(yPrefIndex);
        }
      }
    }
    
    // "Center" the data. If my math is correct, this'll do it.
    double result;
    if (centerData) {
      double meanX = sumX / count;
      double meanY = sumY / count;
      // double centeredSumXY = sumXY - meanY * sumX - meanX * sumY + n * meanX * meanY;
      double centeredSumXY = sumXY - meanY * sumX;
      // double centeredSumX2 = sumX2 - 2.0 * meanX * sumX + n * meanX * meanX;
      double centeredSumX2 = sumX2 - meanX * sumX;
      // double centeredSumY2 = sumY2 - 2.0 * meanY * sumY + n * meanY * meanY;
      double centeredSumY2 = sumY2 - meanY * sumY;
      result = computeResult(count, centeredSumXY, centeredSumX2, centeredSumY2, sumXYdiff2);
    } else {
      result = computeResult(count, sumXY, sumX2, sumY2, sumXYdiff2);
    }
    
    if (!Double.isNaN(result)) {
      result = normalizeWeightResult(result, count, cachedNumItems);
    }
    return result;
  }
  
  /**
   * 计算两个item对应相同的user进行打分
   */
  @Override
  public final double itemSimilarity(long itemID1, long itemID2) throws TasteException {
    DataModel dataModel = getDataModel();
    PreferenceArray xPrefs = dataModel.getPreferencesForItem(itemID1);//返回该itemid对应的一组user-item-value集合,并且按照userid排序好了
    PreferenceArray yPrefs = dataModel.getPreferencesForItem(itemID2);//返回该itemid对应的一组user-item-value集合,并且按照userid排序好了
    int xLength = xPrefs.length();
    int yLength = yPrefs.length();
    
    if (xLength == 0 || yLength == 0) {
      return Double.NaN;
    }
    
    long xIndex = xPrefs.getUserID(0);
    long yIndex = yPrefs.getUserID(0);
    int xPrefIndex = 0;
    int yPrefIndex = 0;
    
    double sumX = 0.0;
    double sumX2 = 0.0;
    double sumY = 0.0;
    double sumY2 = 0.0;
    double sumXY = 0.0;
    double sumXYdiff2 = 0.0;
    int count = 0;
    
    // No, pref inferrers and transforms don't apply here. I think.
    
    while (true) {
      int compare = xIndex < yIndex ? -1 : xIndex > yIndex ? 1 : 0;
      if (compare == 0) {
        // Both users expressed a preference for the item
        double x = xPrefs.getValue(xPrefIndex);
        double y = yPrefs.getValue(yPrefIndex);
        sumXY += x * y;
        sumX += x;
        sumX2 += x * x;
        sumY += y;
        sumY2 += y * y;
        double diff = x - y;
        sumXYdiff2 += diff * diff;
        count++;
      }
      if (compare <= 0) {
        if (++xPrefIndex == xLength) {
          break;
        }
        xIndex = xPrefs.getUserID(xPrefIndex);
      }
      if (compare >= 0) {
        if (++yPrefIndex == yLength) {
          break;
        }
        yIndex = yPrefs.getUserID(yPrefIndex);
      }
    }

    double result;
    if (centerData) {
      // See comments above on these computations
      double n = (double) count;
      double meanX = sumX / n;
      double meanY = sumY / n;
      // double centeredSumXY = sumXY - meanY * sumX - meanX * sumY + n * meanX * meanY;
      double centeredSumXY = sumXY - meanY * sumX;
      // double centeredSumX2 = sumX2 - 2.0 * meanX * sumX + n * meanX * meanX;
      double centeredSumX2 = sumX2 - meanX * sumX;
      // double centeredSumY2 = sumY2 - 2.0 * meanY * sumY + n * meanY * meanY;
      double centeredSumY2 = sumY2 - meanY * sumY;
      result = computeResult(count, centeredSumXY, centeredSumX2, centeredSumY2, sumXYdiff2);
    } else {
      result = computeResult(count, sumXY, sumX2, sumY2, sumXYdiff2);
    }
    
    if (!Double.isNaN(result)) {
      result = normalizeWeightResult(result, count, cachedNumUsers);
    }
    return result;
  }

  @Override
  public double[] itemSimilarities(long itemID1, long[] itemID2s) throws TasteException {
    int length = itemID2s.length;
    double[] result = new double[length];
    for (int i = 0; i < length; i++) {
      result[i] = itemSimilarity(itemID1, itemID2s[i]);
    }
    return result;
  }
  
  /**
   * 对结果进行归一化操作
   * @param result 最终分数
   * @param count 处理了多少个item
   * @param num 总共多少个item
   * @return 确保最终结果是[-1.0, 1.0]
   */
  final double normalizeWeightResult(double result, int count, int num) {
    double normalizedResult = result;
    if (weighted) {
      double scaleFactor = 1.0 - (double) count / (double) (num + 1);
      if (normalizedResult < 0.0) {
        normalizedResult = -1.0 + scaleFactor * (1.0 + normalizedResult);
      } else {
        normalizedResult = 1.0 - scaleFactor * (1.0 - normalizedResult);
      }
    }
    //确保最终结果是[-1.0, 1.0]
    // Make sure the result is not accidentally a little outside [-1.0, 1.0] due to rounding:
    if (normalizedResult < -1.0) {
      normalizedResult = -1.0;
    } else if (normalizedResult > 1.0) {
      normalizedResult = 1.0;
    }
    return normalizedResult;
  }
  
  @Override
  public final void refresh(Collection<Refreshable> alreadyRefreshed) {
    super.refresh(alreadyRefreshed);
    refreshHelper.refresh(alreadyRefreshed);
  }
  
  @Override
  public final String toString() {
    return this.getClass().getSimpleName() + "[dataModel:" + getDataModel() + ",inferrer:" + inferrer + ']';
  }
  
}
