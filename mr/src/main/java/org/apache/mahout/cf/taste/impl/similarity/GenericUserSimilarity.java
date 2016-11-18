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
import java.util.Iterator;

import com.google.common.collect.AbstractIterator;
import org.apache.mahout.cf.taste.common.Refreshable;
import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.common.FastByIDMap;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.recommender.TopItems;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.similarity.PreferenceInferrer;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;
import org.apache.mahout.common.RandomUtils;

import com.google.common.base.Preconditions;

public final class GenericUserSimilarity implements UserSimilarity {
  
  //userid-userid-value(两个userid相似的分数)
  //细节:两个userid小的作为key,
  private final FastByIDMap<FastByIDMap<Double>> similarityMaps = new FastByIDMap<>();
  
  /**
   * @param similarities 一组user-user之间的相似分数集合
   */
  public GenericUserSimilarity(Iterable<UserUserSimilarity> similarities) {
    initSimilarityMaps(similarities.iterator());
  }
  
  /**
   * 
   * @param similarities 一组user-user之间的相似分数集合
   * @param maxToKeep 只是保留分数最高的一些集合
   */
  public GenericUserSimilarity(Iterable<UserUserSimilarity> similarities, int maxToKeep) {
    Iterable<UserUserSimilarity> keptSimilarities =
        TopItems.getTopUserUserSimilarities(maxToKeep, similarities.iterator());//计算要保留的集合
    initSimilarityMaps(keptSimilarities.iterator());//初始化内存映射关系
  }
  
  public GenericUserSimilarity(UserSimilarity otherSimilarity, DataModel dataModel) throws TasteException {
    long[] userIDs = longIteratorToList(dataModel.getUserIDs());//返回userid集合的迭代器,支持跳跃若干个对象,并且按照顺序排列好的userid进行迭代
    initSimilarityMaps(new DataModelSimilaritiesIterator(otherSimilarity, userIDs));
  }
  
  /**
   * 
   * @param otherSimilarity 两个user之间如何计算相似度
   * @param dataModel user-item-value的集合
   * @param maxToKeep 最多保持多少个元素在集合中
   * @throws TasteException
   */
  public GenericUserSimilarity(UserSimilarity otherSimilarity,
                               DataModel dataModel,
                               int maxToKeep) throws TasteException {
    long[] userIDs = longIteratorToList(dataModel.getUserIDs());//返回userid集合的迭代器,支持跳跃若干个对象,并且按照顺序排列好的userid进行迭代
    
    //让每每两个user进行关联,去计算他们之间的相似度
    Iterator<UserUserSimilarity> it = new DataModelSimilaritiesIterator(otherSimilarity, userIDs);
    
    Iterable<UserUserSimilarity> keptSimilarities = TopItems.getTopUserUserSimilarities(maxToKeep, it);
    initSimilarityMaps(keptSimilarities.iterator());
  }

  //将迭代器内的long类型的值在内存中转换成数组
  static long[] longIteratorToList(LongPrimitiveIterator iterator) {
    long[] result = new long[5];
    int size = 0;
    while (iterator.hasNext()) {
      if (size == result.length) {//扩容
        long[] newResult = new long[result.length << 1];
        System.arraycopy(result, 0, newResult, 0, result.length);
        result = newResult;
      }
      result[size++] = iterator.next();
    }
    if (size != result.length) {//缩小,让数组正好全部填满数据
      long[] newResult = new long[size];
      System.arraycopy(result, 0, newResult, 0, size);
      result = newResult;
    }
    return result;
  }
  
  /**
   * 参数是一组用户与用户之间相似度分数集合
   * 
   * 作用:向内存中添加user-user-相似度value映射关系
   */
  private void initSimilarityMaps(Iterator<UserUserSimilarity> similarities) {
    while (similarities.hasNext()) {
      UserUserSimilarity uuc = similarities.next();
      long similarityUser1 = uuc.getUserID1();
      long similarityUser2 = uuc.getUserID2();
      if (similarityUser1 != similarityUser2) {//用户不同,则进行计算
        // Order them -- first key should be the "smaller" one
    	  //找到两个user中较小的一个userid
        long user1;
        long user2;
        if (similarityUser1 < similarityUser2) {
          user1 = similarityUser1;
          user2 = similarityUser2;
        } else {
          user1 = similarityUser2;
          user2 = similarityUser1;
        }
        //向内存中添加user-user-相似度value映射关系
        FastByIDMap<Double> map = similarityMaps.get(user1);
        if (map == null) {
          map = new FastByIDMap<>();
          similarityMaps.put(user1, map);
        }
        map.put(user2, uuc.getValue());
      }
      // else similarity between user and itself already assumed to be 1.0
    }
  }
  
  //返回两个用户的相似度,
  @Override
  public double userSimilarity(long userID1, long userID2) {
    if (userID1 == userID2) {//如果两个用户相同.则返回1
      return 1.0;
    }
    
    //将两个用户id排序
    long first;
    long second;
    if (userID1 < userID2) {
      first = userID1;
      second = userID2;
    } else {
      first = userID2;
      second = userID1;
    }
    FastByIDMap<Double> nextMap = similarityMaps.get(first);
    if (nextMap == null) {//说明没有第一个用户相关的信息
      return Double.NaN;
    }
    Double similarity = nextMap.get(second);//如果有值,则返回,如果没有值,则说明两个用户没有相关度
    return similarity == null ? Double.NaN : similarity;
  }
  
  @Override
  public void setPreferenceInferrer(PreferenceInferrer inferrer) {
    throw new UnsupportedOperationException();
  }
  
  @Override
  public void refresh(Collection<Refreshable> alreadyRefreshed) {
  // Do nothing
  }
  
  //表示两个用户以及对应的相似度分数,仅仅按照分数排序,跟userid没关系
  public static final class UserUserSimilarity implements Comparable<UserUserSimilarity> {

    private final long userID1;
    private final long userID2;
    private final double value;
    
    public UserUserSimilarity(long userID1, long userID2, double value) {
      Preconditions.checkArgument(value >= -1.0 && value <= 1.0, "Illegal value: " + value + ". Must be: -1.0 <= value <= 1.0");
      this.userID1 = userID1;
      this.userID2 = userID2;
      this.value = value;
    }
    
    public long getUserID1() {
      return userID1;
    }
    
    public long getUserID2() {
      return userID2;
    }
    
    public double getValue() {
      return value;
    }
    
    @Override
    public String toString() {
      return "UserUserSimilarity[" + userID1 + ',' + userID2 + ':' + value + ']';
    }
    
    /** Defines an ordering from highest similarity to lowest. 
     * 仅仅按照分数排序,跟userid没关系
     **/
    @Override
    public int compareTo(UserUserSimilarity other) {
      double otherValue = other.getValue();
      return value > otherValue ? -1 : value < otherValue ? 1 : 0;
    }
    
    @Override
    public boolean equals(Object other) {
      if (!(other instanceof UserUserSimilarity)) {
        return false;
      }
      UserUserSimilarity otherSimilarity = (UserUserSimilarity) other;
      return otherSimilarity.getUserID1() == userID1
          && otherSimilarity.getUserID2() == userID2
          && otherSimilarity.getValue() == value;
    }
    
    @Override
    public int hashCode() {
      return (int) userID1 ^ (int) userID2 ^ RandomUtils.hashDouble(value);
    }
    
  }
  
  //迭代器,迭代两个user之间的相似度
  private static final class DataModelSimilaritiesIterator extends AbstractIterator<UserUserSimilarity> {

    private final UserSimilarity otherSimilarity;//计算两个user相似度的计算器
    private final long[] itemIDs;//userid集合
    private int i;
    private long itemID1;//当前user集合中计算第几个了
    private int j;

    private DataModelSimilaritiesIterator(UserSimilarity otherSimilarity, long[] itemIDs) {
      this.otherSimilarity = otherSimilarity;
      this.itemIDs = itemIDs;
      i = 0;
      itemID1 = itemIDs[0];//初始化第1个开始计算
      j = 1;
    }

    /**
     * 相当于两层for循环
     * 计算itemIDs数组中两两组合,使用otherSimilarity计算两个userid的相似度
     */
    @Override
    protected UserUserSimilarity computeNext() {
      int size = itemIDs.length;
      while (i < size - 1) {//从i开始循环
        long itemID2 = itemIDs[j];
        double similarity;
        try {
          similarity = otherSimilarity.userSimilarity(itemID1, itemID2);//计算两个相似度
        } catch (TasteException te) {
          // ugly:
          throw new IllegalStateException(te);
        }
        if (!Double.isNaN(similarity)) {//如果相似度不是0,则创建相似度对象
          return new UserUserSimilarity(itemID1, itemID2, similarity);
        }
        if (++j == size) {//每次i都要跟所有的j相关联,当j到最后一个的时候,则i++即可
          itemID1 = itemIDs[++i];
          j = i + 1;
        }
      }
      return endOfData();
    }
    
  }
  
}
