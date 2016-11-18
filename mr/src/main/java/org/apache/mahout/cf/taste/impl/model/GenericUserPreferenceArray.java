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

package org.apache.mahout.cf.taste.impl.model;

import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import com.google.common.base.Function;
import com.google.common.collect.Iterators;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.common.iterator.CountingIterator;

/**
 * <p>
 * Like {@link GenericItemPreferenceArray} but stores preferences for one user (all user IDs the same) rather
 * than one item.
 * </p>
 *
 * <p>
 * This implementation maintains two parallel arrays, of item IDs and values. The idea is to save allocating
 * {@link Preference} objects themselves. This saves the overhead of {@link Preference} objects but also
 * duplicating the user ID value.
 * </p>
 * 
 * @see BooleanUserPreferenceArray
 * @see GenericItemPreferenceArray
 * @see GenericPreference
 * 表示同一个user对应一组item和偏好度value集合
 * 因此该对象就表示一个user下的item和偏好度value集合
 */
public final class GenericUserPreferenceArray implements PreferenceArray {

  private static final int ITEM = 1;//按照itemid的正序排列数组集合
  private static final int VALUE = 2;//按照value的正序排列数组集合
  private static final int VALUE_REVERSED = 3;//按照value的倒序排列数组集合

  private long id;//userId
  private final long[] ids;//itemID对象集合
  private final float[] values;//itemId对象对应的偏好度集合

  public GenericUserPreferenceArray(int size) {
    this.ids = new long[size];
    values = new float[size];
    this.id = Long.MIN_VALUE; // as a sort of 'unspecified' value
  }

  public GenericUserPreferenceArray(List<? extends Preference> prefs) {
    this(prefs.size());
    int size = prefs.size();
    long userID = Long.MIN_VALUE;
    for (int i = 0; i < size; i++) {
      Preference pref = prefs.get(i);
      if (i == 0) {//第一个可以获取userid
        userID = pref.getUserID();
      } else {
        if (userID != pref.getUserID()) {//必须保证每一个userId都是相同的,否则抛出异常
          throw new IllegalArgumentException("Not all user IDs are the same");
        }
      }
      //设置对应的itemId和偏好度value
      ids[i] = pref.getItemID();
      values[i] = pref.getValue();
    }
    id = userID;
  }

  /**
   * This is a private copy constructor for clone().
   */
  private GenericUserPreferenceArray(long[] ids, long id, float[] values) {
	this.id = id;
    this.ids = ids;
    this.values = values;
  }

  //该user有多少个偏好的物品
  @Override
  public int length() {
    return ids.length;
  }

  //设置某一个item对应的Preference对象
  @Override
  public Preference get(int i) {
    return new PreferenceView(i);
  }

  @Override
  public void set(int i, Preference pref) {
    id = pref.getUserID();
    ids[i] = pref.getItemID();
    values[i] = pref.getValue();
  }

  @Override
  public long getUserID(int i) {
    return id;
  }

  /**
   * {@inheritDoc}
   * 
   * Note that this method will actually set the user ID for <em>all</em> preferences.
   */
  @Override
  public void setUserID(int i, long userID) {
    id = userID;
  }

  @Override
  public long getItemID(int i) {
    return ids[i];
  }

  @Override
  public void setItemID(int i, long itemID) {
    ids[i] = itemID;
  }

  /**
   * @return all item IDs
   */
  @Override
  public long[] getIDs() {
    return ids;
  }

  @Override
  public float getValue(int i) {
    return values[i];
  }

  //根据userid进行对数组排序,都是正序排列
  @Override
  public void setValue(int i, float value) {
    values[i] = value;
  }

  @Override
  public void sortByUser() { }

  //根据itemid进行对数组排序,都是正序排列
  @Override
  public void sortByItem() {
    lateralSort(ITEM);
  }

  //根据偏好度value进行对数组排序,都是正序排列
  @Override
  public void sortByValue() {
    lateralSort(VALUE);
  }

  //倒序排列
  @Override
  public void sortByValueReversed() {
    lateralSort(VALUE_REVERSED);
  }

  //判断该userid是否在数组内
  @Override
  public boolean hasPrefWithUserID(long userID) {
    return id == userID;
  }

  //判断item是否在数组内
  @Override
  public boolean hasPrefWithItemID(long itemID) {
    for (long id : ids) {
      if (itemID == id) {
        return true;
      }
    }
    return false;
  }

  //排序
  private void lateralSort(int type) {
    //Comb sort: http://en.wikipedia.org/wiki/Comb_sort
    int length = length();
    int gap = length;
    boolean swapped = false;
    while (gap > 1 || swapped) {
      if (gap > 1) {
        gap /= 1.247330950103979; // = 1 / (1 - 1/e^phi)
      }
      swapped = false;
      int max = length - gap;
      for (int i = 0; i < max; i++) {
        int other = i + gap;
        if (isLess(other, i, type)) {
          swap(i, other);
          swapped = true;
        }
      }
    }
  }

  private boolean isLess(int i, int j, int type) {
    switch (type) {
      case ITEM:
        return ids[i] < ids[j];
      case VALUE:
        return values[i] < values[j];
      case VALUE_REVERSED:
        return values[i] > values[j];
      default:
        throw new IllegalStateException();
    }
  }

  private void swap(int i, int j) {
    long temp1 = ids[i];
    float temp2 = values[i];
    ids[i] = ids[j];
    values[i] = values[j];
    ids[j] = temp1;
    values[j] = temp2;
  }

  @Override
  public GenericUserPreferenceArray clone() {
    return new GenericUserPreferenceArray(ids.clone(), id, values.clone());
  }

  @Override
  public int hashCode() {
    return (int) (id >> 32) ^ (int) id ^ Arrays.hashCode(ids) ^ Arrays.hashCode(values);
  }

  @Override
  public boolean equals(Object other) {
    if (!(other instanceof GenericUserPreferenceArray)) {
      return false;
    }
    GenericUserPreferenceArray otherArray = (GenericUserPreferenceArray) other;
    return id == otherArray.id && Arrays.equals(ids, otherArray.ids) && Arrays.equals(values, otherArray.values);
  }

  //迭代每一个元素
  @Override
  public Iterator<Preference> iterator() {
    return Iterators.transform(new CountingIterator(length()),
      new Function<Integer, Preference>() {
        @Override
        public Preference apply(Integer from) {
          return new PreferenceView(from);
        }
      });
  }

  @Override
  public String toString() {
    if (ids == null || ids.length == 0) {
      return "GenericUserPreferenceArray[{}]";
    }
    StringBuilder result = new StringBuilder(20 * ids.length);
    result.append("GenericUserPreferenceArray[userID:");
    result.append(id);
    result.append(",{");
    for (int i = 0; i < ids.length; i++) {
      if (i > 0) {
        result.append(',');
      }
      result.append(ids[i]);
      result.append('=');
      result.append(values[i]);
    }
    result.append("}]");
    return result.toString();
  }

  //返回一个视图,对应一个Preference对象
  private final class PreferenceView implements Preference {

    private final int i;//获取第几个item对象

    private PreferenceView(int i) {
      this.i = i;
    }

    //userid都是相同的
    @Override
    public long getUserID() {
      return GenericUserPreferenceArray.this.getUserID(i);
    }

    @Override
    public long getItemID() {
      return GenericUserPreferenceArray.this.getItemID(i);
    }

    @Override
    public float getValue() {
      return values[i];
    }

    //设置该item对应的偏好度
    @Override
    public void setValue(float value) {
      values[i] = value;
    }

  }

}
