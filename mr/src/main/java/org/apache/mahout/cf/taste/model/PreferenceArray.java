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

package org.apache.mahout.cf.taste.model;

import java.io.Serializable;

/**
 * An alternate representation of an array of {@link Preference}. Implementations, in theory, can produce a
 * more memory-efficient representation.
 * 表示一组user-item-偏好度集合
 */
public interface PreferenceArray extends Cloneable, Serializable, Iterable<Preference> {
  
  /**
   * @return size of length of the "array"
   * 集合有多少个元素
   */
  int length();
  
  /**
   * @param i
   *          index
   * @return a materialized {@link Preference} representation of the preference at i
   * 返回某一个元素
   */
  Preference get(int i);
  
  /**
   * Sets preference at i from information in the given {@link Preference}
   * 
   * @param i
   * @param pref
   * 设置某一个元素
   */
  void set(int i, Preference pref);
  
  /**
   * @param i
   *          index
   * @return user ID from preference at i
   * 获取第i个元素对应的userid
   */
  long getUserID(int i);
  
  /**
   * Sets user ID for preference at i.
   * 
   * @param i
   *          index
   * @param userID
   *          new user ID
   */
  void setUserID(int i, long userID);
  
  /**
   * @param i
   *          index
   * @return item ID from preference at i
   * 获取第i个元素对应的itemid
   */
  long getItemID(int i);
  
  /**
   * Sets item ID for preference at i.
   * 
   * @param i
   *          index
   * @param itemID
   *          new item ID
   */
  void setItemID(int i, long itemID);

  /**
   * @return all user or item IDs
   * 获取所有的userid或者itemid集合,这个取决于当前是userid对应itemid集合，还是itemid对应userid集合
   */
  long[] getIDs();
  
  /**
   * @param i
   *          index
   * @return preference value from preference at i
   * 获取第i个对应的偏好度
   */
  float getValue(int i);
  
  /**
   * Sets preference value for preference at i.
   * 
   * @param i
   *          index
   * @param value
   *          new preference value
   */
  void setValue(int i, float value);
  
  /**
   * @return independent copy of this object
   */
  PreferenceArray clone();
  
  /**
   * Sorts underlying array by user ID, ascending.
   * 根据userid进行对数组排序,都是正序排列
   */
  void sortByUser();
  
  /**
   * Sorts underlying array by item ID, ascending.
   * 根据itemid进行对数组排序,都是正序排列
   */
  void sortByItem();
  
  /**
   * Sorts underlying array by preference value, ascending.
   * 根据偏好度value进行对数组排序,都是正序排列
   */
  void sortByValue();
  
  /**
   * Sorts underlying array by preference value, descending.
   * 倒序排列
   */
  void sortByValueReversed();
  
  /**
   * @param userID
   *          user ID
   * @return true if array contains a preference with given user ID
   * 判断该userid是否在数组内
   */
  boolean hasPrefWithUserID(long userID);
  
  /**
   * @param itemID
   *          item ID
   * @return true if array contains a preference with given item ID
   * 判断item是否在数组内
   */
  boolean hasPrefWithItemID(long itemID);
  
}
