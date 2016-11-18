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

package org.apache.mahout.cf.taste.impl.common;

import com.google.common.base.Preconditions;
import org.apache.mahout.cf.taste.common.TasteException;

import java.util.Iterator;

/**
 * <p>
 * An efficient Map-like class which caches values for keys. Values are not "put" into a {@link Cache};
 * instead the caller supplies the instance with an implementation of {@link Retriever} which can load the
 * value for a given key.
 * </p>
 * 一个有效率的map类,能够缓存key-value,
 * value不是被put进来的,而是调用者提供实例加载过来的
 * <p>
 * The cache does not support {@code null} keys.
 * 缓存不支持key为null的
 * </p>
 *
 * <p>
 * Thanks to Amila Jayasooriya for helping evaluate performance of the rewrite of this class, as part of a
 * Google Summer of Code 2007 project.
 * </p>
 */
public final class Cache<K,V> implements Retriever<K,V> {

  private static final Object NULL = new Object();
  
  private final FastMap<K,V> cache;//用于存储缓存的数据
  private final Retriever<? super K,? extends V> retriever;//缓存的数据从这里面加载出来的
  
  /**
   * <p>
   * Creates a new cache based on the given {@link Retriever}.
   * </p>
   * 
   * @param retriever
   *          object which can retrieve values for keys
   */
  public Cache(Retriever<? super K,? extends V> retriever) {
    this(retriever, FastMap.NO_MAX_SIZE);
  }
  
  /**
   * <p>
   * Creates a new cache based on the given {@link Retriever} and with given maximum size.
   * </p>
   * 
   * @param retriever
   *          object which can retrieve values for keys
   * @param maxEntries
   *          maximum number of entries the cache will store before evicting some
   */
  public Cache(Retriever<? super K,? extends V> retriever, int maxEntries) {
    Preconditions.checkArgument(retriever != null, "retriever is null");
    Preconditions.checkArgument(maxEntries >= 1, "maxEntries must be at least 1");
    cache = new FastMap<K, V>(11, maxEntries);
    this.retriever = retriever;
  }
  
  /**
   * <p>
   * Returns cached value for a key. If it does not exist, it is loaded using a {@link Retriever}.
   * </p>
   * 
   * @param key
   *          cache key
   * @return value for that key
   * @throws TasteException
   *           if an exception occurs while retrieving a new cached value
   */
  @Override
  public V get(K key) throws TasteException {
    V value;
    synchronized (cache) {
      value = cache.get(key);//如果缓存有数据,则返回
    }
    if (value == null) {//如果缓存没数据,则从retriever中查找,并且添加到缓存中
      return getAndCacheValue(key);
    }
    return value == NULL ? null : value;
  }
  
  /**
   * <p>
   * Uncaches any existing value for a given key.
   * </p>
   * 
   * @param key
   *          cache key
   * 从缓存中删除该元素
   */
  public void remove(K key) {
    synchronized (cache) {
      cache.remove(key);
    }
  }

  /**
   * Clears all cache entries whose key matches the given predicate.
   * 循环所有缓存的key元素,元素调用predicate返回true的都要被删除
   */
  public void removeKeysMatching(MatchPredicate<K> predicate) {
    synchronized (cache) {
      Iterator<K> it = cache.keySet().iterator();
      while (it.hasNext()) {
        K key = it.next();
        if (predicate.matches(key)) {//元素调用predicate返回true的都要被删除
          it.remove();
        }
      }
    }
  }

  /**
   * Clears all cache entries whose value matches the given predicate.
   * 循环所有缓存的value元素,元素调用predicate返回true的都要被删除
   */
  public void removeValueMatching(MatchPredicate<V> predicate) {
    synchronized (cache) {
      Iterator<V> it = cache.values().iterator();
      while (it.hasNext()) {
        V value = it.next();
        if (predicate.matches(value)) {
          it.remove();
        }
      }
    }
  }
  
  /**
   * <p>
   * Clears the cache.
   * </p>
   */
  public void clear() {
    synchronized (cache) {
      cache.clear();
    }
  }
  
  
  //从retriever中获取数据,添加到缓存中
  private V getAndCacheValue(K key) throws TasteException {
    V value = retriever.get(key);
    if (value == null) {
      value = (V) NULL;
    }
    synchronized (cache) {
      cache.put(key, value);
    }
    return value;
  }
  
  @Override
  public String toString() {
    return "Cache[retriever:" + retriever + ']';
  }

  /**
   * Used by {#link #removeKeysMatching(Object)} to decide things that are matching.
   * 用于匹配key或者value,匹配上的就要去被删除掉
   */
  public interface MatchPredicate<T> {
    boolean matches(T thing);
  }
  
}
