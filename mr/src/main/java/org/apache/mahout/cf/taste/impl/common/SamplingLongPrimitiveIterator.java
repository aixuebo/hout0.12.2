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

import java.util.NoSuchElementException;

import com.google.common.base.Preconditions;
import org.apache.commons.math3.distribution.PascalDistribution;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.RandomWrapper;

/**
 * Wraps a {@link LongPrimitiveIterator} and returns only some subset of the elements that it would,
 * as determined by a sampling rate parameter.
 * 抽样的方式从真正的迭代器中抽取数字
 */
public final class SamplingLongPrimitiveIterator extends AbstractLongPrimitiveIterator {
  
  private final PascalDistribution geometricDistribution;//生成随机数,大概在0-5之间的随机数吧,总之生成随机数
  private final LongPrimitiveIterator delegate;//真正的可以迭代long类型的迭代器
  private long next;//下一个内容
  private boolean hasNext;//是否有下一个
  
  public SamplingLongPrimitiveIterator(LongPrimitiveIterator delegate, double samplingRate) {
    this(RandomUtils.getRandom(), delegate, samplingRate);
  }

  /**
   * @param random
   * @param delegate
   * @param samplingRate 不能大于1,越小,产生的随机数越大,越大,基本上不会产生随机数
   */
  public SamplingLongPrimitiveIterator(RandomWrapper random, LongPrimitiveIterator delegate, double samplingRate) {
    Preconditions.checkNotNull(delegate);
    Preconditions.checkArgument(samplingRate > 0.0 && samplingRate <= 1.0, "Must be: 0.0 < samplingRate <= 1.0");
    // Geometric distribution is special case of negative binomial (aka Pascal) with r=1:
    geometricDistribution = new PascalDistribution(random.getRandomGenerator(), 1, samplingRate);
    this.delegate = delegate;
    this.hasNext = true;
    doNext();
  }
  
  @Override
  public boolean hasNext() {
    return hasNext;
  }
  
  @Override
  public long nextLong() {
    if (hasNext) {
      long result = next;//这个就是最终下一个的值
      doNext();//进行迭代下一个元素
      return result;
    }
    throw new NoSuchElementException();
  }
  
  @Override
  public long peek() {//只是查看下一个元素内容,不会移动指针
    if (hasNext) {
      return next;
    }
    throw new NoSuchElementException();
  }
  
  private void doNext() {
    int toSkip = geometricDistribution.sample();//产生一个随机数
    delegate.skip(toSkip);//跳过随机数个元素
    if (delegate.hasNext()) {//如果还有next,则下一个就是抽样的结果
      next = delegate.next();
    } else {
      hasNext = false;//说明没有数据了
    }
  }
  
  /**
   * @throws UnsupportedOperationException
   */
  @Override
  public void remove() {
    throw new UnsupportedOperationException();
  }
  
  @Override
  public void skip(int n) {//跳过N次
    int toSkip = 0;
    for (int i = 0; i < n; i++) {//每次都跳过若干个随机数
      toSkip += geometricDistribution.sample();
    }
    delegate.skip(toSkip);//整体跳过若干个
    if (delegate.hasNext()) {//查看是否还有数据
      next = delegate.next();
    } else {
      hasNext = false;
    }
  }
  
  //samplingRate不能大于1,越小,产生的随机数越大,越大,基本上不会产生随机数
  public static LongPrimitiveIterator maybeWrapIterator(LongPrimitiveIterator delegate, double samplingRate) {
    return samplingRate >= 1.0 ? delegate : new SamplingLongPrimitiveIterator(delegate, samplingRate);
  }
  
  public static void main(String[] args) {
	
	  RandomWrapper random = RandomUtils.getRandom();
	  double samplingRate = 0.1;
	  
	  PascalDistribution geometricDistribution = new PascalDistribution(random.getRandomGenerator(), 1, samplingRate);
	
	  for(int i=0 ; i<100;i++){
		  System.out.println(geometricDistribution.sample());
	  }
	  
  }
}
