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

package org.apache.mahout.math;

import java.util.Arrays;
import java.util.Iterator;
import java.util.NoSuchElementException;

import com.google.common.base.Preconditions;

/** Implements vector as an array of doubles 
 * DenseVector:密集类型的向量,即用一个数组将向量里面的所有元素都记录
 * RandomAccessSparseVector:基于浮点数的 HashMap 实现的，key 是整形 (int) 类型，value 是浮点数 (double) 类型，它只存储向量中不为空的值，并提供随机访问。
 * SequentialAccessSparseVector：实现为整形 (int) 类型和浮点数 (double) 类型的并行数组，它也只存储向量中不为空的值，但只提供顺序访问。
 **/
public class DenseVector extends AbstractVector {

  private double[] values;//密集向量用一组double表示

  /** For serialization purposes only */
  public DenseVector() {
    super(0);//设置向量默认值
  }

  /** Construct a new instance using provided values
   *  @param values - array of values
   */
  public DenseVector(double[] values) {
    this(values, false);
  }

  public DenseVector(double[] values, boolean shallowCopy) {
    super(values.length);
    this.values = shallowCopy ? values : values.clone();
  }

  public DenseVector(DenseVector values, boolean shallowCopy) {
    this(values.values, shallowCopy);
  }

  /** Construct a new instance of the given cardinality
   * @param cardinality - number of values in the vector 定向量的size
   */
  public DenseVector(int cardinality) {
    super(cardinality);
    this.values = new double[cardinality];
  }

  /**
   * Copy-constructor (for use in turning a sparse vector into a dense one, for example)
   * @param vector The vector to copy
   */
  public DenseVector(Vector vector) {
    super(vector.size());
    values = new double[vector.size()];
    for (Element e : vector.nonZeroes()) {
      values[e.index()] = e.get();
    }
  }

  //计算点积
  @Override
  public double dot(Vector x) {
    if (!x.isDense()) {//如果不是密集的,就没有一个数组组成,因此去计算两个向量分别下标相乘
      return super.dot(x);
    } else {//说明都是密集向量

      int size = x.size();
      if (values.length != size) {
        throw new CardinalityException(values.length, size);
      }

      double sum = 0;
      for (int n = 0; n < size; n++) {
        sum += values[n] * x.getQuick(n);//获取每一个相应下标乘积之和
      }
      return sum;
    }
  }

  @Override
  protected Matrix matrixLike(int rows, int columns) {
    return new DenseMatrix(rows, columns);
  }

  @SuppressWarnings("CloneDoesntCallSuperClone")
  @Override
  public DenseVector clone() {
    return new DenseVector(values.clone());
  }

  /**
   * @return true
   * 是密集向量,因此返回true
   */
  @Override
  public boolean isDense() {
    return true;
  }

  /**
   * @return true
   * 可以顺序访问该向量
   */
  @Override
  public boolean isSequentialAccess() {
    return true;
  }

  //自己乘以自己
  @Override
  protected double dotSelf() {
    double result = 0.0;
    int max = size();
    for (int i = 0; i < max; i++) {
      result += values[i] * values[i];
    }
    return result;
  }

  @Override
  public double getQuick(int index) {
    return values[index];
  }

  @Override
  public DenseVector like() {
    return new DenseVector(size());
  }

  @Override
  public Vector like(int cardinality) {
    return new DenseVector(cardinality);
  }

  @Override
  public void setQuick(int index, double value) {
    invalidateCachedLength();
    values[index] = value;
  }

  @Override
  public void incrementQuick(int index, double increment) {
    invalidateCachedLength();
    values[index] += increment;
  }

  //填充该值
  @Override
  public Vector assign(double value) {
    invalidateCachedLength();
    Arrays.fill(values, value);
    return this;
  }

  @Override
  public int getNumNondefaultElements() {
    return values.length;
  }

  //非0的元素数量
  @Override
  public int getNumNonZeroElements() {
    int numNonZeros = 0;
    for (int index = 0; index < values.length; index++) {
      if (values[index] != 0) {
        numNonZeros++;
      }
    }
    return numNonZeros;
  }

  public Vector assign(DenseVector vector) {
    // make sure the data field has the correct length
    if (vector.values.length != this.values.length) {
      this.values = new double[vector.values.length];
    }
    // now copy the values
    System.arraycopy(vector.values, 0, this.values, 0, this.values.length);
    return this;
  }

  @Override
  public void mergeUpdates(OrderedIntDoubleMapping updates) {
    int numUpdates = updates.getNumMappings();
    int[] indices = updates.getIndices();
    double[] values = updates.getValues();
    for (int i = 0; i < numUpdates; ++i) {
      this.values[indices[i]] = values[i];
    }
  }

  @Override
  public Vector viewPart(int offset, int length) {
    if (offset < 0) {
      throw new IndexException(offset, size());
    }
    if (offset + length > size()) {
      throw new IndexException(offset + length, size());
    }
    return new DenseVectorView(this, offset, length);
  }

  //去估计一个价值---获取一个随机元素的时间
  @Override
  public double getLookupCost() {
    return 1;
  }

  //去估计一个操作的价值--去迭代非0元素的时间
  @Override
  public double getIteratorAdvanceCost() {
    return 1;
  }

  //添加一个非0元素的时间是否是常量时间,true表示常量时间
  @Override
  public boolean isAddConstantTime() {
    return true;
  }

  /**
   * Returns an iterator that traverses this Vector from 0 to cardinality-1, in that order.
   */
  @Override
  public Iterator<Element> iterateNonZero() {
    return new NonDefaultIterator();
  }

  @Override
  public Iterator<Element> iterator() {
    return new AllIterator();
  }

  @Override
  public boolean equals(Object o) {
    if (o instanceof DenseVector) {
      // Speedup for DenseVectors
      return Arrays.equals(values, ((DenseVector) o).values);
    }
    return super.equals(o);
  }

  //每一个元素的值与参数v对应的值相加
  public void addAll(Vector v) {
    if (size() != v.size()) {
      throw new CardinalityException(size(), v.size());
    }

    for (Element element : v.nonZeroes()) {
      values[element.index()] += element.get();
    }
  }

  //迭代非0元素
  private final class NonDefaultIterator implements Iterator<Element> {
    private final DenseElement element = new DenseElement();
    private int index = -1;//已经读取的位置
    private int lookAheadIndex = -1;//下次要读取的位置

    //可以调用N次hasNext也没问题,因为没有改变index
    @Override
    public boolean hasNext() {
      if (lookAheadIndex == index) {  // User calls hasNext() after a next()
        lookAhead();
      } // else user called hasNext() repeatedly.
      return lookAheadIndex < size();
    }

    private void lookAhead() {
      lookAheadIndex++;
      while (lookAheadIndex < size() && values[lookAheadIndex] == 0.0) {//只要没有到最后,就一直查找.找到不是0的为止
        lookAheadIndex++;//每次累加1
      }
    }

    @Override
    public Element next() {
      if (lookAheadIndex == index) { // If user called next() without checking hasNext().
        lookAhead();
      }

      Preconditions.checkState(lookAheadIndex > index);
      index = lookAheadIndex;

      if (index >= size()) { // If the end is reached.
        throw new NoSuchElementException();
      }

      element.index = index;
      return element;
    }

    @Override
    public void remove() {
      throw new UnsupportedOperationException();
    }
  }

  //迭代所有元素
  private final class AllIterator implements Iterator<Element> {
    private final DenseElement element = new DenseElement();

    private AllIterator() {
      element.index = -1;
    }

    @Override
    public boolean hasNext() {
      return element.index + 1 < size();
    }

    @Override
    public Element next() {
      if (element.index + 1 >= size()) { // If the end is reached.
        throw new NoSuchElementException();
      }
      element.index++;
      return element;
    }

    @Override
    public void remove() {
      throw new UnsupportedOperationException();
    }
  }

  //表示一个元素
  private final class DenseElement implements Element {
    int index;

    @Override
    public double get() {
      return values[index];
    }

    @Override
    public int index() {
      return index;
    }

    @Override
    public void set(double value) {
      invalidateCachedLength();
      values[index] = value;
    }
  }

  private final class DenseVectorView extends VectorView {

    public DenseVectorView(Vector vector, int offset, int cardinality) {
      super(vector, offset, cardinality);
    }

    @Override
    public double dot(Vector x) {

      // Apply custom dot kernels for pairs of dense vectors or their views to reduce
      // view indirection.
      if (x instanceof DenseVectorView) {

        if (size() != x.size())
          throw new IllegalArgumentException("Cardinality mismatch during dot(x,y).");

        DenseVectorView xv = (DenseVectorView) x;
        double[] thisValues = ((DenseVector) vector).values;
        double[] thatValues = ((DenseVector) xv.vector).values;
        int untilOffset = offset + size();

        int i, j;
        double sum = 0.0;

        // Provoking SSE
        int until4 = offset + (size() & ~3);
        for (
          i = offset, j = xv.offset;
          i < until4;
          i += 4, j += 4
          ) {
          sum += thisValues[i] * thatValues[j] +
            thisValues[i + 1] * thatValues[j + 1] +
            thisValues[i + 2] * thatValues[j + 2] +
            thisValues[i + 3] * thatValues[j + 3];
        }

        // Picking up the slack
        for (
          i = offset, j = xv.offset;
          i < untilOffset;
          ) {
          sum += thisValues[i++] * thatValues[j++];
        }
        return sum;

      } else if (x instanceof DenseVector ) {

        if (size() != x.size())
          throw new IllegalArgumentException("Cardinality mismatch during dot(x,y).");

        DenseVector xv = (DenseVector) x;
        double[] thisValues = ((DenseVector) vector).values;
        double[] thatValues = xv.values;
        int untilOffset = offset + size();

        int i, j;
        double sum = 0.0;

        // Provoking SSE
        int until4 = offset + (size() & ~3);
        for (
          i = offset, j = 0;
          i < until4;
          i += 4, j += 4
          ) {
          sum += thisValues[i] * thatValues[j] +
            thisValues[i + 1] * thatValues[j + 1] +
            thisValues[i + 2] * thatValues[j + 2] +
            thisValues[i + 3] * thatValues[j + 3];
        }

        // Picking up slack
        for ( ;
          i < untilOffset;
          ) {
          sum += thisValues[i++] * thatValues[j++];
        }
        return sum;

      } else {
        return super.dot(x);
      }
    }

    @Override
    public Vector viewPart(int offset, int length) {
      if (offset < 0) {
        throw new IndexException(offset, size());
      }
      if (offset + length > size()) {
        throw new IndexException(offset + length, size());
      }
      return new DenseVectorView(vector, offset + this.offset, length);
    }
  }
}
