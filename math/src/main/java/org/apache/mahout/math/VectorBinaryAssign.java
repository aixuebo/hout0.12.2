/*
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

import org.apache.mahout.math.Vector.Element;
import org.apache.mahout.math.function.DoubleDoubleFunction;
import org.apache.mahout.math.set.OpenIntHashSet;

import java.util.Iterator;

/**
 * Abstract class encapsulating different algorithms that perform the Vector operations assign().
 * x.assign(y, f), for x and y Vectors and f a DoubleDouble function:
 * - applies the function f to every element in x and y, f(xi, yi)
 * - assigns xi = f(xi, yi) for all indices i
 *
 * The names of variables, methods and classes used here follow the following conventions:
 * The vector being assigned to (the left hand side) is called this or x.
 * The right hand side is called that or y.
 * The function to be applied is called f.
 *
 * The different algorithms take into account the different characteristics of vector classes:
 * - whether the vectors support sequential iteration (isSequential())
 * - whether the vectors support constant-time additions (isAddConstantTime())
 * - what the lookup cost is (getLookupCost())
 * - what the iterator advancement cost is (getIteratorAdvanceCost())
 *
 * The names of the actual classes (they're nested in VectorBinaryAssign) describe the used for assignment.
 * The most important optimization is iterating just through the nonzeros (only possible if f(0, 0) = 0).
 * There are 4 main possibilities:
 * - iterating through the nonzeros of just one vector and looking up the corresponding elements in the other
 * - iterating through the intersection of nonzeros (those indices where both vectors have nonzero values)
 * - iterating through the union of nonzeros (those indices where at least one of the vectors has a nonzero value)
 * - iterating through all the elements in some way (either through both at the same time, both one after the other,
 *   looking up both, looking up just one).
 * Then, there are two additional sub-possibilities:
 * - if a new value can be added to x in constant time (isAddConstantTime()), the *Inplace updates are used
 * - otherwise (really just for SequentialAccessSparseVectors right now), the *Merge updates are used, where
 *   a sorted list of (index, value) pairs is merged into the vector at the end.
 *
 * The internal details are not important and a particular algorithm should generally not be called explicitly.
 * The best one will be selected through assignBest(), which is itself called through Vector.assign().
 *
 * See https://docs.google.com/document/d/1g1PjUuvjyh2LBdq2_rKLIcUiDbeOORA1sCJiSsz-JVU/edit# for a more detailed
 * explanation.
 *
 * 根据f函数的特性,选择一个最优秀的方式去计算  x和y向量的每一个对应的值,当做参数传入到f中,结果改变x向量的每一个元素的值
 */
public abstract class VectorBinaryAssign {
  public static final VectorBinaryAssign[] OPERATIONS = {
    new AssignNonzerosIterateThisLookupThat(),
    new AssignNonzerosIterateThatLookupThisMergeUpdates(),
    new AssignNonzerosIterateThatLookupThisInplaceUpdates(),

    new AssignIterateIntersection(),

    new AssignIterateUnionSequentialMergeUpdates(),
    new AssignIterateUnionSequentialInplaceUpdates(),
    new AssignIterateUnionRandomMergeUpdates(),
    new AssignIterateUnionRandomInplaceUpdates(),

    new AssignAllIterateSequentialMergeUpdates(),
    new AssignAllIterateSequentialInplaceUpdates(),
    new AssignAllIterateThisLookupThatMergeUpdates(),
    new AssignAllIterateThisLookupThatInplaceUpdates(),
    new AssignAllIterateThatLookupThisMergeUpdates(),
    new AssignAllIterateThatLookupThisInplaceUpdates(),
    new AssignAllLoopMergeUpdates(),
    new AssignAllLoopInplaceUpdates(),
  };

  /**
   * Returns true iff we can use this algorithm to apply f to x and y component-wise and assign the result to x.
   * 返回该函数是否有效应对x和y这两个向量
   */
  public abstract boolean isValid(Vector x, Vector y, DoubleDoubleFunction f);

  /**
   * Estimates the cost of using this algorithm to compute the assignment. The algorithm is assumed to be valid.
   * 如果该函数f有效应对,则估算一下计算量的价值
   */
  public abstract double estimateCost(Vector x, Vector y, DoubleDoubleFunction f);

  /**
   * Main method that applies f to x and y component-wise assigning the results to x. It returns the modified vector, x.
   * 真正使用该方法去计算
   * 根据f函数的特性,选择一个最优秀的方式去计算  x和y向量的每一个对应的值,当做参数传入到f中,结果改变x向量的每一个元素的值
   */
  public abstract Vector assign(Vector x, Vector y, DoubleDoubleFunction f);

  /**
   * The best operation is the least expensive valid one.
   * 最好的操作就是最少花钱并且有效的一个方式
   */
  public static VectorBinaryAssign getBestOperation(Vector x, Vector y, DoubleDoubleFunction f) {
    int bestOperationIndex = -1;
    double bestCost = Double.POSITIVE_INFINITY;
    for (int i = 0; i < OPERATIONS.length; ++i) {
      if (OPERATIONS[i].isValid(x, y, f)) {
        double cost = OPERATIONS[i].estimateCost(x, y, f);
        if (cost < bestCost) {//选择一个最有价值的,即cost最小的,选择这个就可以
          bestCost = cost;
          bestOperationIndex = i;
        }
      }
    }
    return OPERATIONS[bestOperationIndex];
  }

  /**
   * This is the method that should be used when assigning. It selects the best algorithm and applies it.
   * Note that it does NOT invalidate the cached length of the Vector and should only be used through the wrapprs
   * in AbstractVector.
   * 选择一个最好的方式去计算结果
   */
  public static Vector assignBest(Vector x, Vector y, DoubleDoubleFunction f) {
    return getBestOperation(x, y, f).assign(x, y, f);
  }

  /**
   * If f(0, y) = 0, the zeros in x don't matter and we can simply iterate through the nonzeros of x.
   * 如果函数f符合无论y是什么,只要x=0,则f结果就是0的时候,我们就很简单的循环x是非0的元素即可
   * To get the corresponding element of y, we perform a lookup.
   * There are no *Merge or *Inplace versions because in this case x cannot become more dense because of f,
   * meaning all changes will occur at indices whose values are already nonzero.
   * 由于函数f的特性,意味着所有的元素的变更将会发生在索引处,索引处的值都已经是非0的元素了,即只会发生在非0元素上才会被更改
   */
  public static class AssignNonzerosIterateThisLookupThat extends VectorBinaryAssign {

    @Override
    public boolean isValid(Vector x, Vector y, DoubleDoubleFunction f) {
      return f.isLikeLeftMult();
    }

    @Override
    public double estimateCost(Vector x, Vector y, DoubleDoubleFunction f) {
      /**
       * 返回非默认值的数量,比如在一个松散向量中,返回的是非0的元素数量
       * 去估计一个操作的价值--去迭代非0元素的时间
       * 去估计一个价值---获取一个随机元素的时间
       */
      return x.getNumNondefaultElements() * x.getIteratorAdvanceCost() * y.getLookupCost();
    }

    @Override
    public Vector assign(Vector x, Vector y, DoubleDoubleFunction f) {
      for (Element xe : x.nonZeroes()) {
        xe.set(f.apply(xe.get(), y.getQuick(xe.index())));//将x和y相对应的数据用函数f处理后,存储到x上
      }
      return x;
    }
  }

  /**
   * If f(x, 0) = x, the zeros in y don't matter and we can simply iterate through the nonzeros of y.
   * y中是0的元素不重要,因为只要y=0,结果一定就是x,因此我们可以通过迭代y的非0元素
   * We get the corresponding element of x through a lookup and update x inplace.
   * 当y是0元素的时候,x的值是不需要变化的,因此只需要迭代y非0的元素即可,计算好的值更改x对应的元素
   */
  public static class AssignNonzerosIterateThatLookupThisInplaceUpdates extends VectorBinaryAssign {

    @Override
    public boolean isValid(Vector x, Vector y, DoubleDoubleFunction f) {
      return f.isLikeRightPlus();
    }

    /**
     * y中非0元素的数量
     * 去估计一个操作的价值--去迭代非0元素的时间
     * 获取x一个随机元素的时间
     */
    @Override
    public double estimateCost(Vector x, Vector y, DoubleDoubleFunction f) {
      return y.getNumNondefaultElements() * y.getIteratorAdvanceCost() * x.getLookupCost() * x.getLookupCost();
    }

    @Override
    public Vector assign(Vector x, Vector y, DoubleDoubleFunction f) {
      for (Element ye : y.nonZeroes()) {
        //第一个参数是定位到要更改x的哪个元素
        //第二个参数是x的值与y的值 用f函数处理,最新值才是更新的值
        x.setQuick(ye.index(), f.apply(x.getQuick(ye.index()), ye.get()));
      }
      return x;
    }
  }

  /**
   * If f(x, 0) = x, the zeros in y don't matter and we can simply iterate through the nonzeros of y.
   * We get the corresponding element of x through a lookup and update x by merging.
   * 与上面方法一样,只是针对如果y是稀松向量的时候会更好用
   */
  public static class AssignNonzerosIterateThatLookupThisMergeUpdates extends VectorBinaryAssign {

    @Override
    public boolean isValid(Vector x, Vector y, DoubleDoubleFunction f) {
      return f.isLikeRightPlus() && y.isSequentialAccess() && !x.isAddConstantTime();
    }

    @Override
    public double estimateCost(Vector x, Vector y, DoubleDoubleFunction f) {
      return y.getNumNondefaultElements() * y.getIteratorAdvanceCost() * y.getLookupCost();
    }

    @Override
    public Vector assign(Vector x, Vector y, DoubleDoubleFunction f) {
      OrderedIntDoubleMapping updates = new OrderedIntDoubleMapping(false);//产生一个稀松向量
      for (Element ye : y.nonZeroes()) {//循环y的非0向量元素
        //更新非0向量序号,和x与y计算的值
        updates.set(ye.index(), f.apply(x.getQuick(ye.index()), ye.get()));
      }

      //x去合并这些值即可
      x.mergeUpdates(updates);
      return x;
    }
  }

  /**
   * If f(x, 0) = x and f(0, y) = 0 the zeros in x and y don't matter and we can iterate through the nonzeros
   * in both x and y.
   * This is only possible if both x and y support sequential access.
   */
  public static class AssignIterateIntersection extends VectorBinaryAssign {

    @Override
    public boolean isValid(Vector x, Vector y, DoubleDoubleFunction f) {
      return f.isLikeLeftMult() && f.isLikeRightPlus() && x.isSequentialAccess() && y.isSequentialAccess();
    }

    @Override
    public double estimateCost(Vector x, Vector y, DoubleDoubleFunction f) {
      return Math.min(x.getNumNondefaultElements() * x.getIteratorAdvanceCost(),
          y.getNumNondefaultElements() * y.getIteratorAdvanceCost());
    }

    @Override
    public Vector assign(Vector x, Vector y, DoubleDoubleFunction f) {
      Iterator<Vector.Element> xi = x.nonZeroes().iterator();
      Iterator<Vector.Element> yi = y.nonZeroes().iterator();
      Vector.Element xe = null;
      Vector.Element ye = null;
      boolean advanceThis = true;
      boolean advanceThat = true;
      while (true) {
        if (advanceThis) {
          if (xi.hasNext()) {
            xe = xi.next();
          } else {
            break;
          }
        }
        if (advanceThat) {
          if (yi.hasNext()) {
            ye = yi.next();
          } else {
            break;
          }
        }
        if (xe.index() == ye.index()) {
          xe.set(f.apply(xe.get(), ye.get()));
          advanceThis = true;
          advanceThat = true;
        } else {
          if (xe.index() < ye.index()) { // f(x, 0) = 0
            advanceThis = true;
            advanceThat = false;
          } else { // f(0, y) = 0
            advanceThis = false;
            advanceThat = true;
          }
        }
      }
      return x;
    }
  }

  /**
   * If f(0, 0) = 0 we can iterate through the nonzeros in either x or y.
   * 只有x和y都是0的时候,结果就是0,要遍历x和y的所有的非0元素,只要有一个不是0的,都要去计算
   * In this case we iterate through them in parallel and update x by merging. Because we're iterating through
   * both vectors at the same time, x and y need to support sequential access.
   */
  public static class AssignIterateUnionSequentialMergeUpdates extends VectorBinaryAssign {

    @Override
    public boolean isValid(Vector x, Vector y, DoubleDoubleFunction f) {
      return !f.isDensifying() && x.isSequentialAccess() && y.isSequentialAccess() && !x.isAddConstantTime();
    }

    @Override
    public double estimateCost(Vector x, Vector y, DoubleDoubleFunction f) {
      return Math.max(x.getNumNondefaultElements() * x.getIteratorAdvanceCost(),
          y.getNumNondefaultElements() * y.getIteratorAdvanceCost());
    }

    @Override
    public Vector assign(Vector x, Vector y, DoubleDoubleFunction f) {
      Iterator<Vector.Element> xi = x.nonZeroes().iterator();
      Iterator<Vector.Element> yi = y.nonZeroes().iterator();
      Vector.Element xe = null;
      Vector.Element ye = null;
      boolean advanceThis = true;
      boolean advanceThat = true;
      OrderedIntDoubleMapping updates = new OrderedIntDoubleMapping(false);
      while (true) {
        if (advanceThis) {
          if (xi.hasNext()) {
            xe = xi.next();
          } else {
            xe = null;
          }
        }
        if (advanceThat) {
          if (yi.hasNext()) {
            ye = yi.next();
          } else {
            ye = null;
          }
        }
        if (xe != null && ye != null) { // both vectors have nonzero elements
          if (xe.index() == ye.index()) {
            xe.set(f.apply(xe.get(), ye.get()));
            advanceThis = true;
            advanceThat = true;
          } else {
            if (xe.index() < ye.index()) { // f(x, 0)
              xe.set(f.apply(xe.get(), 0));
              advanceThis = true;
              advanceThat = false;
            } else {
              updates.set(ye.index(), f.apply(0, ye.get()));
              advanceThis = false;
              advanceThat = true;
            }
          }
        } else if (xe != null) { // just the first one still has nonzeros
          xe.set(f.apply(xe.get(), 0));
          advanceThis = true;
          advanceThat = false;
        } else if (ye != null) { // just the second one has nonzeros
          updates.set(ye.index(), f.apply(0, ye.get()));
          advanceThis = false;
          advanceThat = true;
        } else { // we're done, both are empty
          break;
        }
      }
      x.mergeUpdates(updates);
      return x;
    }
  }

  /**
   * If f(0, 0) = 0 we can iterate through the nonzeros in either x or y.
   * In this case we iterate through them in parallel and update x inplace. Because we're iterating through
   * both vectors at the same time, x and y need to support sequential access.
   */
  public static class AssignIterateUnionSequentialInplaceUpdates extends VectorBinaryAssign {

    @Override
    public boolean isValid(Vector x, Vector y, DoubleDoubleFunction f) {
      return !f.isDensifying() && x.isSequentialAccess() && y.isSequentialAccess() && x.isAddConstantTime();
    }

    @Override
    public double estimateCost(Vector x, Vector y, DoubleDoubleFunction f) {
      return Math.max(x.getNumNondefaultElements() * x.getIteratorAdvanceCost(),
          y.getNumNondefaultElements() * y.getIteratorAdvanceCost());
    }

    @Override
    public Vector assign(Vector x, Vector y, DoubleDoubleFunction f) {
      Iterator<Vector.Element> xi = x.nonZeroes().iterator();
      Iterator<Vector.Element> yi = y.nonZeroes().iterator();
      Vector.Element xe = null;
      Vector.Element ye = null;
      boolean advanceThis = true;
      boolean advanceThat = true;
      while (true) {
        if (advanceThis) {
          if (xi.hasNext()) {
            xe = xi.next();
          } else {
            xe = null;
          }
        }
        if (advanceThat) {
          if (yi.hasNext()) {
            ye = yi.next();
          } else {
            ye = null;
          }
        }
        if (xe != null && ye != null) { // both vectors have nonzero elements
          if (xe.index() == ye.index()) {
            xe.set(f.apply(xe.get(), ye.get()));
            advanceThis = true;
            advanceThat = true;
          } else {
            if (xe.index() < ye.index()) { // f(x, 0)
              xe.set(f.apply(xe.get(), 0));
              advanceThis = true;
              advanceThat = false;
            } else {
              x.setQuick(ye.index(), f.apply(0, ye.get()));
              advanceThis = false;
              advanceThat = true;
            }
          }
        } else if (xe != null) { // just the first one still has nonzeros
          xe.set(f.apply(xe.get(), 0));
          advanceThis = true;
          advanceThat = false;
        } else if (ye != null) { // just the second one has nonzeros
          x.setQuick(ye.index(), f.apply(0, ye.get()));
          advanceThis = false;
          advanceThat = true;
        } else { // we're done, both are empty
          break;
        }
      }
      return x;
    }
  }

  /**
   * If f(0, 0) = 0 we can iterate through the nonzeros in either x or y.
   * In this case, we iterate through the nozeros of x and y alternatively (this works even when one of them
   * doesn't support sequential access). Since we're merging the results into x, when iterating through y, the
   * order of iteration matters and y must support sequential access.
   */
  public static class AssignIterateUnionRandomMergeUpdates extends VectorBinaryAssign {

    @Override
    public boolean isValid(Vector x, Vector y, DoubleDoubleFunction f) {
      return !f.isDensifying() && !x.isAddConstantTime() && y.isSequentialAccess();
    }

    @Override
    public double estimateCost(Vector x, Vector y, DoubleDoubleFunction f) {
      return Math.max(x.getNumNondefaultElements() * x.getIteratorAdvanceCost() * y.getLookupCost(),
          y.getNumNondefaultElements() * y.getIteratorAdvanceCost() * x.getLookupCost());
    }

    @Override
    public Vector assign(Vector x, Vector y, DoubleDoubleFunction f) {
      OpenIntHashSet visited = new OpenIntHashSet();
      for (Element xe : x.nonZeroes()) {
        xe.set(f.apply(xe.get(), y.getQuick(xe.index())));
        visited.add(xe.index());
      }
      OrderedIntDoubleMapping updates = new OrderedIntDoubleMapping(false);
      for (Element ye : y.nonZeroes()) {
        if (!visited.contains(ye.index())) {
          updates.set(ye.index(), f.apply(x.getQuick(ye.index()), ye.get()));
        }
      }
      x.mergeUpdates(updates);
      return x;
    }
  }

  /**
   * If f(0, 0) = 0 we can iterate through the nonzeros in either x or y.
   * In this case, we iterate through the nozeros of x and y alternatively (this works even when one of them
   * doesn't support sequential access). Because updates to x are inplace, neither x, nor y need to support
   * sequential access.
   */
  public static class AssignIterateUnionRandomInplaceUpdates extends VectorBinaryAssign {

    @Override
    public boolean isValid(Vector x, Vector y, DoubleDoubleFunction f) {
      return !f.isDensifying() && x.isAddConstantTime();
    }

    @Override
    public double estimateCost(Vector x, Vector y, DoubleDoubleFunction f) {
      return Math.max(x.getNumNondefaultElements() * x.getIteratorAdvanceCost() * y.getLookupCost(),
          y.getNumNondefaultElements() * y.getIteratorAdvanceCost() * x.getLookupCost());
    }
    @Override
    public Vector assign(Vector x, Vector y, DoubleDoubleFunction f) {
      OpenIntHashSet visited = new OpenIntHashSet();
      for (Element xe : x.nonZeroes()) {
        xe.set(f.apply(xe.get(), y.getQuick(xe.index())));
        visited.add(xe.index());
      }
      for (Element ye : y.nonZeroes()) {
        if (!visited.contains(ye.index())) {
          x.setQuick(ye.index(), f.apply(x.getQuick(ye.index()), ye.get()));
        }
      }
      return x;
    }
  }

  public static class AssignAllIterateSequentialMergeUpdates extends VectorBinaryAssign {

    @Override
    public boolean isValid(Vector x, Vector y, DoubleDoubleFunction f) {
      return x.isSequentialAccess() && y.isSequentialAccess() && !x.isAddConstantTime() && !x.isDense() && !y.isDense();
    }

    @Override
    public double estimateCost(Vector x, Vector y, DoubleDoubleFunction f) {
      return Math.max(x.size() * x.getIteratorAdvanceCost(), y.size() * y.getIteratorAdvanceCost());
    }

    @Override
    public Vector assign(Vector x, Vector y, DoubleDoubleFunction f) {
      Iterator<Vector.Element> xi = x.all().iterator();
      Iterator<Vector.Element> yi = y.all().iterator();
      OrderedIntDoubleMapping updates = new OrderedIntDoubleMapping(false);
      while (xi.hasNext() && yi.hasNext()) {
        Element xe = xi.next();
        updates.set(xe.index(), f.apply(xe.get(), yi.next().get()));
      }
      x.mergeUpdates(updates);
      return x;
    }
  }

  public static class AssignAllIterateSequentialInplaceUpdates extends VectorBinaryAssign {

    @Override
    public boolean isValid(Vector x, Vector y, DoubleDoubleFunction f) {
      return x.isSequentialAccess() && y.isSequentialAccess() && x.isAddConstantTime()
          && !x.isDense() && !y.isDense();
    }

    @Override
    public double estimateCost(Vector x, Vector y, DoubleDoubleFunction f) {
      return Math.max(x.size() * x.getIteratorAdvanceCost(), y.size() * y.getIteratorAdvanceCost());
    }

    @Override
    public Vector assign(Vector x, Vector y, DoubleDoubleFunction f) {
      Iterator<Vector.Element> xi = x.all().iterator();
      Iterator<Vector.Element> yi = y.all().iterator();
      while (xi.hasNext() && yi.hasNext()) {
        Element xe = xi.next();
        x.setQuick(xe.index(), f.apply(xe.get(), yi.next().get()));
      }
      return x;
    }
  }

  public static class AssignAllIterateThisLookupThatMergeUpdates extends VectorBinaryAssign {

    @Override
    public boolean isValid(Vector x, Vector y, DoubleDoubleFunction f) {
      return !x.isAddConstantTime() && !x.isDense();
    }

    @Override
    public double estimateCost(Vector x, Vector y, DoubleDoubleFunction f) {
      return x.size() * x.getIteratorAdvanceCost() * y.getLookupCost();
    }

    @Override
    public Vector assign(Vector x, Vector y, DoubleDoubleFunction f) {
      OrderedIntDoubleMapping updates = new OrderedIntDoubleMapping(false);
      for (Element xe : x.all()) {
        updates.set(xe.index(), f.apply(xe.get(), y.getQuick(xe.index())));
      }
      x.mergeUpdates(updates);
      return x;
    }
  }

  public static class AssignAllIterateThisLookupThatInplaceUpdates extends VectorBinaryAssign {

    @Override
    public boolean isValid(Vector x, Vector y, DoubleDoubleFunction f) {
      return x.isAddConstantTime() && !x.isDense();
    }

    @Override
    public double estimateCost(Vector x, Vector y, DoubleDoubleFunction f) {
      return x.size() * x.getIteratorAdvanceCost() * y.getLookupCost();
    }

    @Override
    public Vector assign(Vector x, Vector y, DoubleDoubleFunction f) {
      for (Element xe : x.all()) {
        x.setQuick(xe.index(), f.apply(xe.get(), y.getQuick(xe.index())));
      }
      return x;
    }
  }

  public static class AssignAllIterateThatLookupThisMergeUpdates extends VectorBinaryAssign {

    @Override
    public boolean isValid(Vector x, Vector y, DoubleDoubleFunction f) {
      return !x.isAddConstantTime() && !y.isDense();
    }

    @Override
    public double estimateCost(Vector x, Vector y, DoubleDoubleFunction f) {
      return y.size() * y.getIteratorAdvanceCost() * x.getLookupCost();
    }

    @Override
    public Vector assign(Vector x, Vector y, DoubleDoubleFunction f) {
      OrderedIntDoubleMapping updates = new OrderedIntDoubleMapping(false);
      for (Element ye : y.all()) {
        updates.set(ye.index(), f.apply(x.getQuick(ye.index()), ye.get()));
      }
      x.mergeUpdates(updates);
      return x;
    }
  }

  public static class AssignAllIterateThatLookupThisInplaceUpdates extends VectorBinaryAssign {

    @Override
    public boolean isValid(Vector x, Vector y, DoubleDoubleFunction f) {
      return x.isAddConstantTime() && !y.isDense();
    }

    @Override
    public double estimateCost(Vector x, Vector y, DoubleDoubleFunction f) {
      return y.size() * y.getIteratorAdvanceCost() * x.getLookupCost();
    }

    @Override
    public Vector assign(Vector x, Vector y, DoubleDoubleFunction f) {
      for (Element ye : y.all()) {
        x.setQuick(ye.index(), f.apply(x.getQuick(ye.index()), ye.get()));
      }
      return x;
    }
  }

  public static class AssignAllLoopMergeUpdates extends VectorBinaryAssign {

    @Override
    public boolean isValid(Vector x, Vector y, DoubleDoubleFunction f) {
      return !x.isAddConstantTime();
    }

    @Override
    public double estimateCost(Vector x, Vector y, DoubleDoubleFunction f) {
      return x.size() * x.getLookupCost() * y.getLookupCost();
    }

    @Override
    public Vector assign(Vector x, Vector y, DoubleDoubleFunction f) {
      OrderedIntDoubleMapping updates = new OrderedIntDoubleMapping(false);
      for (int i = 0; i < x.size(); ++i) {
        updates.set(i, f.apply(x.getQuick(i), y.getQuick(i)));
      }
      x.mergeUpdates(updates);
      return x;
    }
  }

  public static class AssignAllLoopInplaceUpdates extends VectorBinaryAssign {

    @Override
    public boolean isValid(Vector x, Vector y, DoubleDoubleFunction f) {
      return x.isAddConstantTime();
    }

    @Override
    public double estimateCost(Vector x, Vector y, DoubleDoubleFunction f) {
      return x.size() * x.getLookupCost() * y.getLookupCost();
    }

    @Override
    public Vector assign(Vector x, Vector y, DoubleDoubleFunction f) {
      for (int i = 0; i < x.size(); ++i) {
        x.setQuick(i, f.apply(x.getQuick(i), y.getQuick(i)));
      }
      return x;
    }
  }
}
