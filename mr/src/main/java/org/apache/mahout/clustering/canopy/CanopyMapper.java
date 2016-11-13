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

package org.apache.mahout.clustering.canopy;

import java.io.IOException;
import java.util.Collection;

import com.google.common.collect.Lists;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.math.VectorWritable;

@Deprecated
class CanopyMapper extends
    Mapper<WritableComparable<?>, VectorWritable, Text, VectorWritable> {

  private final Collection<Canopy> canopies = Lists.newArrayList();

  private CanopyClusterer canopyClusterer;

  //如果一个中心点所在的圆周,包含超过clusterFilter数量的才可以称之为一个聚类点,否则不算一个分类
  private int clusterFilter;//本次计算观察的点数超过了clusterFilter阈值,则写入程序,如果没有超过该阈值,则不会当做中心点处理,即不会写入到reduce中

  @Override
  protected void map(WritableComparable<?> key, VectorWritable point,
      Context context) throws IOException, InterruptedException {
    canopyClusterer.addPointToCanopies(point.get(), canopies);
  }

  @Override
  protected void setup(Context context) throws IOException,
      InterruptedException {
    super.setup(context);
    canopyClusterer = CanopyConfigKeys.configureCanopyClusterer(context.getConfiguration());
    clusterFilter = Integer.parseInt(context.getConfiguration().get(
        CanopyConfigKeys.CF_KEY));
  }

  @Override
  protected void cleanup(Context context) throws IOException,
      InterruptedException {
    for (Canopy canopy : canopies) {
      canopy.computeParameters();
      if (canopy.getNumObservations() > clusterFilter) {//本次计算观察的点数超过了clusterFilter阈值,则写入程序,如果没有超过该阈值,则不会当做中心点处理,即不会写入到reduce中
        context.write(new Text("centroid"), new VectorWritable(canopy
            .getCenter()));//设置一个中心点,写到输出中
      }
    }
    super.cleanup(context);
  }
}
