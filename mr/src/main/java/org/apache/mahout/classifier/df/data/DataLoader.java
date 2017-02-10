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

package org.apache.mahout.classifier.df.data;

import com.google.common.base.Preconditions;
import com.google.common.collect.Lists;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.mahout.classifier.df.data.Dataset.Attribute;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Scanner;
import java.util.Set;
import java.util.regex.Pattern;

/**
 * Converts the input data to a Vector Array using the information given by the Dataset.<br>
 * Generates for each line a Vector that contains :<br>
 * <ul>
 * <li>double parsed value for NUMERICAL attributes</li>
 * <li>int value for CATEGORICAL and LABEL attributes</li>
 * </ul>
 * <br>
 * adds an IGNORED first attribute that will contain a unique id for each instance, which is the line number
 * of the instance in the input data
 * 数据加载类,解析传入的数据,转换成数据集
 */
@Deprecated
public final class DataLoader {

  private static final Logger log = LoggerFactory.getLogger(DataLoader.class);

  private static final Pattern SEPARATORS = Pattern.compile("[, ]");//正则表达式,按照,号去拆分每一行的数据

  private DataLoader() {}

  /**
   * Converts a comma-separated String to a Vector.
   * 
   * @param attrs
   *          attributes description 每一列的属性信息
   * @param values
   *          used to convert CATEGORICAL attribute values to Integer 添加每一个分类对应哪些分类信息,每一个属性对应一个Set,Set内容存储该属性对应的分类集合
   * @return false if there are missing values '?' or NUMERICAL attribute values is not numeric 
   * 如果有一个缺失值用?号表示,或者数字不是数字,则返回false
   */
  private static boolean parseString(Attribute[] attrs, Set<String>[] values, 
		  CharSequence string,//等待要解析的一行信息
    boolean regression) {//是否要回归
    String[] tokens = SEPARATORS.split(string);//拆分一行的数据
    Preconditions.checkArgument(tokens.length == attrs.length,//确保拆分的每一行数据的列数要与Attribute数量相对应
        "Wrong number of attributes in the string: " + tokens.length + ". Must be: " + attrs.length);

    // extract tokens and check is there is any missing value
    for (int attr = 0; attr < attrs.length; attr++) {
      if (!attrs[attr].isIgnored() && "?".equals(tokens[attr])) {//不是忽略的属性.有缺失值
        return false; // missing value
      }
    }

    for (int attr = 0; attr < attrs.length; attr++) {
      if (!attrs[attr].isIgnored()) {//只要不是忽略的属性.则就处理
        String token = tokens[attr];
        if (attrs[attr].isCategorical() || (!regression && attrs[attr].isLabel())) {//是分类表情,或者是label但不是回归(不太理解),总之就是如果是label,但是regression=false的时候也会收集标签内容
          // update values
          if (values[attr] == null) {
            values[attr] = new HashSet<>();
          }
          values[attr].add(token);//添加分类集合
        } else {
          try {
            Double.parseDouble(token);//必须是double类型的
          } catch (NumberFormatException e) {
            return false;
          }
        }
      }
    }

    return true;
  }

  /**
   * Loads the data from a file
   * 
   * @param fs
   *          file system
   * @param fpath
   *          data file path
   * @throws IOException
   *           if any problem is encountered
   * 从path读取数据,每一行数据都根据Dataset中设置的title来生成Instance实例对象,最终生成数据集合
   */
  public static Data loadData(Dataset dataset, FileSystem fs, Path fpath) throws IOException {
    FSDataInputStream input = fs.open(fpath);
    Scanner scanner = new Scanner(input, "UTF-8");

    List<Instance> instances = new ArrayList<>();

    DataConverter converter = new DataConverter(dataset);

    while (scanner.hasNextLine()) {//读取每一行数据
      String line = scanner.nextLine();
      if (!line.isEmpty()) {
        Instance instance = converter.convert(line);//将数据转换成Instance
        if (instance != null) {
          instances.add(instance);
        } else {
          // missing values found
          log.warn("{}: missing values", instances.size());
        }
      } else {
        log.warn("{}: empty string", instances.size());
      }
    }

    scanner.close();
    return new Data(dataset, instances);
  }


  /** Loads the data from multiple paths specified by pathes 
   * 从多个数据path中获取数据集合
   **/
  public static Data loadData(Dataset dataset, FileSystem fs, Path[] pathes) throws IOException {
    List<Instance> instances = new ArrayList<>();

    for (Path path : pathes) {
      Data loadedData = loadData(dataset, fs, path);
      for (int index = 0; index <= loadedData.size(); index++) {
        instances.add(loadedData.get(index));
      }
    }
    return new Data(dataset, instances);
  }

  /** Loads the data from a String array */
  public static Data loadData(Dataset dataset, String[] data) {
    List<Instance> instances = new ArrayList<>();

    DataConverter converter = new DataConverter(dataset);

    for (String line : data) {
      if (!line.isEmpty()) {
        Instance instance = converter.convert(line);
        if (instance != null) {
          instances.add(instance);
        } else {
          // missing values found
          log.warn("{}: missing values", instances.size());
        }
      } else {
        log.warn("{}: empty string", instances.size());
      }
    }

    return new Data(dataset, instances);
  }

  /**
   * Generates the Dataset by parsing the entire data
   * 
   * @param descriptor  attributes description
   * @param regression  if true, the label is numerical
   * @param fs  file system
   * @param path  data path
   * 从path路径加载数据集合,具体逻辑参见generateDataset方法即可
   */
  public static Dataset generateDataset(CharSequence descriptor,
                                        boolean regression,
                                        FileSystem fs,
                                        Path path) throws DescriptorException, IOException {
    Attribute[] attrs = DescriptorUtils.parseDescriptor(descriptor);

    FSDataInputStream input = fs.open(path);
    Scanner scanner = new Scanner(input, "UTF-8");

    // used to convert CATEGORICAL attribute to Integer
    @SuppressWarnings("unchecked")
    Set<String>[] valsets = new Set[attrs.length];//加载每一个属性一个Set,Set内容是该属性对应的分类或者标签集合

    int size = 0;
    while (scanner.hasNextLine()) {
      String line = scanner.nextLine();
      if (!line.isEmpty()) {
        if (parseString(attrs, valsets, line, regression)) {
          size++;
        }
      }
    }

    scanner.close();

    @SuppressWarnings("unchecked")
    List<String>[] values = new List[attrs.length];
    for (int i = 0; i < valsets.length; i++) {
      if (valsets[i] != null) {
        values[i] = Lists.newArrayList(valsets[i]);
      }
    }

    return new Dataset(attrs, values, size, regression);
  }

  /**
   * Generates the Dataset by parsing the entire data
   * 解析传入的数据,转换成数据集
   * @param descriptor
   *          attributes description
   */
  public static Dataset generateDataset(CharSequence descriptor,//数据中每一个属性的性质
                                        boolean regression,//是否回归
                                        String[] data) //要解析的原始数据 
                                        		throws DescriptorException {
    Attribute[] attrs = DescriptorUtils.parseDescriptor(descriptor); //解析每一个属性的性质

    // used to convert CATEGORICAL attributes to Integer
    @SuppressWarnings("unchecked")
    Set<String>[] valsets = new Set[attrs.length];//加载每一个属性一个Set,Set内容是该属性对应的分类或者标签集合

    int size = 0;//有多少有效的行数数据
    for (String aData : data) {
      if (!aData.isEmpty()) {
        if (parseString(attrs, valsets, aData, regression)) {
          size++;
        }
      }
    }

    @SuppressWarnings("unchecked")
    List<String>[] values = new List[attrs.length];//加载每一个属性一个List,List内容是该属性对应的分类或者标签集合
    for (int i = 0; i < valsets.length; i++) {
      if (valsets[i] != null) {
        values[i] = Lists.newArrayList(valsets[i]);
      }
    }

    return new Dataset(attrs, values, size, regression);
  }

}
