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
import com.google.common.io.Closeables;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.codehaus.jackson.map.ObjectMapper;
import org.codehaus.jackson.type.TypeReference;

import java.io.IOException;
import java.nio.charset.Charset;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Contains information about the attributes.
 * 该对象表示数据的title
 */
@Deprecated
public class Dataset {

  /**
   * Attributes type
   * 表示每一个属性的性质
   */
  public enum Attribute {
    IGNORED,//忽略该属性,一般是每一行数据的ID,要被忽略的
    NUMERICAL,//整数形式
    CATEGORICAL,//分类形式
    LABEL;//该属性是标签

    public boolean isNumerical() {
      return this == NUMERICAL;
    }

    public boolean isCategorical() {
      return this == CATEGORICAL;
    }

    public boolean isLabel() {
      return this == LABEL;
    }

    public boolean isIgnored() {
      return this == IGNORED;
    }
    
    private static Attribute fromString(String from) {
      Attribute toReturn = LABEL;
      if (NUMERICAL.toString().equalsIgnoreCase(from)) {
        toReturn = NUMERICAL;
      } else if (CATEGORICAL.toString().equalsIgnoreCase(from)) {
        toReturn = CATEGORICAL;
      } else if (IGNORED.toString().equalsIgnoreCase(from)) {
        toReturn = IGNORED;
      }
      return toReturn;
    }
  }

  private Attribute[] attributes;//有效的属性集合

  /**
   * list of ignored attributes
   * 哪些属性是忽略的,内容就是所有忽略的属性集合
   */
  private int[] ignored;

  /**
   * distinct values (CATEGORIAL attributes only)
   */
  private String[][] values;//每一个分类属性中,分类的数据集合,即每一个分类属性对应一个数组,数组内包含的是分类信息

  /**
   * index of the label attribute in the loaded data (without ignored attributed)
   * 第几个有效的属性集合是label属性
   */
  private int labelId;

  /**
   * number of instances in the dataset
   * 一共有多少条数据
   */
  private int nbInstances;
  
  /** JSON serial/de-serial-izer */
  private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

  // Some literals for JSON representation
  static final String TYPE = "type";//写入该属性的类型
  static final String VALUES = "values";//写入该属性如果是分类类型的,要写入分类明细
  static final String LABEL = "label";//写入boolean值,表示该属性是否是label属性

  protected Dataset() {}

  /**
   * Should only be called by a DataLoader
   *
   * @param attrs  attributes description 每一个属性对应的性质
   * @param values distinct values for all CATEGORICAL attributes 数组List,每一个List表示对应的属性包含哪些分类或者label
   */
  Dataset(Attribute[] attrs, List<String>[] values,
		  int nbInstances, //一共有多少条数据
		  boolean regression) {
    validateValues(attrs, values);

    int nbattrs = countAttributes(attrs);//计算有效的属性数量(刨除忽略的属性)

    // the label values are set apart
    attributes = new Attribute[nbattrs];
    this.values = new String[nbattrs][];
    ignored = new int[attrs.length - nbattrs]; // nbignored = total - nbattrs

    labelId = -1;
    int ignoredId = 0;//忽略的属性下标
    int ind = 0;
    for (int attr = 0; attr < attrs.length; attr++) {
      if (attrs[attr].isIgnored()) {
        ignored[ignoredId++] = attr;
        continue;
      }

      if (attrs[attr].isLabel()) {
        if (labelId != -1) {
          throw new IllegalStateException("Label found more than once");
        }
        labelId = ind;
        if (regression) {
          attrs[attr] = Attribute.NUMERICAL;
        } else {
          attrs[attr] = Attribute.CATEGORICAL;
        }
      }

      if (attrs[attr].isCategorical() || (!regression && attrs[attr].isLabel())) {
        this.values[ind] = new String[values[attr].size()];
        values[attr].toArray(this.values[ind]);
      }

      attributes[ind++] = attrs[attr];
    }

    if (labelId == -1) {
      throw new IllegalStateException("Label not found");
    }

    this.nbInstances = nbInstances;
  }

  public int nbValues(int attr) {//该分类属性有多少个分类
    return values[attr].length;
  }

  //返回有多少个label 标签属性
  public int nblabels() {
    return values[labelId].length;
  }
  
  //返回label标签集合
  public String[] labels() {
    return Arrays.copyOf(values[labelId], nblabels());
  }

  public int getLabelId() {
    return labelId;
  }

  public double getLabel(Instance instance) {//获取标签属性对应的值
    return instance.get(getLabelId());
  }
  
  public Attribute getAttribute(int attr) {//获取一个属性
    return attributes[attr];
  }

  /**
   * Returns the code used to represent the label value in the data
   *
   * @param label label's value to code
   * @return label's code
   */
  public int labelCode(String label) {
    return ArrayUtils.indexOf(values[labelId], label);//获取一个标签内容对应第几个位置
  }

  /**
   * Returns the label value in the data
   * This method can be used when the criterion variable is the categorical attribute.
   *
   * @param code label's code
   * @return label's value
   * 获取一个位置对应的标签内容
   */
  public String getLabelString(double code) {
    // handle the case (prediction is NaN)
    if (Double.isNaN(code)) {
      return "unknown";
    }
    return values[labelId][(int) code];
  }
  
  @Override
  public String toString() {
    return "attributes=" + Arrays.toString(attributes);
  }

  /**
   * Converts a token to its corresponding integer code for a given attribute
   *
   * @param attr attribute index
   * 获取某一个标签给定分类内容对应的序号
   */
  public int valueOf(int attr, String token) {
    Preconditions.checkArgument(!isNumerical(attr), "Only for CATEGORICAL attributes");
    Preconditions.checkArgument(values != null, "Values not found (equals null)");
    return ArrayUtils.indexOf(values[attr], token);
  }

  public int[] getIgnored() {
    return ignored;
  }

  /**
   * @return number of attributes that are not IGNORED
   * 计算有效的属性数量(刨除忽略的属性)
   */
  private static int countAttributes(Attribute[] attrs) {
    int nbattrs = 0;
    for (Attribute attr : attrs) {
      if (!attr.isIgnored()) {//刨除忽略的属性
        nbattrs++;
      }
    }
    return nbattrs;
  }

  //校验合法性
  private static void validateValues(Attribute[] attrs, List<String>[] values) {
    Preconditions.checkArgument(attrs.length == values.length, "attrs.length != values.length");
    
    //校验不是分类的属性.是不需要有属性值集合的
    for (int attr = 0; attr < attrs.length; attr++) {
      Preconditions.checkArgument(!attrs[attr].isCategorical() || values[attr] != null,
          "values not found for attribute " + attr);
    }
  }

  /**
   * @return number of attributes 总属性数量
   */
  public int nbAttributes() {
    return attributes.length;
  }

  /**
   * Is this a numerical attribute ?
   *
   * @param attr index of the attribute to check
   * @return true if the attribute is numerical
   * 该属性是否是数字属性
   */
  public boolean isNumerical(int attr) {
    return attributes[attr].isNumerical();
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof Dataset)) {
      return false;
    }

    Dataset dataset = (Dataset) obj;

    if (!Arrays.equals(attributes, dataset.attributes)) {
      return false;
    }

    for (int attr = 0; attr < nbAttributes(); attr++) {
      if (!Arrays.equals(values[attr], dataset.values[attr])) {
        return false;
      }
    }

    return labelId == dataset.labelId && nbInstances == dataset.nbInstances;
  }

  @Override
  public int hashCode() {
    int hashCode = labelId + 31 * nbInstances;
    for (Attribute attr : attributes) {
      hashCode = 31 * hashCode + attr.hashCode();
    }
    for (String[] valueRow : values) {
      if (valueRow == null) {
        continue;
      }
      for (String value : valueRow) {
        hashCode = 31 * hashCode + value.hashCode();
      }
    }
    return hashCode;
  }

  /**
   * Loads the dataset from a file
   *
   * @throws java.io.IOException
   * 从path路径加载json数据.最终生成Dataset对象
   */
  public static Dataset load(Configuration conf, Path path) throws IOException {
    FileSystem fs = path.getFileSystem(conf);
    long bytesToRead = fs.getFileStatus(path).getLen();//字节内容
    byte[] buff = new byte[Long.valueOf(bytesToRead).intValue()];
    FSDataInputStream input = fs.open(path);
    try {
      input.readFully(buff);
    } finally {
      Closeables.close(input, true);
    }
    String json = new String(buff, Charset.defaultCharset());
    return fromJSON(json);
  }
  

  /**
   * Serialize this instance to JSON
   * @return some JSON
   */
  public String toJSON() {
    List<Map<String, Object>> toWrite = new LinkedList<>();
    // attributes does not include ignored columns and it does include the class label
    int ignoredCount = 0;
    for (int i = 0; i < attributes.length + ignored.length; i++) {
      Map<String, Object> attribute;
      int attributesIndex = i - ignoredCount;
      if (ignoredCount < ignored.length && i == ignored[ignoredCount]) {//输入忽略属性
        // fill in ignored atttribute
        attribute = getMap(Attribute.IGNORED, null, false);
        ignoredCount++;
      } else if (attributesIndex == labelId) {//该输入label标签
        // fill in the label
        attribute = getMap(attributes[attributesIndex], values[attributesIndex], true);
      } else  {//输入正常有效的标签
        // normal attribute
        attribute = getMap(attributes[attributesIndex], values[attributesIndex], false);
      }
      toWrite.add(attribute);
    }
    try {
      return OBJECT_MAPPER.writeValueAsString(toWrite);
    } catch (Exception ex) {
      throw new RuntimeException(ex);
    }
  }

  /**
   * De-serialize an instance from a string
   * @param json From which an instance is created
   * @return A shiny new Dataset
   */
  public static Dataset fromJSON(String json) {
    List<Map<String, Object>> fromJSON;
    try {
      fromJSON = OBJECT_MAPPER.readValue(json, new TypeReference<List<Map<String, Object>>>() {});
    } catch (Exception ex) {
      throw new RuntimeException(ex);
    }
    List<Attribute> attributes = new LinkedList<>();
    List<Integer> ignored = new LinkedList<>();
    String[][] nominalValues = new String[fromJSON.size()][];
    Dataset dataset = new Dataset();
    for (int i = 0; i < fromJSON.size(); i++) {
      Map<String, Object> attribute = fromJSON.get(i);
      if (Attribute.fromString((String) attribute.get(TYPE)) == Attribute.IGNORED) {
        ignored.add(i);
      } else {
        Attribute asAttribute = Attribute.fromString((String) attribute.get(TYPE));
        attributes.add(asAttribute);
        if ((Boolean) attribute.get(LABEL)) {
          dataset.labelId = i - ignored.size();
        }
        if (attribute.get(VALUES) != null) {
          List<String> get = (List<String>) attribute.get(VALUES);
          String[] array = get.toArray(new String[get.size()]);
          nominalValues[i - ignored.size()] = array;
        }
      }
    }
    dataset.attributes = attributes.toArray(new Attribute[attributes.size()]);
    dataset.ignored = new int[ignored.size()];
    dataset.values = nominalValues;
    for (int i = 0; i < dataset.ignored.length; i++) {
      dataset.ignored[i] = ignored.get(i);
    }
    return dataset;
  }
  
  /**
   * Generate a map to describe an attribute
   * @param type The type
   * @param values - values
   * @param isLabel - is a label
   * @return map of (AttributeTypes, Values)
   */
  private Map<String, Object> getMap(Attribute type, String[] values, boolean isLabel) {
    Map<String, Object> attribute = new HashMap<>();
    attribute.put(TYPE, type.toString().toLowerCase(Locale.getDefault()));
    attribute.put(VALUES, values);
    attribute.put(LABEL, isLabel);
    return attribute;
  }
}
