package org.apache.mahout.classifier.df.split;

import java.util.ArrayList;
import java.util.List;

import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;

public class Test {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		FullRunningAverage ra = new FullRunningAverage();
		
		List<Double> list = new ArrayList<Double>();
		/*list.add(21.0d);
		list.add(18.0d);
		list.add(19.0d);
		list.add(15.0d);
		list.add(20.0d);*/
		list.add(20.0d);
		list.add(20.0d);
		list.add(20.0d);
		list.add(20.0d);
		
		double v = 0.0;
		ra.addDatum(19.0d);
		
		for(Double score:list){
			 double mk = ra.getAverage();//获取此时的平均值(标签)
			 System.out.println("getAverage:"+mk);
		     ra.addDatum(score);//添加该标签对应的值
		     double v_temp = (score - mk) * (score - ra.getAverage());
		     v += v_temp;
		     System.out.println(v+"=="+v_temp+"=="+v/ra.getCount());
		}

	}

}

