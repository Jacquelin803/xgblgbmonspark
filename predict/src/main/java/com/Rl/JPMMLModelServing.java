package com.Rl;


import org.dmg.pmml.FieldName;
import java.util.HashMap;
import java.util.Map;

public class JPMMLModelServing {
    public static void main(String[] args){

//        JavaModelServer javaModelServer = new JavaModelServer("xgb-model/src/main/resources/pmml/XgbClf.pmml");
        JavaModelServer javaModelServer = new JavaModelServer("lgbm-model/src/main/data/pmml/LgbmReg.pmml");

        HashMap<String, Object> featureMap = new HashMap<>();
        //binary or multi
//        featureMap.put("sepal_length", 5.1);
//        featureMap.put("sepal_width", 3.5);
//        featureMap.put("petal_length", 1.4);
        featureMap.put("petal_width", 0.2);

        //reg
        featureMap.put("RM", 3.863);
        featureMap.put("LSTAT", 13.33);
        featureMap.put("PTRATIO", 20.2);

;



        Map<FieldName, ?> result = javaModelServer.forecast(featureMap);

        for (Map.Entry<FieldName, ?> field : result.entrySet()){
            System.out.println(field.getKey().getValue() + ":\t" +  field.getValue());
        }

        for(int i = 0 ; i < result.size(); i++){
            System.out.println(result);
        }

//        System.out.println(result.get(new FieldName("probability(1)")));

    }
}
