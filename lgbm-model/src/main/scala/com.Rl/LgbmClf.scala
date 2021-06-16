package com.Rl

import java.io.{ByteArrayInputStream,FileOutputStream}
import com.microsoft.ml.spark.lightgbm.{LightGBMBooster, LightGBMClassificationModel, LightGBMClassifier}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.jpmml.lightgbm.LightGBMUtil
import org.jpmml.model.MetroJAXBUtil

class LgbmClf {
  def train(oriDF: DataFrame): Unit = {

    val labelCol = "IS_TARGET"

    var originalData=oriDF
    val data_col = originalData.columns
    val vecCols = data_col.filter(!_.contains(labelCol))
    val assembler = new VectorAssembler()
      .setInputCols(vecCols)
      .setOutputCol("features")

    val classifier: LightGBMClassifier = new LightGBMClassifier()
      .setLearningRate(0.03)
      .setNumIterations(100)
      .setNumLeaves(31)
      .setEarlyStoppingRound(100)
      .setLabelCol(labelCol)

    val pipeline: Pipeline = new Pipeline().setStages(Array(assembler, classifier))

    val Array(tr, te) = originalData.randomSplit(Array(0.7, 0.3), 666)
    val model = pipeline.fit(tr)
    val transDF = model.transform(te)
    transDF.show()
    val evaluator = new BinaryClassificationEvaluator().setLabelCol(labelCol).setRawPredictionCol("prediction")
    println("areaUnderROC",evaluator.evaluate(transDF))

    //导出pmml
    val classificationModel = model.stages(1).asInstanceOf[LightGBMClassificationModel]
    saveToPmml(classificationModel.getModel, "lgbm-model/src/main/data/pmml/LgbmClf.pmml")

  }

  //保存pmml模型
  def saveToPmml(booster: LightGBMBooster, path: String): Unit = {
    try {
      val gbdt = LightGBMUtil.loadGBDT(new ByteArrayInputStream(booster.model.getBytes))
      import scala.collection.JavaConversions.mapAsJavaMap
      val pmml = gbdt.encodePMML(null, null, Map("compact" -> true))
      MetroJAXBUtil.marshalPMML(pmml, new FileOutputStream(path))
    } catch {
      case e: Exception => e.printStackTrace()
    }
  }

}

object RunLgbmClf{
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.WARN)
    val conf = new SparkConf()
      .setMaster("local[*]")
      .setAppName("spark");
    val spark=  SparkSession.builder().config(conf).getOrCreate()

    var originalData: DataFrame = spark.read.option("header", "true")
      .option("inferSchema", "true")
      .csv("lgbm-model/src/main/data/iris.csv")
    originalData=originalData.drop("CUST_NO")
    val model = new LgbmClf
    model.train(originalData)
  }

}
