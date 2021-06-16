package com.Rl


import java.nio.file.{Files, Paths}
import ml.dmlc.xgboost4j.scala.spark.{XGBoostClassificationModel, XGBoostClassifier}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{countDistinct, udf}
import org.jpmml.sparkml.PMMLBuilder

class XgbModel {
  def train(spark:SparkSession,rawSamples: DataFrame): Unit = {
    val numClass: Int = rawSamples.agg(countDistinct("IS_TARGET")).collect().map(_ (0)).toList(0).toString.toInt
    println(numClass)
    val train_data=rawSamples.drop("CUST_NO")
    val data_col=train_data.columns
    val feat_col = data_col.filter(!_.contains("IS_TARGET"))
    import spark.implicits._
    //split samples into training samples and validation samples
    val Array(trainingSamples, validationSamples) = train_data.randomSplit(Array(0.6, 0.4))
    val featureAssembler = new VectorAssembler().setInputCols(feat_col).setOutputCol("features")
    val t1 = System.currentTimeMillis()
    val (xgboost,evaluator)= if(numClass == 2){
      (
        new XGBoostClassifier(
          Map("objective" -> "binary:logistic", "missing" -> 0.0,"eval_metric"->"error"))
          .setFeaturesCol("features").setLabelCol("IS_TARGET"),
        new BinaryClassificationEvaluator().setLabelCol("IS_TARGET").setRawPredictionCol("probability"))
    }else{
      (
        new XGBoostClassifier(
          Map("objective" -> "multi:softprob", "missing" -> 0.0,"num_class" -> numClass,"eval_metric"->"merror"))
          .setFeaturesCol("features").setLabelCol("IS_TARGET"),
        new MulticlassClassificationEvaluator().setLabelCol("IS_TARGET"))
    }
    val pipeline_xgb =  new Pipeline().setStages(Array(featureAssembler,xgboost))
    val paramGrid_xgb = new ParamGridBuilder().
      addGrid(xgboost.maxDepth, Array(5, 8)).
      addGrid(xgboost.eta, Array(0.05, 0.3)).
      addGrid(xgboost.numRound,Array(10,20,30)).
      build()

    val CV_xgb = new CrossValidator().
      setEstimator(pipeline_xgb).
      setEvaluator(evaluator).
      setEstimatorParamMaps(paramGrid_xgb).
      setNumFolds(2).
      setParallelism(24)

    val CVModel_xgb = CV_xgb.fit(trainingSamples)
    val bestModel_xgb_CV = CVModel_xgb.bestModel.asInstanceOf[PipelineModel].stages(1).asInstanceOf[XGBoostClassificationModel]
    val bestPara: ParamMap = bestModel_xgb_CV.extractParamMap()
    println("The params of best Xgb model : " + bestPara)

    val model=  CVModel_xgb.bestModel.asInstanceOf[PipelineModel]
    val vecToArray = udf((p: Vector) => p.toArray)
    var trans_data = model.transform(validationSamples)
    trans_data = trans_data.withColumn("prob",vecToArray($"probability")(1))
    val pmmlBytes = new PMMLBuilder(trainingSamples.schema, model).buildByteArray()
    Files.write(Paths.get("xgb-model/src/main/resources/pmml/XgbClf.pmml"), pmmlBytes)
    val t2 = System.currentTimeMillis()
    println("GBDT TIME COST：" + (t2 - t1)/1000)
    trans_data.show(100)
    //    println("accuracy",evaluator.evaluate(trans_data))
  }

}
object RunXgb{
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.WARN)
    val conf = new SparkConf()
      .set("spark.driver.host", "localhost")
      .setMaster("local[*]")
      .setAppName("Xgb")

    val spark = SparkSession.builder.config(conf).getOrCreate()
    println(spark.version)
    val rawSamples = spark.read.format("csv").option("sep", ",").option("inferSchema", "true")
      .option("header", "true").load("xgb-model/src/main/resources/iris.csv")
    val model=new XgbModel
    model.train(spark,rawSamples)
    spark.close()
  }
}
