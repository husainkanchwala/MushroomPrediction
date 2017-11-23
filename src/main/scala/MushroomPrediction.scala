case class MushroomSchema(eatable: String, capshape: String, capsurface: String,capcolor: String,bruises: String,odor: String,gillattachment: String
                          ,gillspacing: String,gillsize: String,gillcolor: String,stalkshape: String,stalkroot: String, stalksurfacear: String, stalksurfacebr:String,
                          stalkcolorar:String,stalkcolorbr:String,vtype:String,vcolor:String,ringnumber:String,ringtype:String,
                          sporeprintcolor:String,population:String,habitat:String)
object MushroomPrediction {
  def main(args: Array[String]) {
    import org.apache.spark.sql.Encoders
    import org.apache.spark.sql.functions._
    import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
    import org.apache.spark.ml.feature.Binarizer
    import org.apache.spark.ml.feature.RFormula
    import org.apache.spark.SparkContext
    import org.apache.spark.SparkConf
    import org.apache.spark.sql.SQLContext
//    val inputFile = args(0);
//    val outputFileModel = args(1);
//    val outputFileConfusion = args(2);
    val conf = new SparkConf().setAppName("Mushroom Categorization").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)
    var schema = Encoders.product[MushroomSchema].schema
    var data = sqlContext.read.format("csv").
      option("header", "true").
      option("delimiter", ",").
      schema(schema).
      load("mushrooms.csv")
    val cleanData = data.na.drop()
    val supervised = new RFormula().setFormula("eatable ~ capshape + capsurface + capcolor + bruises + odor + gillattachment + gillspacing + gillsize + gillcolor + stalkshape + stalkroot + stalksurfacear + stalksurfacebr + stalkcolorar + stalkcolorbr  + stalkcolorar + stalkcolorbr + vcolor + ringnumber + ringtype + sporeprintcolor + population + habitat")
    val fittedRF = supervised.fit(cleanData)
    val preparedDF = fittedRF.transform(cleanData)
    val Array(train, test) = preparedDF.randomSplit(Array(0.7, 0.3))
    val dt = new RandomForestClassifier().setLabelCol("label").setFeaturesCol("features")
    val model = dt.fit(train)
//    model.save(outputFileModel);
    val predictions = model.transform(test)
    val binarizer: Binarizer = new Binarizer().
      setInputCol("prediction").
      setOutputCol("binarized_prediction").
      setThreshold(0.5)
    val predictionBinary = binarizer.transform(predictions)
    val wrongPredictions = predictionBinary.where(expr("label != binarized_prediction"))
    val countErrors = wrongPredictions.groupBy("label").agg(count("prediction").alias("Errors"))
    val correctPredictions = predictionBinary.where(expr("label == binarized_prediction"))
    val countCorrectPredictions = correctPredictions.groupBy("label").agg(count("prediction").alias("Correct"))
    predictionBinary.show

    var FP : Long = 0;
    val errorMatrixFP = countErrors.filter(countErrors("label").equalTo("0.0")).select("Errors").collectAsList()
    if(errorMatrixFP.size() != 0) {
      FP = errorMatrixFP.get(0).getInt(0)
    }

    var FN : Long = 0;
    val errorMatrixFN = countErrors.filter(countErrors("label").equalTo("1.0")).select("Errors").collectAsList()
    if(errorMatrixFN.size() != 0) {
      FN = errorMatrixFN.get(0).getLong(0)
    }

    var TN : Long = 0;
    val correctMatrixTN = countCorrectPredictions.filter(countCorrectPredictions("label").equalTo("0.0")).select("Correct").collectAsList()
    if(correctMatrixTN.size() != 0) {
      TN = correctMatrixTN.get(0).getLong(0)
    }

    var TP : Long = 0;
    val correctMatrixTP = countCorrectPredictions.filter(countCorrectPredictions("label").equalTo("1.0")).select("Correct").collectAsList()
    if(correctMatrixTP.size() != 0) {
      TP = correctMatrixTP.get(0).getLong(0)
    }

    var total = FP+FN+TN+TP;
    var mushroomConfusion = "n = "+total+"        Predicted Not Edible  Predicted Edible \n" + "Actual Not Edible    " + TN + "      " + FP + "\n" + "Actual Edible     " + FN + "      " + TP
    val mushroomRdd= sc.parallelize(Seq(mushroomConfusion))
//    mushroomRdd.saveAsTextFile(outputFileConfusion)
  }
}