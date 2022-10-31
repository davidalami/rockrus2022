import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.{StringType, StructType, StructField, FloatType}


object ScoreSort {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("PasswordAnalyzer")
      .master("local[*]")
      .getOrCreate()

    val passwordsFile = "scored_passwords.txt"
    val outputFile = s"sorted_result.txt"

    val schema = StructType(Array(
      StructField("score", FloatType),
      StructField("password", StringType)))

    spark
      .read
      .option("delimiter", "\t")
      .option("header", "false")
      .schema(schema)
      .csv(passwordsFile)
      .orderBy(col("score").desc)
      .drop(col("score"))
      .write
      .text(outputFile)

    spark.stop()
  }
}
