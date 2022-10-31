import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.rand


object Sampler {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("PasswordAnalyzer")
      .master("local[*]")
      .getOrCreate()

    val passwordsFile = "result.txt"
    val outputFile = s"negative_samples.txt"

    spark
      .read
      .textFile(passwordsFile)
      .orderBy(rand())
      .sample(fraction = 0.00333)
      .write
      .text(outputFile)

    spark.stop()
  }
}
