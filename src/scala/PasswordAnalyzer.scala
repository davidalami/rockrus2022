import org.apache.spark.sql.SparkSession
import scala.io
import org.apache.spark.sql.functions.{col, udf}


object PasswordAnalyzer {
  //  Taken from https://alvinalexander.com/source-code/scala-function-read-text-file-into-array-list/
  def readFile(filename: String): List[String] = {
    val bufferedSource = io.Source.fromFile(filename)
    val lines = (for (line <- bufferedSource.getLines()) yield line).toList
    bufferedSource.close
    lines
  }

  def contains(line: String, tokens: List[String]): Boolean = {
    tokens.exists(token => line.contains(token))
  }

  def intersectionCount(tokens: List[String])(line: String): Int = {
    tokens.count(token => line.contains(token))
  }

  def main(args: Array[String]) {
    val spark = SparkSession
      .builder
      .appName("PasswordAnalyzer")
      .master("local[*]")
      .getOrCreate()

    val tokensFile: String = "tokens.txt"
    val passwordsFile = "rockyou2021.txt"
    val outputFile = s"result.txt"
    val partitionValue = 3
    val tokens: List[String] = readFile(tokensFile)

    val ranker = intersectionCount(tokens) _
    val pwdRanker = udf((x: String) => ranker(x))
    spark.udf.register("pwdRanker", pwdRanker)

    spark
      .read
      .textFile(passwordsFile)
      .filter(!_.isEmpty())
      .filter(!_.forall(_.isDigit))
      .filter(pwd => contains(pwd, tokens))
      .distinct
      .withColumn("pwdRank", pwdRanker(col("value")))
      .filter(s"pwdRank >= $partitionValue")
      .drop(col("pwdRank"))
      .write
      .text(outputFile)

    spark.stop()

  }
}