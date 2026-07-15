from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pyspark.sql.types import ArrayType, StringType
import pandas as pd
import io
from pypdf import PdfReader
import logging

spark = (
    SparkSession.builder.appName("hr-rag-assistant")
    .master("local[*]")
    .config("spark.sql.execution.arrow.pyspark.enabled", "True")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")


def build_vectorstore(
    path_to_pdf: str = "./data/raw/",
    chunk: int = 500,
    overlap: int = 50,
) -> None:
    """
    embeds pdf into vectors and store in vector database
    :param path_to_pdf:
    :return:
    """

    df = (
        spark.read.format("binaryFile")
        .option("pathGlobFilter", "*.pdf")
        .load(path_to_pdf)
    )

    df.show()

    # extract text from the byte array
    def extract_text_from_bytes(content_bytes):
        try:
            # wrap bytes into file-like stream
            pdf_stream = io.BytesIO(content_bytes)
            reader = PdfReader(pdf_stream)
            # get text from all pages
            text = "".join(
                [page.extract_text() for page in reader.pages]
            )
            return text
        except Exception as e:
            return f"Error parsing: {str(e)}"

    # create spark udf for reading pdfs
    extract_text_udf = F.udf(extract_text_from_bytes, StringType())

    # read pdf content
    parsed_df = df.withColumn("pdf_text", extract_text_udf("content"))

    parsed_df.show()
    logging.info(f"parsed_df: {parsed_df.count()}")

    # initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    # pandas udf to split texts
    @F.pandas_udf(ArrayType(StringType()))
    def split_into_chunks(text_series: pd.Series) -> pd.Series:
        """
        pandas udf to split text into overlapping chunks
        :param text_series:
        :return:
        """
        return text_series.apply(
            lambda text: (
                text_splitter.split_text(text)
                if text and isinstance(text, str)
                else []
            )
        )

    # apply pandas df for chunking
    chunked_df = (
        parsed_df.withColumn(
            "chunks", split_into_chunks(F.col("pdf_text"))
        )
        .withColumn("chunk_text", F.explode(F.col("chunks")))
        .withColumn(
            "file_name", F.element_at(F.split(F.col("path"), "/"), -1)
        )
    )

    chunked_df.show()
    print(chunked_df.count())

    # TODO: embed each chunks with gemma4 or smaller model
    # TODO: store embedded vectors in FAISS

    return


build_vectorstore()
