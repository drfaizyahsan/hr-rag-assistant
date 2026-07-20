from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pyspark.sql.types import ArrayType, StringType
import pandas as pd
import io
from pypdf import PdfReader
import hashlib
from sentence_transformers import SentenceTransformer
import chromadb
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

spark = (
    SparkSession.builder.appName("hr-rag-assistant")
    .master("local[*]")
    .config("spark.sql.execution.arrow.pyspark.enabled", "True")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")


class Build_VectorStore:
    def __init__(
        self,
        path_to_pdf: str = "./data/raw/",
        path_to_save_chunks: str = "./data/processed/chunks",
        path_to_chromadb: str = "./chromadb",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        self.path_to_pdf = path_to_pdf
        self.path_to_save_chunks = path_to_save_chunks
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # initialize embedding model
        self.embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        # initialize ChromaDB
        self.client = chromadb.PersistentClient(path=path_to_chromadb)

    def ingest_pdfs(self) -> None:
        """
        ingest pdfs into overlapping chunks
        :return:
        """
        pdf_df = (
            spark.read.format("binaryFile")
            .option("pathGlobFilter", "*.pdf")
            .load(self.path_to_pdf)
        )

        pdf_df.show()

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
        extract_text_udf = F.udf(
            extract_text_from_bytes, StringType()
        )

        # read pdf content
        parsed_df = pdf_df.withColumn(
            "pdf_text", extract_text_udf("content")
        )

        parsed_df.show()
        logging.info(f"parsed_df: {parsed_df.count()}")

        # initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
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
                "file_name",
                F.element_at(F.split(F.col("path"), "/"), -1),
            )
        )

        chunked_df.show()
        logging.info(f"chunked_df lines: {chunked_df.count()}")

        # save chunked data
        chunked_df.write.mode("overwrite").parquet(
            self.path_to_save_chunks
        )

        return

    # TODO: embed each chunks with gemma4 or smaller model
    def ingest_vectors(
        self,
    ) -> None:
        """
        ingests vectors from chunks
        :return:
        """

        def generate_chunk_id(chunk_index: int) -> str:
            """
            create chunk id for each chunk in order to have
            idempotency keys
            :param chunk_index:
            :return:
            """
            k = f"{chunk_index}:"
            return hashlib.sha256(k.encode()).hexdigest()

        # load chunks
        chunked_df = spark.read.parquet(self.path_to_save_chunks)
        chunked_df.show()
        logging.info(f"chunked_df lines: {chunked_df.count()}")

        collection = self.client.get_or_create_collection(
            name="hr_rag_assistant"
        )

        # convert spark df to pandas batches
        batch_size = 128

        chunked_pandas = chunked_df.toPandas()

        for start in range(0, len(chunked_pandas), batch_size):
            batch = chunked_pandas.iloc[start : start + batch_size]
            documents = batch["chunk_text"].tolist()

            # generate embeddings
            embeddings = self.embedding_model.encode(
                documents, show_progress_bar=False
            ).tolist()

            logging.info(
                f"rows: {len(embeddings)}, cols: {len(embeddings[0])}"
            )

            # generate ids for idempotency keys
            ids = [
                generate_chunk_id(i)
                for i in range(start, start + len(batch))
            ]

            logging.info(f"ids: {ids}")

            # TODO: add metadata

            # TODO: store embedded vectors in FAISS

            # insert in chroma
            collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
            )

            logging.info(
                f"Inserted batch: {start}--{start + len(batch)}"
            )

        logging.info("vector ingestion completed")
        logging.info(f"Total vectors in chroma: {collection.count()}")


vs = Build_VectorStore()
vs.ingest_pdfs()
vs.ingest_vectors()

spark.stop()
