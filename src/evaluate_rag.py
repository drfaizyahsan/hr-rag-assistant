from datasets import Dataset

# TODO: this is workaround for ragas 0.3.9 and langchain > 1
import sys
import types

_stub = types.ModuleType("langchain_community.chat_models.vertexai")


class ChatVertexAI:  # dummy to make the workaround
    pass


_stub.ChatVertexAI = ChatVertexAI
sys.modules["langchain_community.chat_models.vertexai"] = _stub
# end TODO: the workaround for ragas and langchain


from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from rag import BuildRagChain
import json
from langchain_ollama import ChatOllama
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class EvaluateRAGPipeline:
    def __init__(
        self,
        pth_to_ground_truth: str = "./data/raw/ragas_questions_short.json",
    ):

        # create RAG Pipeline object
        self.chain = BuildRagChain()

        # initialize embedding model in langchain format
        self.embeddings = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        )

        # make sure ollama has the mentioned model and is running
        # ollama pull model_name
        # ollama serve
        # after usage
        # ollama rm model_name
        # self.llm = LangchainLLMWrapper(
        #     ChatOllama(model="llama3.2:3b", temperature=0)
        # )
        self.llm = LangchainLLMWrapper(
            ChatOllama(model="llama3.1:8b", temperature=0)
        )

        # create eval pair list
        with open(pth_to_ground_truth) as fname:
            eval_pair_json = json.load(fname)

        self.eval_pairs = [
            (item["question"], item["ground_truth"])
            for item in eval_pair_json
        ]
        logging.info(f"eval_pairs: {self.eval_pairs}")

    def build_eval_dataset(
        self,
    ) -> Dataset:
        """

        :param chain:
        :param eval_pairs:
        :return:
        """
        questions, answers, contexts, ground_truths = [], [], [], []
        for q, gt in self.eval_pairs:
            questions.append(q)
            answers.append(self.chain.query(question=q))
            retrieved_docs = self.chain.retrieve(q)
            context = [doc["text"] for doc in retrieved_docs]
            contexts.append(context)
            ground_truths.append(gt)

        return Dataset.from_dict(
            {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths,
            }
        )

    def run_ragas(self, dataset: Dataset) -> dict:
        """

        :param dataset:
        :return:
        """

        result = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
            llm=self.llm,
            embeddings=self.embeddings,
            run_config=RunConfig(timeout=300, max_retries=3),
        )
        logging.info(f"result: {result}")
        return result.to_pandas().mean(numeric_only=True).to_dict()


er = EvaluateRAGPipeline()
logging.info(
    f"ragas metrics: {er.run_ragas(er.build_eval_dataset())}"
)
