import chromadb
from sentence_transformers import SentenceTransformer
from langchain_ollama import ChatOllama
import logging
from typing import List, Dict


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class BuildRagChain:
    def __init__(
        self,
        path_to_chromadb: str = "./chromadb",
    ):
        # initialize embedding model
        self.embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        # initialize chromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=path_to_chromadb
        )

        # get chroma collection
        self.collection = self.chroma_client.get_collection(
            "hr_rag_assistant"
        )

        # make sure ollama has the mentioned model and is running
        # ollama pull model_name
        # ollama serve
        # after usage
        # ollama rm model_name
        self.llm = ChatOllama(model="llama3.2:3b", temperature=0)

        self.system_prompt = """
        You are an HR assistant. Use ONLY the context below to answer the question.
        If the answer is not in the context, say 'I don't have that information'.
        Context: {context}
        """

    # get similar docs
    def retrieve(
        self,
        question: str,
        top_k: int = 5,
    ) -> List[Dict]:
        """

        :param question:
        :param top_k:
        :return:
        """

        # get embedding for the question
        query_embedding = self.embedding_model.encode(
            question
        ).tolist()

        # now, get the similar docs from the vector database based on the above embedding
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "distances"],
        )

        retrieved_docs = []
        for doc, score in zip(
            results["documents"][0], results["distances"]
        ):
            retrieved_docs.append({"text": doc, "score": score})

        return retrieved_docs

    @staticmethod
    def build_context(retrieved_docs: List[Dict] = None) -> str:
        """

        :param retrieved_docs:
        :return:
        """

        context = []
        # append each documents' content
        for idx, doc in enumerate(retrieved_docs, start=1):
            context.append(
                f"""
                Document: {idx}
                
                Content: {doc["text"]}
                """
            )

        return "\n".join(context)

    # ask answer
    def query(self, question: str) -> str:
        """

        :param question:
        :return:
        """

        logging.info("creating prompt")

        # first get similar docs
        docs = self.retrieve(question)

        # build context using similar docs
        context = self.build_context(docs)

        # create prompt
        prompt = self.system_prompt.format(context=context)

        logging.info(f"prompt: {prompt}")

        # get response
        response = self.llm.invoke(
            [
                ("system", prompt),
                ("human", question),
            ]
        )

        return response.content


rc = BuildRagChain()
given_question = "How many sick leaves are allowed?"
logging.info(
    f"question: {given_question}\nresponse: {rc.query(question=given_question)}"
)
