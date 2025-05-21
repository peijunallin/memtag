import time
import os

os.environ["HF_HOME"] = [REDACTED]
os.environ["TRANSFORMERS_CACHE"] = [REDACTED]
os.environ["HF_DATASETS_CACHE"] = [REDACTED]
import torch

import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import json
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset,Dataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from transformers.utils import PaddingStrategy


import numpy as np
from transformers import AutoTokenizer, AutoModel
from safetensors.torch import load_file
from argparse import ArgumentParser
import networkx as nx
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import HuggingFaceBgeEmbeddings
from typing import List, Tuple

import ast
from peft import PeftConfig, PeftModel # Add these lines
import torch.nn as nn
import sys
from langchain_openai import OpenAIEmbeddings
import pickle
import argparse
import torch

from prompts import *

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pandas as pd

sys.path.append(os.path.abspath(
    os.path.join(os.getcwd(), '..')))
from helper_functions import *

from utils import rewrite_query
import numpy as np
from loguru import logger
from langchain_core.pydantic_v1 import BaseModel, Field
from tqdm import tqdm

from prompts import *
import tiktoken
import time
import re
from transformers import AutoTokenizer, AutoModel,AutoModelForCausalLM
from safetensors.torch import load_file
from generator import load_llm_tokenizer_and_model, Generator

from sentence_transformers import SentenceTransformer


class RatingScore(BaseModel):
    document: int
    relevance_score: float = Field(..., description="The relevance score of a document to a query.")


# Define a MultiRatingScore schema to handle multiple ratings
class MultiRatingScore(BaseModel):
    ratings: List[RatingScore] = Field(..., description="A list of document relevance scores.")




class MemoBox:
    def __init__(self, dataset, tokenizer, weights_path,reader_model=None, reader=None):
        self.llm = ChatOpenAI(temperature=0,
                              model_name="gpt-4o-mini",
                              max_tokens=4096)
        # self.embedding_model = OpenAIEmbeddings(model='text-embedding-3-small')
        # self.embedding_model = SentenceTransformer("Alibaba-NLP/gte-Qwen2-1.5B-instruct", trust_remote_code=True)
        self.embedding_model = HuggingFaceBgeEmbeddings(
            cache_folder=[REDACTED],
            model_name="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
            model_kwargs={"trust_remote_code": True}
        )
        # In case you want to reduce the maximum length:
        # self.embedding_model.max_seq_length = 8192
        self.atomic_fact_graph = nx.Graph()  # self.atomic_fact_graph.nodes(data=True): (i, {"text": , "key_elements"})
        # self.token_encoder = tiktoken.encoding_for_model("gpt-4o-mini")
        self.token_encoder = tokenizer
        self.reader_model = reader_model
        self.reader = Generator(reader_tokenizer, reader_llm, max_length=32768, max_new_tokens=64)
        self.dataset = dataset
        self.weights_path = weights_path
        self.device = f"cuda" if torch.cuda.is_available() else "cpu"
        self.attributes = ["triples", "atomic facts", "summary", "chunks"]
        self.weights_path = weights_path


    
        # Set default weights path if not provided
        if self.weights_path is None:
            raise Exception("There is no regression weights path")
        else:
            self.weights_path = self.weights_path

        logger.info(f"Loading  memory regression linear layer weights model from {self.weights_path}...")



        #self.lora_weights_path = os.path.join(weights_path,"lora_final")
        self.lora_weights_path = os.path.join(weights_path,"lora_epoch_3")

        # ----------- LOADING LoRA MODEL + REGRESSION HEAD for WEIGHT PREDICTION ONLY ----------- #
        logger.info(f"Loading backbone with LoRA for reward model...")
        # config = PeftConfig.from_pretrained(self.lora_weights_path)
        # rm_backbone = AutoModelForCausalLM.from_pretrained(
        #     config.base_model_name_or_path,
        #     trust_remote_code=True,
        #     attn_implementation="flash_attention_2",
        #     torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        # )

        rm_backbone = reader
        # rm_backbone = reader_model


        self.rm = PeftModel.from_pretrained(rm_backbone, self.lora_weights_path)

        self.rm.eval().to(self.device)
        self.rm_tokenizer = tokenizer

        # Load regression head
        logger.info(f"Loading saved regression head weights from {weights_path}...")
        self.regression_head = nn.Linear(self.rm.config.hidden_size, 4)
        #self.regression_head.load_state_dict(torch.load(os.path.join(weights_path, "regression_head_final.pt"), map_location="cpu"))
        self.regression_head.load_state_dict(torch.load(os.path.join(weights_path, "regression_head_epoch_3.pt"), map_location="cpu"))
        
        self.regression_head = self.regression_head.to(dtype=torch.bfloat16, device=self.device)
        self.regression_head.eval()


        # ---- ADDED FOR PARAMETER COUNTING ----
        self.regression_head_params = sum(p.numel() for p in self.regression_head.parameters())
        logger.info(f"Regression head parameters: {self.regression_head_params}")

        if self.rm:  # self.rm is the PeftModel
            try:
                # get_nb_trainable_parameters() returns (trainable_params, all_params_in_peft_model)
                lora_trainable_params, _ = self.rm.get_nb_trainable_parameters()
                self.lora_params = lora_trainable_params
            except Exception as e:
                logger.warning(f"Could not get LoRA parameters using get_nb_trainable_parameters: {e}. Falling back to manual count.")
                self.lora_params = 0
                for name, param in self.rm.named_parameters():
                    if param.requires_grad: # PEFT marks only adapter params as trainable
                         self.lora_params += param.numel()
            logger.info(f"LoRA block parameters: {self.lora_params}")
            # self.rm.print_trainable_parameters() # Optional: for detailed breakdown

            # Log the base model name for PEFT if available
            peft_base_model_name = "N/A"
            if hasattr(self.rm, 'config') and hasattr(self.rm.config, 'base_model_name_or_path') and self.rm.config.base_model_name_or_path:
                peft_base_model_name = self.rm.config.base_model_name_or_path
            elif rm_backbone and hasattr(rm_backbone, 'config') and hasattr(rm_backbone.config, '_name_or_path'):
                 peft_base_model_name = rm_backbone.config._name_or_path
            logger.info(f"Router's LoRA is applied to PEFT base model: {peft_base_model_name}")

        else:
            self.lora_params = 0
            logger.warning("self.rm (PeftModel) is not initialized. LoRA parameters will be 0.")
        # ---- END OF ADDED PARAMETER COUNTING ----





        self.source_descriptions = {
            "2wikimultihopqa": "Passages from two or more Wikipedia pages.",
[REDACTED]
            "hotpotqa": "Two Wikipedia pages, typically their introductory paragraphs.",
            "musique": "A broad collection of Wikipedia paragraphs, demanding hops between several of these Wikipedia paragraphs and integration of information across them.",
[REDACTED]
            "quality": "A general question answering dataset that involves articles about a wide range of topics." # Added a description for 'quality'
            # Add other dataset names used in your args.dataset choices if they map to different contexts
        }

    def predict_weights(self, question, norm_method,norm_params):
        """
        Predict attribute weights for a given question.

        Args:
            question: The user's question
            norm_method: Normalization method to use

        Returns:
            dict: Dictionary mapping attribute names to normalized weights
        """
                        # choices=['hotpotqa',
                        #          'musique',
                        #          '2wikimultihopqa',
[REDACTED]
[REDACTED]
                        #          'quality'],

        document_description = self.source_descriptions.get(
            self.dataset, "General document context." # Fallback description
        )

        formatted_prompt_content = PROMPT_TEMPLATE_FOR_WEIGHT_PREDICTION.format(
            context=document_description,
            question=question
        )


        # Format the question as a conversation
        # messages = [{"role": "system", "content": "You are a helpful assistant."},
        #             {"role": "user", "content": question}]
        messages = [
            {'role': 'system', 'content': 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'},
            {"role": "user", "content": formatted_prompt_content}]



        conv_formatted = self.rm_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        conv_tokenized = self.rm_tokenizer(conv_formatted, return_tensors="pt").to(self.device)


        with torch.no_grad():
            # outputs = self.rm.model(input_ids=conv_tokenized['input_ids'], attention_mask=conv_tokenized['attention_mask'], return_dict=True)
            # hidden_states = outputs.last_hidden_state

            # MODIFICATION START
            outputs = self.rm( # Call self.rm directly
                input_ids=conv_tokenized['input_ids'],
                attention_mask=conv_tokenized['attention_mask'],
                return_dict=True,
                output_hidden_states=True  # Explicitly request all hidden states
            )
            hidden_states = outputs.hidden_states[-1]

            # Pick last non-padding token
            lengths = conv_tokenized['attention_mask'].sum(dim=1) - 1
            last_token_hidden_states = hidden_states[torch.arange(hidden_states.size(0)), lengths]

            # Predict scores
            predicted_scores = self.regression_head(last_token_hidden_states)

        results = {attr: predicted_scores[0, i].item() for i, attr in enumerate(self.attributes)}

        logger.info(f"Predicted attribute weights before norm: {results}")

        normalized = self.normalize_scores(results, method=norm_method,params=norm_params)
        logger.info(f"Predicted attribute weights after norm: {normalized}")
        normalized['atomic_facts'] = normalized.pop('atomic facts')
        return normalized

    def normalize_scores(self, results, method, params):
        """
        Normalize the attribute scores so that they sum to 1.

        Args:
            results: Dictionary mapping attribute names to predicted scores
            method: Normalization method to use
            params: Additional parameters for specific normalization methods

        Returns:
            dict: Dictionary with normalized scores that sum to 1
        """
        normalized = {}
        scores = np.array(list(results.values()))

        # Apply the requested normalization method
        if method == "minmax":
            # Min-Max normalization: (x - min) / (max - min)
            min_val = np.min(scores) if params is None or "min" not in params else params["min"]
            max_val = np.max(scores) if params is None or "max" not in params else params["max"]
            # Avoid division by zero
            if max_val == min_val:
                normalized_scores = np.ones_like(scores) * 0.5
            else:
                normalized_scores = (scores - min_val) / (max_val - min_val)
        elif method == "sigmoid":
            # Sigmoid normalization: 1 / (1 + exp(-x))
            scale = 1.0 if params is None or "scale" not in params else params["scale"]
            normalized_scores = 1 / (1 + np.exp(-scale * scores))
        elif method == "softmax":
            # Softmax normalization already sums to 1, so we can return it directly
            #temp = 1.0 if params is None or "temperature" not in params else params["temperature"]
            temp = 1.0 if params is None else params
            exp_scores = np.exp(scores / temp)
            normalized_scores = exp_scores / np.sum(exp_scores)
            # Create normalized results dictionary and return
            for i, (attr, _) in enumerate(results.items()):
                normalized[attr] = float(normalized_scores[i])
            return normalized
        elif method == "tanh":
            # Tanh normalization: (tanh(x) + 1) / 2
            scale = 0.5 if params is None else params
            normalized_scores = (np.tanh(scale * scores) + 1) / 2
        elif method == "exp":
            # Exponential normalization: 1 - exp(-x)
            scale = 0.5 if params is None or "scale" not in params else params["scale"]
            normalized_scores = 1 - np.exp(-scale * scores)
        elif method == "log":
            # Logarithmic normalization: log(1 + x) / log(1 + max)
            offset = 1.0 if params is None or "offset" not in params else params["offset"]
            # Ensure all values are positive by adding offset
            adjusted_scores = scores + offset
            max_val = np.max(adjusted_scores)
            normalized_scores = np.log(adjusted_scores) / np.log(max_val)
        elif method == "percentile":
            # Percentile-based normalization
            from scipy import stats
            normalized_scores = np.array([stats.percentileofscore(scores, s) / 100 for s in scores])
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        # Ensure non-negative values (some methods might produce negative values)
        normalized_scores = np.maximum(normalized_scores, 0)

        # Now ensure the scores sum to 1 by dividing by their sum
        sum_scores = np.sum(normalized_scores)
        if sum_scores > 0:  # Avoid division by zero
            normalized_scores = normalized_scores / sum_scores
        else:
            # If all scores are 0, distribute evenly
            normalized_scores = np.ones_like(scores) / len(scores)

        # Create normalized results dictionary
        for i, (attr, _) in enumerate(results.items()):
            normalized[attr] = float(normalized_scores[i])

        return normalized
    
    
    def load_dataset(self, memory_reprs, dataset, qa_order, num_noise_docs=0):
        """
        Load multiple memory representations

        Args:
            memory_reprs: List of memory representations to load
            dataset: Dataset name
            qa_order: Question order
            num_noise_docs: Number of noise documents to add
        """
        main_path = f"./datasets/{dataset}/"
        self.noise_memory_units = []
        self.noise_memory_vector_stores = []

        # Initialize dictionaries to store different memory representations
        self.memory_units_dict = {}
        self.memory_units2docs_dict = {}
        self.memory_units_vector_store_dict = {}

        # Load each memory representation
        for memory_repr in memory_reprs:
            logger.info(f"|> Loading memory representation: {memory_repr}")

            if num_noise_docs == 0:
                memory_units, memory_units2docs = self._load_memory_units(main_path, memory_repr, qa_order)
                logger.info(f"|> Loaded {len(memory_units)} memory units for {memory_repr}")
                memory_units_vector_store = self._load_vector_store(main_path, memory_repr, qa_order)
                logger.info(
                    f"|> Loaded {memory_units_vector_store.index.ntotal} memory units in vector store for {memory_repr}")
            else:
                memory_units, memory_units2docs = self._load_noise_memory_units(main_path, memory_repr, qa_order,
                                                                                num_noise_docs)
                logger.info(f"|> Loaded {len(memory_units)} memory units for {memory_repr}")
                memory_units_vector_store = self._load_noise_vector_store(main_path, memory_repr, qa_order,
                                                                          num_noise_docs)
                logger.info(
                    f"|> Loaded {memory_units_vector_store.index.ntotal} memory units in vector store for {memory_repr}")

            # Store in dictionaries
            self.memory_units_dict[memory_repr] = memory_units
            self.memory_units2docs_dict[memory_repr] = memory_units2docs
            self.memory_units_vector_store_dict[memory_repr] = memory_units_vector_store

    def _load_memory_units(self, main_path, memory_repr, order):
        """Load memory units for a specified order."""
        logger.info(f"|> Loading memory units for order {order}")
        memory_units_path = os.path.join(main_path, f"{memory_repr}/memory_units/{order}.pkl")
        _memory_units = pickle.load(open(memory_units_path, "rb"))
        memory_units = []
        memory_units2docs = {}
        for unit in _memory_units:
            memory_units.extend(unit[memory_repr])
            cluster_id = unit['cluster_id']
            for _unit in unit[memory_repr]:
                text = _unit['text']
                memory_units2docs[text] = {'cluster_id': cluster_id}
        return memory_units, memory_units2docs

    def _load_noise_memory_units(self, main_path, memory_repr, order, num_noise_docs):
        """Load memory units for a specified order."""
        logger.info(f"|> Loading memory units for order {order}")
        memory_units_path = os.path.join(main_path,
                                         f"{memory_repr}/memory_units_noise_docs_{num_noise_docs}/{order}.pkl")
        _memory_units = pickle.load(open(memory_units_path, "rb"))
        memory_units = []
        memory_units2docs = {}
        for unit in _memory_units:
            memory_units.extend(unit[memory_repr])
            cluster_id = unit['cluster_id']
            for _unit in unit[memory_repr]:
                text = _unit['text']
                memory_units2docs[text] = {'cluster_id': cluster_id}
        return memory_units, memory_units2docs

    def _load_vector_store(self, main_path, memory_repr, order):
        """Load vector store for a specified order."""
        logger.info(f"|> Loading vector store for order {order}")
        vector_store_path = os.path.join(main_path, f"{memory_repr}/vector_stores/vector_store_{order}")
        return FAISS.load_local(vector_store_path, self.embedding_model, allow_dangerous_deserialization=True)

    def _load_noise_vector_store(self, main_path, memory_repr, order, num_noise_docs):
        """Load vector store for a specified order."""
        logger.info(f"|> Loading vector store for order {order}")
        vector_store_path = os.path.join(main_path,
                                         f"{memory_repr}/vector_stores_noise_docs_{num_noise_docs}/vector_store_{order}")
        return FAISS.load_local(vector_store_path, self.embedding_model, allow_dangerous_deserialization=True)

    def _add_noise_memory_units(self, main_path, memory_repr, qa_order, num_noise_docs):
        """Add specified number of noise memory units and vector stores."""
[REDACTED]
        logger.info(f"|> Adding {num_noise_docs} noise documents")
        for noise_order in tqdm(noise_orders, desc="Adding noise memory units"):
            logger.info(f"|> Adding noise memory units for order {noise_order}")
            try:
                noise_memory_units, _ = self._load_memory_units(main_path, memory_repr, noise_order)
                # noise_vector_store = self._load_vector_store(main_path, memory_repr, noise_order)
                # self.noise_memory_vector_stores.append(noise_vector_store)
                # self.memory_units_vector_store.merge_from(noise_vector_store)
                self.memory_units.extend(noise_memory_units)
            except:
                logger.info(f"|> Error merging vector store for order {noise_order}")

    def build_vector_store(self, memory_units):
        memory_units_docs = []
        memory_units_count = 0
        for memory_unit in memory_units:
            doc = Document(page_content=memory_unit['text'], metadata={'global_id': memory_units_count})
            memory_units_docs.append(doc)
            memory_units_count += 1
        new_vector_store = FAISS.from_documents(memory_units_docs, self.embedding_model)
        return new_vector_store

    
    
    
    
    
    
    
    # Retrieval methods-----Retrieval methods-----Retrieval methods------Retrieval methods----Retrieval methods-----Retrieval methods----Retrieval methods
    def retrieve_top_k_memory_units(self, query: str, top_k: int):
        return self.memory_units_vector_store.similarity_search_with_score(query, k=top_k)



    def hm_retrieve_top_k_memory_units(self,
                                       query: str,
                                       top_k: int,
                                       memory_repr_weights: List[float] = None,
                                       memory_reprs: List[str] = None):
        """ 
        Retrieve a mixed set of top k memory units based on weights for different memory representations

        Args:
            query: Query string
            top_k: Total number of memory units to retrieve
            memory_repr_weights: List of weights for each memory representation (must sum to 1.0)
            memory_reprs: List of memory representations to use (if None, use all available)

        Returns:
            List of (document, score) tuples from various memory representations
        """
        if memory_reprs is None:
            memory_reprs = list(self.memory_units_vector_store_dict.keys())

        if memory_repr_weights is None:
            # Equal weights if not specified
            memory_repr_weights = [1.0 / len(memory_reprs)] * len(memory_reprs)

        # Ensure weights sum to 1.0
        if abs(sum(memory_repr_weights) - 1.0) > 1e-6:
            logger.warning(f"Memory representation weights do not sum to 1.0. Normalizing weights.")
            total_weight = sum(memory_repr_weights)
            memory_repr_weights = [w / total_weight for w in memory_repr_weights]

        # Normalize and quantize weights to determine how many documents to retrieve from each memory representation
        quantized_counts = self._quantize_weights(memory_repr_weights, top_k)

        logger.info(
            f"Retrieving from multiple memory representations with weights: {dict(zip(memory_reprs, memory_repr_weights))}")
        logger.info(f"Quantized document counts: {dict(zip(memory_reprs, quantized_counts))}")

        # Retrieve documents from each memory representation according to quantized counts
        final_results = []

        for i, memory_repr in enumerate(memory_reprs):
            count = quantized_counts[i]
            if count > 0:
                results = self.memory_units_vector_store_dict[memory_repr].similarity_search_with_score(query, k=count)
                for doc, score in results:
                    if not hasattr(doc.metadata, 'memory_repr'):
                        doc.metadata['memory_repr'] = memory_repr
                    final_results.append((doc, score))


        return final_results[:top_k]

    def _quantize_weights(self, weights: List[float], total_count: int) -> List[int]:
        """
        Convert weights to integer counts that sum to total_count

        Args:
            weights: List of weights (should sum to 1.0)
            total_count: Total number of items to distribute

        Returns:
            List of integer counts
        """
        # Initial allocation based on weights
        float_counts = [w * total_count for w in weights]
        int_counts = [int(c) for c in float_counts]

        # Distribute remaining items based on fractional parts
        remaining = total_count - sum(int_counts)
        if remaining > 0:
            fractions = [(i, c - int(c)) for i, c in enumerate(float_counts)]
            fractions.sort(key=lambda x: x[1], reverse=True)

            for i in range(remaining):
                int_counts[fractions[i % len(fractions)][0]] += 1

        return int_counts

    def _deduplicate_results(self, results: List[Tuple], memory_reprs: List[str], weights: List[float]) -> List[Tuple]:
        """
        Deduplicate results, preferring results from memory representations with higher weights

        Args:
            results: List of (document, score, memory_repr) tuples
            memory_reprs: List of memory representation names
            weights: List of weights for each memory representation

        Returns:
            Deduplicated list of (document, score, memory_repr) tuples
        """
        # Sort by weight of memory representation (descending) and then by score (ascending)
        repr_to_weight = {memory_repr: weight for memory_repr, weight in zip(memory_reprs, weights)}
        sorted_results = sorted(results, key=lambda x: (-repr_to_weight[x[2]], x[1]))

        # Deduplicate based on content
        seen_content = set()
        deduplicated = []

        for doc, score, memory_repr in sorted_results:
            content = doc.page_content
            if content not in seen_content:
                seen_content.add(content)
                deduplicated.append((doc, score, memory_repr))

        return deduplicated

    def get_memory_units_texts(self, memory_unit_ids_to_scores: dict, memory_unit_ids_to_unit: dict):
        sorted_memory_unit_ids_scores = sorted(memory_unit_ids_to_scores.items(), key=lambda x: x[1])
        memory_units_texts = [memory_unit_ids_to_unit[unit_id].page_content for unit_id, _ in
                              sorted_memory_unit_ids_scores]
        return memory_units_texts

    def get_corresponding_raw_documents_by_memory_units(self, memory_units_text):
        raw_documents_ids = [self.memory_units2docs[text]['cluster_id'] for text in memory_units_text]
        raw_documents_ids = list(set(raw_documents_ids))
        return [self.raw_documents[ids] for ids in raw_documents_ids]


    def hm_iterative_retrieve_top_k_memory_units(self, memory_reprs, memory_repr_weights, query: str, top_k: int, top_t: int = 10, num_turns: int = 4):   
                                                                                
        thoughts = []
        memory_unit_ids_to_scores, memory_unit_ids_to_unit = {}, {}

        # Get predicted memory weights
        # memory_repr_weights_dict = self.predict_weights(query)
        # memory_reprs = list(memory_repr_weights_dict.keys())
        # memory_repr_weights = list(memory_repr_weights_dict.values())

        for _ in tqdm(range(num_turns)):
            retrieval_query = query if len(thoughts) == 0 else query + " " + thoughts[-1]
            retrieval_result = self.hm_retrieve_top_k_memory_units(
                query=retrieval_query,
                top_k=top_t,
                memory_repr_weights=memory_repr_weights,
                memory_reprs=memory_reprs
            )
            for unit, score in retrieval_result:
                unit_id = unit.metadata['global_id']
                memory_unit_ids_to_scores[unit_id] = min(memory_unit_ids_to_scores.get(unit_id, 1e9), score)
                memory_unit_ids_to_unit[unit_id] = unit
            memory_units_texts = self.get_memory_units_texts(memory_unit_ids_to_scores, memory_unit_ids_to_unit)
            thought = self.generate_iterative_retrieval_thought(query=query, context=memory_units_texts)
            thoughts.append(thought.strip())

        sorted_memory_unit_ids_scores = sorted(memory_unit_ids_to_scores.items(), key=lambda x: x[1])
        retrieved_memory_units = [memory_unit_ids_to_unit[unit_id] for unit_id, _ in sorted_memory_unit_ids_scores[:top_k]]
        return retrieved_memory_units
    
    # Query methods
    def query_with_only_memory_units(self, query, maximum_tokens):
        context = [ele['text'] for ele in self.memory_units]
        if maximum_tokens != -1 and maximum_tokens > 0:
            context = self.trim_context_by_tokens(context, maximum_tokens)
        context = self.get_generate_answer_context(context)
        return self.generate_answer(query=query, context=context)

    def query_with_top_k_memory_units(self, query, top_k: int, top_t: int, num_turns: int,
                                      answer_with_memory_units: int, maximum_tokens: int, use_iterative_retrieval: int, memory_reprs, memory_repr_weights
                                      ):
        
        if use_iterative_retrieval:
            retrieved_memory_units = self.hm_iterative_retrieve_top_k_memory_units(memory_reprs,
                                                                                memory_repr_weights,
                                                                                query,
                                                                                top_k,
                                                                                top_t,
                                                                                num_turns,
                                                                                
                                                                                )
            top_k_memory_units_text = [ele.page_content for ele in retrieved_memory_units]
        else:
            retrieved_top_k_memory_units = self.retrieve_top_k_memory_units(query, top_k)
            top_k_memory_units_text = [ele.page_content for ele, _ in retrieved_top_k_memory_units]
        # if maximum_tokens != -1 and maximum_tokens > 0:
        #     top_k_memory_units_text = self.trim_context_by_tokens(top_k_memory_units_text, maximum_tokens)
        if answer_with_memory_units:
            logger.info(f"We are going to answer with memory units")
            context = self.get_generate_answer_context(top_k_memory_units_text)
        else:
            logger.info(f"We are going to answer with documents")
            corresponding_raw_documents = self.get_corresponding_raw_documents_by_memory_units(top_k_memory_units_text)
            context = self.get_generate_answer_context(corresponding_raw_documents)
        return self.generate_answer(query=query, context=context)



# TODO------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def hm_query_with_top_k_memory_units(self,
                                         query: str,
                                         top_k: int,
                                         memory_reprs: List[str] = ['triples', 'summary', 'chunks', 'atomic_facts'],
                                         memory_repr_weights: List[float] = [0.1, 0.2, 0.3, 0.4]):
        """
        Query using a weighted mix of memory representations

        Args:
            query: Query string
            top_k: Total number of memory units to retrieve
            memory_reprs: List of memory representations to use
            memory_repr_weights: Weights for each memory representation
        """
        # Retrieve mixed memory units
        retrieved_memory_units = self.hm_retrieve_top_k_memory_units(
            query=query,
            top_k=top_k,
            memory_reprs=memory_reprs,
            memory_repr_weights=memory_repr_weights
        )

        # Extract text from all documents
        top_k_memory_units_text = [ele.page_content for ele, _ in retrieved_memory_units]

        # Generate answer
        context = self.get_generate_answer_context(top_k_memory_units_text)
        return self.generate_answer(query=query, context=context)




    def query_with_iterative_retrieve_top_k_memory_units(self, query: str, top_k: int, top_t: int = 10,
                                                         num_turns: int = 4, maximum_tokens: int = -1):
        thoughts = []
        memory_unit_ids_to_scores, memory_unit_ids_to_unit = {}, {}
        for _ in tqdm(range(num_turns)):
            retrieval_query = query if len(thoughts) == 0 else query + " " + thoughts[-1]
            retrieval_result = self.retrieve_top_k_memory_units(retrieval_query, top_t)
            for unit, score in retrieval_result:
                unit_id = unit.metadata['global_id']
                memory_unit_ids_to_scores[unit_id] = min(memory_unit_ids_to_scores.get(unit_id, 1e9), score)
                memory_unit_ids_to_unit[unit_id] = unit
            memory_units_texts = self.get_memory_units_texts(memory_unit_ids_to_scores, memory_unit_ids_to_unit)
            thought = self.generate_iterative_retrieval_thought(
                query=query, context=memory_units_texts
            )
            thoughts.append(thought.strip())
        sorted_memory_unit_ids_scores = sorted(memory_unit_ids_to_scores.items(), key=lambda x: x[1])
        retrieved_memory_units = [memory_unit_ids_to_unit[unit_id] for unit_id, _ in
                                  sorted_memory_unit_ids_scores[:top_k]]
        if maximum_tokens != -1 and maximum_tokens > 0:
            retrieved_memory_units = self.trim_context_by_tokens(retrieved_memory_units, maximum_tokens)
        context = self.get_generate_answer_context(retrieved_memory_units)
        return self.generate_answer(query=query, context=context)

    def query_with_top_k_with_rerank_top_r_memory_units(self, query: str, top_k: int, top_r: int, top_t: int,
                                                        num_turns: int, answer_with_memory_units: int,
                                                        maximum_tokens: int, use_iterative_retrieval: int,
                                                        memory_reprs,
                                                        memory_repr_weights):
        if use_iterative_retrieval:
            retrieved_memory_units = self.hm_iterative_retrieve_top_k_memory_units(
                                                                                memory_reprs,
                                                                                memory_repr_weights,
                                                                                query,
                                                                                top_k,
                                                                                top_t,
                                                                                num_turns,
                                                    
                                                                                )
            top_k_memory_units = [ele for ele in retrieved_memory_units]
        else:
            retrieved_top_k_memory_units = self.hm_retrieve_top_k_memory_units(query, top_k,memory_reprs=memory_reprs,memory_repr_weights=memory_repr_weights,)
            top_k_memory_units = [ele for ele, _ in retrieved_top_k_memory_units]


        reranked_top_r_memory_units = self.rerank_documents_batch(query,
                                                                  top_k_memory_units,
                                                                  top_r=top_r)
        top_r_memory_units_texts = [fact.page_content for fact in reranked_top_r_memory_units]
        if answer_with_memory_units:
            logger.info(f"We are going to answer with memory units")
            # if maximum_tokens != -1 and maximum_tokens > 0:
            #     top_r_memory_units_texts = self.trim_context_by_tokens(top_r_memory_units_texts, maximum_tokens)
            context = self.get_generate_answer_context(top_r_memory_units_texts)
        else:
            logger.info(f"We are going to answer with documents")
            corresponding_raw_documents = self.get_corresponding_raw_documents_by_memory_units(top_r_memory_units_texts)
            # if maximum_tokens != -1 and maximum_tokens > 0:
            #     corresponding_raw_documents = self.trim_context_by_tokens(corresponding_raw_documents, maximum_tokens)
            context = self.get_generate_answer_context(corresponding_raw_documents)
        return self.generate_answer(query=query, context=context)

    def query_with_cluster_with_raw_documents(self, query: str):
        all_raw_documents = []
        for ids in self.summaries_ids:
            all_raw_documents.append(self.raw_documents[ids])
        context = self.get_generate_answer_context(all_raw_documents)
        return self.generate_answer(query=query, context=context)

    def query_with_memory_units_with_common_cluster(self, query: str, top_k: int, top_s: int,
                                                    answer_with_memory_units: int, maximum_tokens: int,
                                                    top_t: int, num_turns: int, use_iterative_retrieval: int):
        top_s_cluster_from_fasiss = self.summary_vector_store.similarity_search(query, k=top_s)
        top_s_cluster_id_from_faiss = [ele.metadata['cluster_id'] for ele in top_s_cluster_from_fasiss]
        if use_iterative_retrieval:
            retrieved_memory_units = self.iterative_retrieve_top_k_memory_units(query,
                                                                                top_k,
                                                                                top_t,
                                                                                num_turns)
            top_k_memory_units = [ele for ele in retrieved_memory_units]
            cluster_id_from_memory_units = list(set([fact.metadata['cluster_id'] for fact in top_k_memory_units]))
        else:
            retrieved_top_k_memory_units = self.retrieve_top_k_memory_units(query, top_k)
            top_k_memory_units = [ele for ele, _ in retrieved_top_k_memory_units]
            cluster_id_from_memory_units = list(set([fact.metadata['cluster_id'] for fact, _ in top_k_memory_units]))
        common_cluster_ids = list(set(top_s_cluster_id_from_faiss) & set(cluster_id_from_memory_units))
        if answer_with_memory_units:
            filtered_memory_units = [ele for ele, _ in top_k_memory_units if
                                     ele.metadata['cluster_id'] in common_cluster_ids]
            filtered_memory_units_text = [ele.page_content for ele in filtered_memory_units]
            # if maximum_tokens != -1 and maximum_tokens > 0:
            #     filtered_memory_units_text = self.trim_context_by_tokens(filtered_memory_units_text, maximum_tokens)
            context = self.get_generate_answer_context(filtered_memory_units_text)
        else:
            raw_documents = [self.raw_documents[ids] for ids in common_cluster_ids]
            # if maximum_tokens != -1 and maximum_tokens > 0:
            #     raw_documents = self.trim_context_by_tokens(raw_documents, maximum_tokens)
            context = self.get_generate_answer_context(raw_documents)
        return self.generate_answer(query=query, context=context)

    # Text generation and parsing methods
    def generate_iterative_retrieval_thought(self, query: str, context: List[str]):
        if self.reader_model == "gpt-4o-mini":
            thought_prompt = PromptTemplate(input_variables=["context", "question"],
                                            template=iterative_retrieval_template)
            thought_chain = thought_prompt | self.llm | StrOutputParser()
            input_data = {
                "context": "\n\n".join(["{}. {}".format(i + 1, text) for i, text in enumerate(context)]),
                "question": query
            }
            thought = thought_chain.invoke(input_data)
        else:
            if "instruct" in self.reader_model:
                instruction = iterative_retrieval_instruction
                input_text = iterative_retrieval_input_template.format(
                    context="\n\n".join(["{}. {}".format(i + 1, text) for i, text in enumerate(context)]),
                    question=query
                )
                prompt_chat_template = self.reader.get_generator_prompts_chat_format([instruction], [input_text])
                inputs = self.reader.tokenizer_encode_chat_format(prompt_chat_template)
            else:
                raise NotImplementedError(
                    f"generate_iterative_retrieval_thought using f{self.reader_model} is not implemented!")
            generated_token_ids, _ = self.reader.generate(inputs)
            generated_text = self.reader.tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)[0]
            thought = generated_text
        return thought

    def generate_answer(self, query, context):
        if self.reader_model == "gpt-4o-mini":
            if self.dataset == "quality":
                answer_prompt = PromptTemplate(input_variables=["query", "context"],
                                               template=answer_prompt_template_for_quality_dataset)
            else:
                answer_prompt = PromptTemplate(input_variables=["query", "context"], template=answer_prompt_template)
            answer_chain = answer_prompt | self.llm | StrOutputParser()
            input_data = {"query": query, "context": context}
            answer = answer_chain.invoke(input_data)
        else:
            if "instruct" in self.reader_model:
                instruction = answer_prompt_instruction_for_quality_dataset if self.dataset == "quality" else answer_prompt_instruction
                input_template = answer_prompt_input_template_for_quality_dataset if self.dataset == "quality" else answer_prompt_input_template
                input_text = input_template.format(query=query, context=context)
                logger.info(f"|> The input text is:  {input_text}")

                prompt_chat_template = self.reader.get_generator_prompts_chat_format([instruction], [input_text])
                logger.info(f"|> The final input in template is:  {prompt_chat_template}")
                

                #----------
                text_strs = self.reader.tokenizer.apply_chat_template(prompt_chat_template, tokenize=False, add_generation_prompt=True)
                if isinstance(text_strs, str):
                    text_strs = [text_strs]

                for i, raw_text in enumerate(text_strs):
                    num_tokens_before = len(self.reader.tokenizer(raw_text, add_special_tokens=False).input_ids)
                    logger.info(f"[Before Tokenization #{i}] Raw text length: {len(raw_text)} characters, approx {num_tokens_before} tokens")
                    logger.debug(f"[Raw Text #{i}]: {raw_text}")

                #----------  


                inputs = self.reader.tokenizer_encode_chat_format(prompt_chat_template)



                #sadadasdasdsadasdsa
                for i, input_ids in enumerate(inputs["input_ids"]):
                    actual_tokens = input_ids.shape[-1]
                    if actual_tokens >= self.reader.max_length:
                        logger.warning(f"[Truncation Detected #{i}] Input was truncated to {actual_tokens} tokens (max: {self.reader.max_length})")
                        logger.debug(f"[Truncated Token IDs #{i}]: {input_ids}")
                    else:
                        logger.info(f"[No Truncation #{i}] Final input token length: {actual_tokens}")


            else:
                prompt_template = answer_prompt_template_for_quality_dataset if self.dataset == "quality" else answer_prompt_template
                prompt = prompt_template.format(query=query, context=context)
                inputs = self.reader.tokenizer_encode([prompt])
            generated_token_ids, _ = self.reader.generate(inputs)
            generated_text = self.reader.tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)[0]
            logger.info(f"|> The generated_text is:  {generated_text}")

            answer = self.parse_reader_answers(generated_text)
            logger.info(f"|> The parsed final answer  is:  {answer}")

        context_length = self.count_tokens(context)
        return {"answer": answer, "context_length": context_length}

    def parse_reader_answers(self, text: str):
        candidate_answers = text.split("\n")
        answer = ""
        i = 0
        while len(answer) < 1 and i < len(candidate_answers):
            answer = candidate_answers[i].strip()
            i += 1
        if "answer is" in answer:
            idx = answer.find("answer is")
            answer = answer[idx + len("answer is"):].strip()
            if answer.startswith(":"):
                answer = answer[1:].strip()
        return answer

    def parse_reader_rerank_scores(self, text: str):
        pattern = r'Doc: (\d+), Relevance Score: (\d+)'
        matches = re.findall(pattern, text)
        ratings = [RatingScore(document=int(match[0]), relevance_score=float(match[1])) for match in matches]
        return MultiRatingScore(ratings=ratings)

    # Utility methods
    def get_generate_answer_context(self, text_list):
        return "\n".join([f"{i + 1}. {text}" for i, text in enumerate(text_list)])

    def count_tokens(self, context):
        context = "\n".join(context)
        encodings = self.token_encoder.encode(context)
        return len(encodings)

    def trim_context_by_tokens(self, context, maximum_tokens=1000):
        logger.info(f"We are going to trim the context by {maximum_tokens} tokens")
        context = "\n".join(context)
        encodings = self.token_encoder.encode(context)
        num_tokens = len(encodings)
        if num_tokens > maximum_tokens:
            context = self.token_encoder.decode(encodings[:maximum_tokens])
        return context.split("\n")

    # Reranking methods
    def generate_rerank_scores(self, input_data, use_reader_model=True):
        if not use_reader_model or self.reader_model == "gpt-4o-mini":
            prompt_template = PromptTemplate(
                input_variables=["context_str", "query"],
                template=batch_rerank_template
            )
            llm_chain = prompt_template | self.llm.with_structured_output(MultiRatingScore)
            structured_output = llm_chain.invoke(input_data)
        else:
            context = input_data["context_str"]
            query = input_data["query"]
            if "instruct" in self.reader_model:
                instruction = batch_rerank_instruction
                input_text = batch_rerank_input_template.format(context_str=context, query=query)
                prompt_chat_template = self.reader.get_generator_prompts_chat_format([instruction], [input_text])
                inputs = self.reader.tokenizer_encode_chat_format(prompt_chat_template)
            else:
                raise NotImplementedError(f"batch_rerank using f{self.reader_model} is not implemented!")
            generated_token_ids, _ = self.reader.generate(inputs)
            generated_text = self.reader.tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)[0]
            structured_output = self.parse_reader_rerank_scores(generated_text)
        return structured_output

    def rerank_documents_batch(self, query: str, docs: List[Document], top_r: int = 3, batch_num: int = 10) -> List[
        Document]:
        # First we need to split the documents into batches
        doc_batches = [docs[i:i + batch_num] for i in range(0, len(docs), batch_num)]
        doc_batches_str = []
        for i, doc_batch in enumerate(doc_batches):
            context_str = ""
            for j, doc in enumerate(doc_batch):
                context_str += f"Document {i * batch_num + j}:\n{doc.page_content}\n\n"
            doc_batches_str.append(context_str)
        scored_docs = []
        for doc_batch_str in doc_batches_str:
            input_data = {"context_str": doc_batch_str, "query": query}
            structured_output = self.generate_rerank_scores(
                input_data=input_data,
                use_reader_model=True
            )
            logger.info(structured_output)
            for rating in structured_output.ratings:
                doc_id = rating.document
                score = rating.relevance_score
                scored_docs.append((docs[doc_id], score))
        reranked_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in reranked_docs[:top_r]]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--norm_params",
                        type=int, default=None, help="root dir")
    
    parser.add_argument("--normalize_method",
                        type=str, default=None, help="root dir")
    parser.add_argument("--out_dir",
                        type=str, default=None, help="root dir")
    parser.add_argument("--weights_path",
                        type=str, default=None, help="root dir")
[REDACTED]
[REDACTED]
    parser.add_argument("--memory_repr",
                        type=str,
                        choices=['raw_documents',
                                 'chunks',
                                 'triples',
                                 'atomic_facts',
                                 'summary',
                                 'mix',
                                 'mix2'],
                        default='mix',
                        help="memory_representation")

    parser.add_argument("--memory_reprs",
                        type=str,
                        default='summary,chunks,triples,atomic_facts',
                        help="Comma-separated list of memory representations to use")

    parser.add_argument("--memory_repr_weights",
                        type=str,
                        default='0.25,0.25,0.25,0.25',
                        help="Comma-separated list of weights for each memory representation")
    
    parser.add_argument("--manual_ratio",
                        type=str,
                        default="router",
                        choices=['router',
                                 'manual',
                                 'best1',
                                 'best2',
                                 'best3',
                                 'best4'],
                        help="if mannally set the memory ratio")

    parser.add_argument("--dataset",
                        type=str,
                        choices=['hotpotqa',
                                 'musique',
                                 '2wikimultihopqa',
[REDACTED]
[REDACTED]
                                 'quality'],
                        default='2wikimultihopqa',
                        help='The qa dataset that you want to experiments')
    parser.add_argument("--ablation_type",
                        type=str,
                        choices=[
                            'only_memory_units',
                            'memory_units_with_top_k',
                            'memory_units_with_top_k_with_rerank_top_r',
                            'cluster_summary_with_raw_documents',
                            'memory_units_with_common_cluster',
                            'hm_memory_units_with_top_k',
                            'hm_iter_memory_units_with_top_k',
                        ],
                        default='memory_units_with_top_k', )
    parser.add_argument("--answer_with_memory_units",
                        type=int,
                        choices=[0, 1],
                        default=1,
                        help='Answer with memory units or its corresponding raw documents')
    parser.add_argument("--use_iterative_retrieval",
                        type=int,
                        choices=[0, 1],
                        default=0,
                        help='Whether use iterative retrieval')
    parser.add_argument("--top_k", type=int, default=100, help="Top k memory units to retrieve")
    parser.add_argument("--top_r", type=int, default=0, help="Top r memory units to rerank"),
    parser.add_argument("--top_s", type=int, default=0, help="Top s clusters summary to retrieve")
    parser.add_argument("--top_t", type=int, default=0, help='Top r memory units to retrieve in each iteration')
    parser.add_argument("--num_turns", type=int, default=0, help="Number of turns for iterative retrieval")
    parser.add_argument("--maximum_tokens", type=int, default=4096, help="Maximum tokens for the context")
    parser.add_argument("--debug", type=int, default=0, choices=[0, 1], help="1 represents debug mode")

    parser.add_argument("--reader_model",
                        type=str,
                        choices=['gpt-4o-mini',
                                 'llama3_8b_instruct',
                                 'llama3.1_8b_instruct',
                                 'llama3.1_70b_instruct',
                                 'qwen2.5_7b_instruct',
                                 'qwen2.5_32b_instruct',
                                 'qwen2.5_72b_instruct',
                                 'gemma2_9b_instruct',
                                 "gemma2_27b_instruct"],
                        default="gpt-4o-mini",
                        help="LLM models used to generate answer")
    parser.add_argument("--num_noise_docs", type=int, default=0, help="Number of noise documents to add")

    args = parser.parse_args()

[REDACTED]

    args_dict = vars(args)
    logger.info("|> Running with the following parameters:")
    for key, value in args_dict.items():
        logger.info(f"|> {key}: {value}")


    if args_dict['dataset'] == "hotpotqa":
[REDACTED]
    elif args_dict['dataset'] == 'musique':
[REDACTED]
    elif args_dict['dataset'] == '2wikimultihopqa':
[REDACTED]
[REDACTED]
        load_data_file = [REDACTED]
[REDACTED]
[REDACTED]
    elif args_dict['dataset'] == 'quality':
[REDACTED]
    else:
        raise ValueError("Dataset not supported.")

    dataset = args.dataset

    memory_reprs = args.memory_reprs.split(',')
    memory_repr_weights = [float(w) for w in args.memory_repr_weights.split(',')]

    top_k = args.top_k
    top_r = args.top_r
    top_s = args.top_s
    top_t = args.top_t
    num_turns = args.num_turns
    ablation_type = args.ablation_type
    manual_ratio = args.manual_ratio
    
    answer_with_memory_units = args.answer_with_memory_units
    use_iterative_retrieval = args.use_iterative_retrieval
    maximum_tokens = args.maximum_tokens
    debug_mode = args.debug
    reader_model_name = args.reader_model
    num_noise_docs = args.num_noise_docs

    # results_path = f"./hm_average025_results/{dataset}"
    results_path = os.path.join(args.out_dir, dataset)

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    save_file_name = f"{results_path}/HM_reader_model_[{reader_model_name}]_topk_{top_k}_num_turns_{num_turns}_answer_with_memory_units_{answer_with_memory_units}_use_iterative_retrieval_{use_iterative_retrieval}_maximum_tokens_{maximum_tokens}.csv"
    logger.info(f"|> The save file path is located in {save_file_name}")


#------------------------------------------------------------------------------------------------------------------------------------------
    # if os.path.exists(save_file_name):
    #     logger.info(f"Loading existing results from {save_file_name}")
    #     df = pd.read_csv(save_file_name)
    #     processed_questions = df.loc[df['predictions'] != 'BLANK'].shape[0]
    #     logger.info(f"|> {processed_questions} questions have been processed. Resuming from the next question.")

    # else:
    #     df = pd.read_csv(load_data_file)
    #     df['predictions'] = ["BLANK"] * df.shape[0]
    #     df['total_tokens'] = ["BLANK"] * df.shape[0]
    #     df['total_prompt_tokens'] = ["BLANK"] * df.shape[0]
    #     df['total_completion_tokens'] = ["BLANK"] * df.shape[0]
    #     df['total_cost'] = ["BLANK"] * df.shape[0]
    df = pd.read_csv(load_data_file)
    df['predictions'] = ["BLANK"] * df.shape[0]
    df['total_tokens'] = ["BLANK"] * df.shape[0]
    df['total_prompt_tokens'] = ["BLANK"] * df.shape[0]
    df['total_completion_tokens'] = ["BLANK"] * df.shape[0]
    df['total_cost'] = ["BLANK"] * df.shape[0]




    # ---- ADDED FOR LATENCY/PARAM ANALYSIS ----
    total_predict_weights_time = 0
    predict_weights_calls = 0
    total_generate_answer_time = 0
    generate_answer_calls = 0
    base_model_total_params = 0
    # ---- END OF ADDITIONS ----



    num_of_questions = df.shape[0] if debug_mode != 1 else 1

    if args.reader_model == "gpt-4o-mini":
        reader = None
    else:
        if args.reader_model in ["llama3.1_70b_instruct", "qwen2.5_32b_instruct", "qwen2.5_72b_instruct",
                                 "gemma2_27b_instruct"]:
            reader_tokenizer, reader_llm = load_llm_tokenizer_and_model(args.reader_model, load_in_4bit=True)
        else:
            reader_tokenizer, reader_llm = load_llm_tokenizer_and_model(args.reader_model, load_in_4bit=False)
        #reader = Generator(reader_tokenizer, reader_llm, max_length=32768, max_new_tokens=64)

        if reader_llm: # If a local model is loaded
            base_model_total_params = sum(p.numel() for p in reader_llm.parameters())
            logger.info(f"Loaded reader model {args.reader_model} with {base_model_total_params} parameters.")
            logger.info(f"The context length for the reader model is:  {reader_llm.config.max_position_embeddings}")
        else:
            logger.warning(f"Reader LLM {args.reader_model} could not be loaded. Base parameter count will be 0.")
    
    logger.info(f"The context length for the reader model is:  {reader_llm.config.max_position_embeddings}")
    memobox = MemoBox(dataset=dataset, weights_path = args.weights_path, reader_model=args.reader_model, reader=reader_llm, tokenizer=reader_tokenizer)
    
    for qa_order in tqdm(range(0, num_of_questions)):

        logger.info(f"Processing question order {qa_order}")

        # try:
        # current_prediction = df.at[qa_order, 'predictions']
        # if current_prediction != "BLANK":
        #     logger.info(f"Question {qa_order} has already been processed. Skipping...")
        #     continue

        #memobox = MemoBox(dataset=dataset, weights_path = args.weights_path, reader_model=args.reader_model, reader=reader, tokenizer=reader_tokenizer)

        memobox.load_dataset(memory_reprs, dataset, qa_order, num_noise_docs)

        logger.info(f"Answering questions for order {qa_order}")
        query = df['question'].tolist()[qa_order]
        gold_answer = df['answer'].tolist()[qa_order]
        assert "formatted_sentences" in df.columns, "formatted_sentences column is not in the dataframe"
        memobox.raw_documents = ast.literal_eval(df['formatted_sentences'].tolist()[qa_order])



        memory_reprs = args.memory_reprs.split(',')
        #memory_repr_weights = [float(w) for w in args.memory_repr_weights.split(',')]

        norm_method = args.normalize_method
        norm_params = args.norm_params
        
        if args.manual_ratio == "manual":
            memory_repr_weights = [float(w) for w in args.memory_repr_weights.split(',')]
        
        elif args.manual_ratio == "best1":

            memory_repr_weight = memobox.predict_weights(question=query,norm_method=norm_method,norm_params=norm_params)
            best_repr = max(memory_repr_weight, key=memory_repr_weight.get)
            memory_repr_weights = [1.0 if memo == best_repr else 0.0 for memo in memory_reprs]      

        elif args.manual_ratio == "router":

            start_time_pw = time.time()
            # predict_weights returns a dict. Keys are 'triples', 'atomic_facts', 'summary', 'chunks'
            memory_repr_weight = memobox.predict_weights(question=query,norm_method=norm_method,norm_params=norm_params)

            total_predict_weights_time += (time.time() - start_time_pw)
            predict_weights_calls += 1

            memory_repr_weights = [memory_repr_weight[memo] for memo in memory_reprs]



        start_time_ga = time.time()
        with get_openai_callback() as cb:

            if ablation_type == 'only_memory_units':
                logger.info(f"We are using only memory units")
                final_answer = memobox.query_with_only_memory_units(query=query,
                                                                    maximum_tokens=maximum_tokens)
            elif ablation_type == 'hm_iter_memory_units_with_top_k':
                logger.info(f"We are using memory units with top k")
                final_answer = memobox.query_with_top_k_memory_units(query=query,
                                                                     top_k=top_k,
                                                                     top_t=top_t,
                                                                     num_turns=num_turns,
                                                                     answer_with_memory_units=answer_with_memory_units,
                                                                     maximum_tokens=maximum_tokens,
                                                                     use_iterative_retrieval=use_iterative_retrieval,
                                                                     memory_reprs=memory_reprs,
                                                                     memory_repr_weights=memory_repr_weights,
                                                                    )


            elif ablation_type == 'hm_memory_units_with_top_k':

                # memory_reprs = args.memory_reprs.split(',')
                # #memory_repr_weights = [float(w) for w in args.memory_repr_weights.split(',')]


                # if args.manual_ratio == "manual":
                #     memory_repr_weights = [float(w) for w in args.memory_repr_weights.split(',')]
                
                # elif args.manual_ratio == "best1":

                #     memory_repr_weight = memobox.predict_weights(question=query)
                #     best_repr = max(memory_repr_weight, key=memory_repr_weight.get)
                #     memory_repr_weights = [1.0 if memo == best_repr else 0.0 for memo in memory_reprs]      

                # elif args.manual_ratio == "router":
                #     memory_repr_weight = memobox.predict_weights(question=query)
                #     memory_repr_weights = [memory_repr_weight[memo] for memo in memory_reprs]

                # Validate inputs
                if len(memory_reprs) != len(memory_repr_weights):
                    raise ValueError("Number of memory representations must match number of weights")

                # Use the mixed retrieval in your code
                final_answer = memobox.hm_query_with_top_k_memory_units(
                    query=query,
                    top_k=top_k,
                    memory_reprs=memory_reprs,
                    memory_repr_weights=memory_repr_weights,
                )


            elif ablation_type == 'memory_units_with_top_k_with_rerank_top_r':
                logger.info(f"We are using memory units with top k with rerank top r")
                final_answer = memobox.query_with_top_k_with_rerank_top_r_memory_units(query=query,
                                                                                       top_k=top_k,
                                                                                       top_r=top_r,
                                                                                       top_t=top_t,
                                                                                       num_turns=num_turns,
                                                                                       use_iterative_retrieval=use_iterative_retrieval,
                                                                                       answer_with_memory_units=answer_with_memory_units,
                                                                                       maximum_tokens=maximum_tokens,
                                                                                       memory_reprs=memory_reprs,
                                                                                       memory_repr_weights=memory_repr_weights,
                                                                                       )
            elif ablation_type == 'cluster_summary_with_raw_documents':
                logger.info(f"We are using cluster summary with raw documents")
                final_answer = memobox.query_with_cluster_with_raw_documents(query=query)
            elif ablation_type == 'memory_units_with_common_cluster':
                logger.info(f"We are using memory units with common cluster")
                final_answer = memobox.query_with_memory_units_with_common_cluster(query=query,
                                                                                   top_k=top_k,
                                                                                   top_s=top_s,
                                                                                   top_t=top_t,
                                                                                   num_turns=num_turns,
                                                                                   maximum_tokens=maximum_tokens,
                                                                                   use_iterative_retrieval=use_iterative_retrieval,
                                                                                   answer_with_memory_units=answer_with_memory_units)



            total_generate_answer_time += (time.time() - start_time_ga)
            generate_answer_calls += 1



            logger.info(f"Predicted Answer: {final_answer['answer']} | Gold Answer: {gold_answer}")
            logger.info(f"Total Tokens: {cb.total_tokens}")
            logger.info(f"Prompt Tokens: {cb.prompt_tokens}")
            logger.info(f"Completion Tokens: {cb.completion_tokens}")
            logger.info(f"Total Cost (USD): ${cb.total_cost}")
            logger.info(f"Context Length: {final_answer['context_length']}")

            df.at[qa_order, 'predictions'] = final_answer['answer']
            df.at[qa_order, 'total_tokens'] = cb.total_tokens
            df.at[qa_order, 'total_prompt_tokens'] = cb.prompt_tokens
            df.at[qa_order, 'total_completion_tokens'] = cb.completion_tokens
            df.at[qa_order, 'total_cost'] = cb.total_cost
            df.at[qa_order, 'context_length'] = final_answer['context_length']

            df.to_csv(save_file_name, index=False)

    # ---- ADDED FOR LATENCY/PARAM ANALYSIS LOGGING (after the loop) ----
    logger.info("--------------------------------------------------------------")
    logger.info("-------------- LATENCY AND PARAMETER ANALYSIS --------------")
    logger.info("--------------------------------------------------------------")

    # Latency Analysis
    avg_predict_weights_time = (total_predict_weights_time / predict_weights_calls) if predict_weights_calls > 0 else 0
    avg_generate_answer_time = (total_generate_answer_time / generate_answer_calls) if generate_answer_calls > 0 else 0


    logger.info("[Latency]")
    logger.info(f"Total time for 'predict_weights' calls: {total_predict_weights_time:.4f} seconds")
    logger.info(f"Number of 'predict_weights' calls: {predict_weights_calls}")
    logger.info(f"Average time per 'predict_weights' call: {avg_predict_weights_time:.4f} seconds")
    
    logger.info(f"Total time for answer generation: {total_generate_answer_time:.4f} seconds")
    logger.info(f"Number of answer generation calls: {generate_answer_calls}")
    logger.info(f"Average time per answer generation call: {avg_generate_answer_time:.4f} seconds")
    if generate_answer_calls > 0 and predict_weights_calls > 0 and total_generate_answer_time > 0 :
        router_overhead_percentage = (avg_predict_weights_time / (avg_generate_answer_time/generate_answer_calls)) * 100 if (avg_generate_answer_time/generate_answer_calls) > 0 else "N/A"
        if isinstance(router_overhead_percentage, float):
             logger.info(f"Average 'predict_weights' time as a percentage of average total per-question pipeline time (predict_weights + generate_answer): {(avg_predict_weights_time / (avg_predict_weights_time + avg_generate_answer_time)) * 100 if (avg_predict_weights_time + avg_generate_answer_time) > 0 else 'N/A':.2f}%")


    # Parameter Analysis
    logger.info("\n[Parameter Sizes]")
    # base_model_total_params is already calculated when reader_llm is loaded
    logger.info(f"Base Model for Router/Reader ('{args.reader_model}'): {base_model_total_params} parameters")
    
    # Regression head and LoRA parameters are stored in memobox instance
    regression_params = memobox.regression_head_params if hasattr(memobox, 'regression_head_params') else "N/A"
    lora_params_val = memobox.lora_params if hasattr(memobox, 'lora_params') else "N/A"
    
    logger.info(f"Regression Head (Router): {regression_params} parameters")
    logger.info(f"LoRA Block (Router): {lora_params_val} parameters")

    if isinstance(regression_params, int) and isinstance(lora_params_val, int):
        total_router_additional_params = regression_params + lora_params_val
        logger.info(f"Total Additional Parameters for Router (LoRA + Regression Head): {total_router_additional_params} parameters")
        if base_model_total_params > 0:
            percentage_overhead = (total_router_additional_params / base_model_total_params) * 100
            logger.info(f"Router parameter overhead compared to base model: {percentage_overhead:.2f}%")
        else:
            logger.info("Base model parameters are 0 or N/A, cannot calculate percentage overhead for router.")
    else:
        logger.info("Could not calculate total additional router parameters due to N/A values.")
        
    logger.info("--------------------------------------------------------------")
    # ---- END OF ADDITIONS ----